"""The ``transcribe`` flow: Avalon response (fresh or cached) -> voiceprints -> matching -> session and Markdown."""

import re
import sys
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from kikiwake.audio import (
    load_wav,
    probe_duration,
    require_ffmpeg,
    sha256_file,
    to_upload_mp3,
    to_wav16k,
)
from kikiwake.config import Settings
from kikiwake.embedder import Embedder
from kikiwake.errors import AppError, UsageError
from kikiwake.labeling import apply_label
from kikiwake.matcher import assign, person_score, score_table
from kikiwake.models import parse_diarized
from kikiwake.render import render_markdown
from kikiwake.session import (
    Session,
    SpeakerState,
    load_session,
    save_session,
    session_paths,
)
from kikiwake.speaker_db import Sample, SpeakerDB, atomic_write_text
from kikiwake.voiceprint import build_voiceprint

_TOO_LONG = "録音が 4 時間を超えています（上限 4 時間）"
_OPTIONS_IGNORED = (
    "保存済みの文字起こしを再利用するため --language/--prompt/--max-speakers は反映されません"
    "（反映するには --force-api）"
)


@dataclass
class TranscribeOptions:
    output_dir: Path | None = None
    language: str | None = None
    prompt: str | None = None
    max_speakers: int | None = None
    force_api: bool = False


# Takes the MP3 to upload and returns the raw Avalon ``diarized_json`` response.
Transcriber = Callable[[Path, TranscribeOptions], dict]

# Locks the voiceprint DB and yields it freshly loaded; the lock is held until the block ends.
DBOpener = Callable[[], AbstractContextManager[SpeakerDB]]


def _progress(message: str) -> None:
    print(message, file=sys.stderr)


def check_audio(audio: Path) -> None:
    """Fail with a usage error unless ffmpeg is installed and ``audio`` is a file."""
    require_ffmpeg()
    if not audio.is_file():
        raise UsageError(f"音声ファイルが見つかりません: {audio}")


_VOICE_MEMO_STEM = re.compile(r"^(\d{8}) (\d{6})")  # e.g. "20261016 125520-357FE498"


def recorded_at_for(audio: Path) -> str:
    """When ``audio`` was recorded: ISO 8601 in local time, to the second.

    A Voice Memos name (``YYYYMMDD HHMMSS-...``) holds the time; otherwise (or if the name is not a
    real date and time) the file's modification time is used.
    """
    found = _VOICE_MEMO_STEM.match(audio.stem)
    if found is not None:
        try:
            when = datetime.strptime(" ".join(found.groups()), "%Y%m%d %H%M%S")
        except ValueError:
            pass  # not a real date and time
        else:
            return when.astimezone().isoformat(timespec="seconds")
    return datetime.fromtimestamp(audio.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _load_previous(path: Path, sha: str) -> Session | None:
    """The session saved at ``path`` if it is readable and made from the same audio."""
    if not path.exists():
        return None
    try:
        prev = load_session(path)
    except UsageError as e:
        _progress(f"警告: 既存のセッションを読み込めないため無視します: {e}")
        return None
    return prev if prev.audio_sha256 == sha else None


def _relabel(session: Session, db: SpeakerDB, speaker_id: str, name: str) -> None:
    for warning in apply_label(session, db, speaker_id, name):
        _progress(f"警告: {warning}")


def _restore_labels(
    session: Session,
    db: SpeakerDB,
    prev: Session,
    taken_back: dict[str, list[Sample]],
    known: set[str],
) -> None:
    """Name again the speakers ``prev`` had confirmed (the cached response keeps the speaker ids).

    A name follows its DB sample to whoever owns it now (``speakers rename``), and a person deleted
    since is not brought back. A speaker without a voiceprint, or whose sample was evicted from a
    person still in the DB (``known``), keeps the name it had.
    """
    owners = {sample.id: name for name, samples in taken_back.items() for sample in samples}
    for sid, old in prev.speakers.items():
        if old.status != "confirmed" or old.name is None or sid not in session.speakers:
            continue
        current = [owners[i] for i in old.db_sample_ids if i in owners]
        if current:
            _relabel(session, db, sid, current[0])
        elif not old.db_sample_ids or old.name in known:
            _relabel(session, db, sid, old.name)


def _carry_over_labels(
    session: Session, db: SpeakerDB, taken_back: dict[str, list[Sample]], settings: Settings
) -> list[str]:
    """Give each name this recording was labelled with to the new speaker who sounds most like it.

    Used when the earlier session is not at hand, so speaker ids cannot be matched up: each person's
    taken-back ``label`` samples are compared with the new voiceprints. Returns the names left over
    (no speaker at ``auto_threshold``, or that speaker already got another name).
    """
    lost: list[str] = []
    for name, samples in taken_back.items():
        labelled = [s.embedding for s in samples if s.source == "label"]
        if not labelled:
            continue
        similarity = {
            sid: person_score(np.asarray(st.centroid, dtype=np.float32), labelled, 1)
            for sid, st in session.speakers.items()
            if st.centroid is not None
        }
        best = max(similarity, key=similarity.__getitem__, default=None)
        if (
            best is not None
            and similarity[best] >= settings.auto_threshold
            and session.speakers[best].status != "confirmed"
        ):
            _relabel(session, db, best, name)
        else:
            lost.append(name)
    return lost


def run_transcribe(
    audio: Path,
    opts: TranscribeOptions,
    *,
    settings: Settings,
    open_db: DBOpener,
    embedder: Embedder,
    transcriber: Transcriber,
    now: Callable[[], str],
) -> tuple[Session, Path, Path]:
    """Transcribe ``audio`` and name its speakers; return ``(session, markdown_path, session_path)``.

    The DB is opened (locked and loaded) only once the transcript and the voiceprints are ready,
    so a ``label`` run while the API is working is not overwritten; it is saved before the lock ends.
    """
    check_audio(audio)
    sha = sha256_file(audio)
    recorded_at = recorded_at_for(audio)  # decided afresh on every run, while the audio is there
    markdown_path, session_path, _ = session_paths(audio, opts.output_dir)
    prev = _load_previous(session_path, sha)
    cached = None if opts.force_api else prev  # the session whose Avalon response may be reused

    if probe_duration(audio) > settings.max_duration:
        raise AppError(_TOO_LONG)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        to_wav16k(audio, wav_path)
        if cached is not None:
            _progress("キャッシュを再利用します")
            if (opts.language, opts.prompt, opts.max_speakers) != (None, None, None):
                _progress(_OPTIONS_IGNORED)  # they only steer the API call
            raw = cached.avalon_response
        else:
            mp3_path = Path(tmp) / "upload.mp3"
            to_upload_mp3(audio, mp3_path)
            _progress("文字起こし中…")
            raw = transcriber(mp3_path, opts)
            # Keep the paid response at once: if anything below fails, the next run reuses it.
            save_session(
                Session(
                    audio_path=str(audio),
                    audio_sha256=sha,
                    created_at=now(),
                    model_id=embedder.model_id,
                    avalon_response=raw,
                    speakers={},
                    warnings=[],
                    recorded_at=recorded_at,
                ),
                session_path,
            )
        wave = load_wav(wav_path)

    tr = parse_diarized(raw)

    _progress("声紋を照合中…")
    voiceprints = {
        sid: build_voiceprint(wave, tr, sid, embedder, settings) for sid in tr.speaker_ids
    }
    centroids = {sid: vp.centroid for sid, vp in voiceprints.items() if vp.centroid is not None}

    with open_db() as db:
        if cached is not None:
            # A `label` may have changed the session since it was read; under the lock it cannot.
            latest = _load_previous(session_path, sha)
            if latest is not None and latest.avalon_response == raw:
                prev = cached = latest
        known = set(db.people())
        # Take back this recording's earlier samples first, so it cannot match its own voices
        # and a rerun cannot add them twice.
        taken_back = db.remove_by_session(sha)
        scores = score_table(centroids, db.embeddings(), settings.top_k)
        matches = assign(scores, settings)

        speakers: dict[str, SpeakerState] = {}
        for number, sid in enumerate(tr.speaker_ids, start=1):
            vp = voiceprints[sid]
            sample_ids: list[str] = []
            if vp.centroid is None:
                status, name, score = "insufficient", None, None
            else:
                match = matches[sid]
                status, name, score = match.status, match.name, match.score
                if status == "auto" and name is not None:
                    sample_ids = [db.add_sample(name, vp.centroid, sha, "auto")]
            speakers[sid] = SpeakerState(
                id=sid,
                number=number,
                segments=vp.segments,
                total_seconds=vp.total_seconds,
                centroid=None if vp.centroid is None else vp.centroid.tolist(),
                consistency=vp.consistency,
                scores=scores.get(sid, {}),
                status=status,
                name=name,
                score=score,
                db_sample_ids=sample_ids,
            )

        warnings: list[dict] = []
        if "speaker_limit_reached" in tr.warnings:
            warnings.append({"code": "speaker_limit_reached"})
        for sid, vp in voiceprints.items():
            if vp.consistency is not None and vp.consistency < settings.min_consistency:
                warnings.append({"code": "low_consistency", "speaker": sid})
        if (
            prev is not None
            and opts.force_api
            and any(st.status == "confirmed" for st in prev.speakers.values())
        ):
            warnings.append({"code": "labels_discarded"})

        session = Session(
            audio_path=str(audio),
            audio_sha256=sha,
            created_at=now(),
            model_id=embedder.model_id,
            avalon_response=raw,
            speakers=speakers,
            warnings=warnings,
            recorded_at=recorded_at,
        )

        if prev is None or not prev.speakers:
            # No earlier session here (the audio moved, or only the response was kept after a
            # failure), so labels this recording had are found by voice instead.
            lost = _carry_over_labels(session, db, taken_back, settings)
            if lost:
                session.warnings.append({"code": "labels_lost", "names": lost})
        elif cached is not None:
            # With the cached response the speaker ids are unchanged, so earlier names can be restored.
            _restore_labels(session, db, cached, taken_back, known)

        # The session first: it records what was added to the DB, so a rerun can take it back.
        save_session(session, session_path)
        db.save()
        atomic_write_text(markdown_path, render_markdown(session))
    return session, markdown_path, session_path
