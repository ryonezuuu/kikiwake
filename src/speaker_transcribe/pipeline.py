"""The ``transcribe`` flow: Avalon response (fresh or cached) -> voiceprints -> matching -> session and Markdown."""

import sys
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from speaker_transcribe.audio import (
    load_wav,
    probe_duration,
    require_ffmpeg,
    sha256_file,
    to_upload_mp3,
    to_wav16k,
)
from speaker_transcribe.config import Settings
from speaker_transcribe.embedder import Embedder
from speaker_transcribe.errors import AppError, UsageError
from speaker_transcribe.labeling import apply_label
from speaker_transcribe.matcher import assign, score_table
from speaker_transcribe.models import parse_diarized
from speaker_transcribe.render import render_markdown
from speaker_transcribe.session import (
    Session,
    SpeakerState,
    load_session,
    save_session,
    session_paths,
)
from speaker_transcribe.speaker_db import SpeakerDB, atomic_write_text
from speaker_transcribe.voiceprint import build_voiceprint

_TOO_LONG = "録音が 4 時間を超えています（上限 4 時間）"


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
    require_ffmpeg()
    if not audio.is_file():
        raise UsageError(f"音声ファイルが見つかりません: {audio}")
    sha = sha256_file(audio)
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
            raw = cached.avalon_response
        else:
            mp3_path = Path(tmp) / "upload.mp3"
            to_upload_mp3(audio, mp3_path)
            _progress("文字起こし中…")
            raw = transcriber(mp3_path, opts)
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
        # Take back this recording's earlier samples first, so it cannot match its own voices
        # and a rerun cannot add them twice.
        db.remove_by_session(sha)
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
        )

        # With the cached response the speaker ids are unchanged, so earlier names can be restored.
        if cached is not None:
            for sid, old in cached.speakers.items():
                if old.status == "confirmed" and old.name is not None and sid in speakers:
                    for warning in apply_label(session, db, sid, old.name):
                        _progress(f"警告: {warning}")

        db.save()
        save_session(session, session_path)
        atomic_write_text(markdown_path, render_markdown(session))
    return session, markdown_path, session_path
