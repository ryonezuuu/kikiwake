"""Name a speaker (``label``): resolve KEYs, validate names, and keep the voiceprint DB in step."""

import numpy as np

from speaker_transcribe.errors import UsageError
from speaker_transcribe.session import Session, SpeakerState
from speaker_transcribe.speaker_db import SpeakerDB


def _fullwidth(text: str) -> str:
    return text.replace("?", "？")


def resolve_key(session: Session, key: str) -> str:
    """Return the speaker id that ``key`` (a speaker id or a display label) names.

    A half-width ``?`` in ``key`` stands for the full-width ``？`` of a candidate's label.
    """
    if key in session.speakers:
        return key
    wanted = _fullwidth(key)
    matches = [sid for sid, st in session.speakers.items() if _fullwidth(st.display_label) == wanted]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        valid = [*session.speakers, *(st.display_label for st in session.speakers.values())]
        raise UsageError(
            f"KEY「{key}」に一致する話者がいません。指定できる KEY: {', '.join(dict.fromkeys(valid))}"
        )
    raise UsageError(
        f"KEY「{key}」は複数の話者（{', '.join(matches)}）に一致します。話者 ID で指定してください"
    )


def _clean_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise UsageError("名前が空です")
    if cleaned.endswith(("？", "?")):
        raise UsageError(f"名前は「？」または「?」で終えられません: {cleaned}")
    return cleaned


def _label(session: Session, db: SpeakerDB, st: SpeakerState, name: str) -> list[str]:
    # A candidate guess for this name on another speaker is superseded by the explicit label.
    for other in session.speakers.values():
        if other is not st and other.status == "candidate" and other.name == name:
            other.status, other.name, other.score = "unknown", None, None

    warnings: list[str] = []
    if st.centroid is None:
        warnings.append(f"{name} は声紋が不足しているため、DB には登録しません")
    else:
        db.remove_samples(st.db_sample_ids)
        sample_id = db.add_sample(
            name, np.array(st.centroid, dtype=np.float32), session.audio_sha256, "label"
        )
        st.db_sample_ids = [sample_id]
    st.name = name
    st.status = "confirmed"
    return warnings


def apply_labels(session: Session, db: SpeakerDB, pairs: list[tuple[str, str]]) -> list[str]:
    """Name the speakers ``pairs`` of ``(KEY, NAME)``; return warnings.

    Every KEY and NAME is checked before anything changes, so a bad pair leaves ``session``
    and ``db`` untouched. The caller saves both.
    """
    resolved = [(resolve_key(session, key), _clean_name(name)) for key, name in pairs]
    ids = [sid for sid, _ in resolved]
    for sid in dict.fromkeys(ids):
        if ids.count(sid) > 1:
            raise UsageError(f"話者 {sid} に複数の名前が指定されています")
    warnings: list[str] = []
    for sid, name in resolved:
        warnings += _label(session, db, session.speakers[sid], name)
    return warnings


def apply_label(session: Session, db: SpeakerDB, speaker_id: str, name: str) -> list[str]:
    """Name one speaker; same as ``apply_labels`` with a single pair."""
    return apply_labels(session, db, [(speaker_id, name)])
