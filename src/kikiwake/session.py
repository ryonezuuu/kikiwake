"""Per-audio session (``<stem>.session.json``): the cached Avalon response plus each speaker's state."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast, get_args

from kikiwake.errors import UsageError
from kikiwake.models import DiarizedTranscription, parse_diarized
from kikiwake.speaker_db import atomic_write_text

SESSION_VERSION = 1

Status = Literal["auto", "candidate", "unknown", "insufficient", "confirmed"]
_STATUSES: tuple[str, ...] = get_args(Status)


@dataclass
class SpeakerState:
    id: str
    number: int
    segments: list[tuple[float, float]]
    total_seconds: float
    centroid: list[float] | None
    consistency: float | None
    scores: dict[str, float]
    status: Status
    name: str | None
    score: float | None
    db_sample_ids: list[str]

    @property
    def display_label(self) -> str:
        if self.name is not None:
            if self.status in ("auto", "confirmed"):
                return self.name
            if self.status == "candidate":
                return f"{self.name}？"
        return f"話者{self.number}"


@dataclass
class Session:
    audio_path: str
    audio_sha256: str
    created_at: str
    model_id: str
    avalon_response: dict
    speakers: dict[str, SpeakerState]
    warnings: list[dict]
    recorded_at: str | None = None  # when the meeting was recorded (ISO 8601, local time)
    version: int = SESSION_VERSION

    @property
    def transcription(self) -> DiarizedTranscription:
        return parse_diarized(self.avalon_response)


def session_paths(audio: Path, output_dir: Path | None) -> tuple[Path, Path, Path]:
    """Output files for ``audio``: ``(<stem>.md, <stem>.session.json, <stem>.voices.png)``."""
    out = audio.parent if output_dir is None else output_dir
    return (
        out / f"{audio.stem}.md",
        out / f"{audio.stem}.session.json",
        out / f"{audio.stem}.voices.png",
    )


def _speaker_to_json(st: SpeakerState) -> dict[str, Any]:
    data = asdict(st)
    del data["id"]  # the key of ``speakers``
    data["display_label"] = st.display_label  # for readers; recomputed on load
    return data


def save_session(session: Session, path: Path) -> None:
    data = {
        "version": session.version,
        "audio_path": session.audio_path,
        "audio_sha256": session.audio_sha256,
        "created_at": session.created_at,
        "recorded_at": session.recorded_at,
        "model_id": session.model_id,
        "avalon_response": session.avalon_response,
        "speakers": {sid: _speaker_to_json(st) for sid, st in session.speakers.items()},
        "warnings": session.warnings,
    }
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _optional(raw: dict[str, Any], key: str, kind: type) -> Any:
    value = raw[key]
    return None if value is None else kind(value)


def _speaker_from_json(sid: str, raw: dict[str, Any]) -> SpeakerState:
    status = raw["status"]
    if status not in _STATUSES:
        raise ValueError(f"unknown status '{status}'")
    centroid = raw["centroid"]
    return SpeakerState(
        id=sid,
        number=int(raw["number"]),
        segments=[(float(start), float(end)) for start, end in raw["segments"]],
        total_seconds=float(raw["total_seconds"]),
        centroid=None if centroid is None else [float(x) for x in centroid],
        consistency=_optional(raw, "consistency", float),
        scores={str(name): float(score) for name, score in raw["scores"].items()},
        status=cast(Status, status),
        name=_optional(raw, "name", str),
        score=_optional(raw, "score", float),
        db_sample_ids=[str(i) for i in raw["db_sample_ids"]],
    )


def load_session(path: Path) -> Session:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise UsageError(f"{path}: セッションを読み込めません: {e}") from e
    try:
        data = json.loads(text)
        version = data["version"]
        if version != SESSION_VERSION:
            raise UsageError(f"{path}: 未対応のセッションバージョンです: {version!r}")
        avalon_response = data["avalon_response"]
        if not isinstance(avalon_response, dict):
            raise TypeError("'avalon_response' must be an object")
        recorded_at = data.get("recorded_at")  # absent in sessions saved before the key existed
        if recorded_at is not None and not isinstance(recorded_at, str):
            raise TypeError("'recorded_at' must be a string or null")
        return Session(
            audio_path=str(data["audio_path"]),
            audio_sha256=str(data["audio_sha256"]),
            created_at=str(data["created_at"]),
            model_id=str(data["model_id"]),
            avalon_response=avalon_response,
            speakers={
                sid: _speaker_from_json(sid, raw) for sid, raw in data["speakers"].items()
            },
            warnings=list(data["warnings"]),
            recorded_at=recorded_at,
            version=version,
        )
    except UsageError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        # ValueError covers invalid JSON; the rest cover a wrong shape.
        raise UsageError(f"{path}: セッションの形式が不正です: {type(e).__name__}: {e}") from e
