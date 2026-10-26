from dataclasses import dataclass
from typing import Any, NoReturn

from kikiwake.errors import AppError


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker_confidence: float
    overlap: bool


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float
    text: str
    words: tuple[Word, ...]


@dataclass(frozen=True)
class DiarizedTranscription:
    text: str
    language: str
    duration: float
    speaker_ids: tuple[str, ...]  # ordered by speakers[].first_seen ascending
    turns: tuple[Turn, ...]
    warnings: tuple[str, ...]


def _unexpected(path: str) -> NoReturn:
    raise AppError(f"Avalon の応答形式が想定外です: {path}")


def _field(obj: Any, key: str, path: str, kind: type | tuple[type, ...]) -> Any:
    """Return ``obj[key]``, failing with the field's path if absent or not a ``kind``."""
    where = f"{path}.{key}" if path else key
    if not isinstance(obj, dict) or key not in obj:
        _unexpected(where)
    value = obj[key]
    # bool is an int subclass; a JSON true/false is never a valid number.
    if not isinstance(value, kind) or (kind is not bool and isinstance(value, bool)):
        _unexpected(where)
    return value


def _str(obj: Any, key: str, path: str) -> str:
    return _field(obj, key, path, str)


def _float(obj: Any, key: str, path: str) -> float:
    return float(_field(obj, key, path, (int, float)))


def _bool(obj: Any, key: str, path: str) -> bool:
    return _field(obj, key, path, bool)


def _list(obj: Any, key: str, path: str) -> list:
    return _field(obj, key, path, list)


def _parse_word(raw: Any, path: str) -> Word:
    return Word(
        text=_str(raw, "text", path),
        start=_float(raw, "start", path),
        end=_float(raw, "end", path),
        speaker_confidence=_float(raw, "speaker_confidence", path),
        overlap=_bool(raw, "overlap", path),
    )


def _parse_turn(raw: Any, path: str) -> Turn:
    words = _list(raw, "words", path)
    return Turn(
        speaker=_str(raw, "speaker", path),
        start=_float(raw, "start", path),
        end=_float(raw, "end", path),
        text=_str(raw, "text", path),
        words=tuple(_parse_word(w, f"{path}.words[{i}]") for i, w in enumerate(words)),
    )


def parse_diarized(raw: dict) -> DiarizedTranscription:
    """Validate an Avalon ``diarized_json`` response and convert it to typed models."""
    speakers = _list(raw, "speakers", "")
    first_seen = [
        (_float(s, "first_seen", f"speakers[{i}]"), _str(s, "id", f"speakers[{i}]"))
        for i, s in enumerate(speakers)
    ]
    turns = _list(raw, "turns", "")
    warnings = _list(raw, "warnings", "")
    for i, w in enumerate(warnings):
        if not isinstance(w, str):
            _unexpected(f"warnings[{i}]")
    return DiarizedTranscription(
        text=_str(raw, "text", ""),
        language=_str(raw, "language", ""),
        duration=_float(raw, "duration", ""),
        speaker_ids=tuple(sid for _, sid in sorted(first_seen, key=lambda p: p[0])),
        turns=tuple(_parse_turn(t, f"turns[{i}]") for i, t in enumerate(turns)),
        warnings=tuple(warnings),
    )
