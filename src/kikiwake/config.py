import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kikiwake.errors import UsageError


@dataclass(frozen=True)
class Settings:
    min_word_confidence: float = 0.7
    max_word_gap: float = 0.3
    edge_trim: float = 0.1
    min_segment: float = 1.5
    max_total_per_speaker: float = 60.0
    min_total_per_speaker: float = 4.0
    chunk_seconds: float = 3.0
    min_consistency: float = 0.5
    auto_threshold: float = 0.60
    candidate_threshold: float = 0.45
    max_samples_per_person: int = 20
    top_k: int = 3
    job_timeout: float = 3600.0
    poll_initial: float = 5.0
    poll_max: float = 15.0
    max_retries: int = 5
    max_duration: float = 14400.0
    model: str = "avalon-v1.5"
    base_url: str = "https://api.aquavoice.com"
    # IANA name (e.g. "Asia/Tokyo") for the times kikiwake writes; "" is the system's zone
    # (in a Cowork VM likely UTC, hence this setting).
    timezone: str = ""


def local_zone(settings: Settings) -> tzinfo | None:
    """The zone of the times kikiwake writes (``recorded_at``, ``created_at``).

    ``None`` when ``timezone`` is empty: Python's own "system local time" (``datetime.astimezone()``),
    which, unlike a snapshot of today's offset, keeps the system zone's daylight saving rules.
    """
    return ZoneInfo(settings.timezone) if settings.timezone else None


def data_home(env: Mapping[str, str] = os.environ) -> Path:
    override = env.get("KIKIWAKE_HOME")
    if override:
        return Path(override)
    return Path(env["HOME"]) / ".local/share/kikiwake"


def load_settings(home: Path) -> Settings:
    path = home / "config.toml"
    if not path.exists():
        return Settings()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise UsageError(f"{path}: invalid TOML: {e}") from e

    defaults = {f.name: f.default for f in fields(Settings)}
    overrides = {}
    for key, value in raw.items():
        if key not in defaults:
            raise UsageError(f"{path}: unknown setting '{key}'")
        expected = type(defaults[key])
        # bool is an int subclass; a TOML true/false is never a valid number.
        is_number = isinstance(value, int | float) and not isinstance(value, bool)
        if expected is str:
            valid = isinstance(value, str)
        elif expected is int:
            valid = is_number and isinstance(value, int)
        else:  # float fields accept ints as well
            valid = is_number
        if not valid:
            raise UsageError(
                f"{path}: setting '{key}' must be {expected.__name__}, "
                f"got {type(value).__name__}"
            )
        overrides[key] = expected(value)
    zone = overrides.get("timezone", "")
    if zone:
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError, OSError):  # unknown, a path, a directory
            raise UsageError(
                f"{path}: setting 'timezone' must be an IANA time zone name such as "
                f"\"Asia/Tokyo\", got {zone!r}"
            ) from None
    return Settings(**overrides)
