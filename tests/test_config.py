from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kikiwake.config import Settings, data_home, load_settings, local_zone
from kikiwake.errors import AppError, UsageError


def test_defaults_match_spec():
    s = Settings()
    assert (s.auto_threshold, s.candidate_threshold) == (0.60, 0.45)
    assert (s.min_segment, s.max_total_per_speaker, s.min_total_per_speaker) == (1.5, 60.0, 4.0)
    assert s.max_samples_per_person == 20 and s.top_k == 3


def test_data_home_env_override(tmp_path):
    assert data_home({"KIKIWAKE_HOME": str(tmp_path)}) == tmp_path


def test_data_home_default():
    assert data_home({"HOME": "/Users/x"}) == Path("/Users/x/.local/share/kikiwake")


def test_load_settings_overrides(tmp_path):
    (tmp_path / "config.toml").write_text("auto_threshold = 0.7\n")
    assert load_settings(tmp_path).auto_threshold == 0.7


def test_load_settings_missing_file(tmp_path):
    assert load_settings(tmp_path) == Settings()


def test_load_settings_unknown_key(tmp_path):
    (tmp_path / "config.toml").write_text("nope = 1\n")
    with pytest.raises(UsageError, match="nope"):
        load_settings(tmp_path)


def test_load_settings_wrong_type(tmp_path):
    (tmp_path / "config.toml").write_text('auto_threshold = "high"\n')
    with pytest.raises(UsageError, match="auto_threshold"):
        load_settings(tmp_path)


def test_exit_codes():
    assert AppError("x").exit_code == 1 and UsageError("x").exit_code == 2


def test_timezone_defaults_to_the_system_zone():
    assert Settings().timezone == ""
    assert local_zone(Settings()) is None  # Python's "system local time" (keeps its DST rules)


@pytest.mark.parametrize("name", ["Asia/Tokyo", "UTC", "America/New_York"])
def test_timezone_setting(tmp_path, name):
    (tmp_path / "config.toml").write_text(f'timezone = "{name}"\n', encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings.timezone == name and local_zone(settings) == ZoneInfo(name)


@pytest.mark.parametrize("name", ["Nowhere/City", "JST", "America", "../etc/passwd", "/etc/localtime"])
def test_unknown_timezone_is_a_usage_error_naming_the_key(tmp_path, name):
    (tmp_path / "config.toml").write_text(f'timezone = "{name}"\n', encoding="utf-8")
    with pytest.raises(UsageError, match=r"config\.toml: setting 'timezone'") as e:
        load_settings(tmp_path)
    assert name in str(e.value)


def test_timezone_must_be_a_string(tmp_path):
    (tmp_path / "config.toml").write_text("timezone = 9\n", encoding="utf-8")
    with pytest.raises(UsageError, match="timezone"):
        load_settings(tmp_path)
