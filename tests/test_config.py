from pathlib import Path

import pytest

from speaker_transcribe.config import Settings, data_home, load_settings
from speaker_transcribe.errors import AppError, UsageError


def test_defaults_match_spec():
    s = Settings()
    assert (s.auto_threshold, s.candidate_threshold) == (0.60, 0.45)
    assert (s.min_segment, s.max_total_per_speaker, s.min_total_per_speaker) == (1.5, 60.0, 4.0)
    assert s.max_samples_per_person == 20 and s.top_k == 3


def test_data_home_env_override(tmp_path):
    assert data_home({"SPEAKER_TRANSCRIBE_HOME": str(tmp_path)}) == tmp_path


def test_data_home_default():
    assert data_home({"HOME": "/Users/x"}) == Path("/Users/x/.local/share/speaker-transcribe")


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
