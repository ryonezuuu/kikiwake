import copy
import shutil
import wave
from dataclasses import replace

import numpy as np
import pytest
from conftest import raw_diarized, unit

from speaker_transcribe import pipeline
from speaker_transcribe.audio import SAMPLE_RATE
from speaker_transcribe.config import Settings
from speaker_transcribe.errors import AppError, UsageError
from speaker_transcribe.labeling import apply_labels
from speaker_transcribe.pipeline import TranscribeOptions, run_transcribe
from speaker_transcribe.session import load_session, save_session
from speaker_transcribe.speaker_db import SpeakerDB

DURATION = 30.0
# (speaker, first word start, end, tone frequency): clean speech, 0.5 s per word, no gaps or overlap.
REGIONS = [("spk_0", 1.0, 9.0, 150.0), ("spk_1", 11.0, 19.0, 250.0)]
NOW = "2026-10-26T10:00:00+09:00"


class FreqEmbedder:
    """Embeds by pitch: a peak below 200 Hz is ``unit(0)``, otherwise ``unit(1)``."""

    model_id = "fake"
    dim = 4

    def embed(self, wave: np.ndarray) -> np.ndarray:
        peak = np.argmax(np.abs(np.fft.rfft(wave))) * SAMPLE_RATE / len(wave)
        return unit(0 if peak < 200 else 1)


def write_meeting(path, amplitude: float) -> None:
    t = np.arange(round(DURATION * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for _, start, end, freq in REGIONS:
        inside = (t >= start) & (t < end)
        signal[inside] = amplitude * np.sin(2 * np.pi * freq * t[inside])
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((signal * 32767).astype("<i2").tobytes())


def clean_words(speaker: str, start: float, end: float) -> list[tuple[str, float, float, float, bool]]:
    count = round((end - start) / 0.5)
    return [(speaker, start + 0.5 * i, start + 0.5 * (i + 1), 0.9, False) for i in range(count)]


def meeting_response() -> dict:
    words = [w for speaker, start, end, _ in REGIONS for w in clean_words(speaker, start, end)]
    return raw_diarized(words, duration=DURATION)


class Env:
    def __init__(self, tmp_path):
        self.audio = tmp_path / "meeting.wav"
        write_meeting(self.audio, 0.3)
        self.other_meeting_same_voices = tmp_path / "followup.wav"
        write_meeting(self.other_meeting_same_voices, 0.2)
        self.response = meeting_response()
        self.settings = Settings()
        self.db_path = tmp_path / "db.json"
        self.db = SpeakerDB.load(self.db_path, "fake", 4, self.settings.max_samples_per_person)
        self.calls = 0

    def transcribe(self, mp3, opts):
        assert mp3.suffix == ".mp3" and mp3.stat().st_size > 0
        self.calls += 1
        return copy.deepcopy(self.response)

    def run(self, audio=None, **opts):
        return run_transcribe(
            audio or self.audio,
            TranscribeOptions(**opts),
            settings=self.settings,
            db=self.db,
            embedder=FreqEmbedder(),
            transcriber=self.transcribe,
            now=lambda: NOW,
        )


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def label_first_run(env):
    """Run once, name spk_0 「田中」, and persist both the session and the DB (as the `label` command does)."""
    s1, md, ss = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")])
    save_session(s1, ss)
    env.db.save()
    return s1, md, ss


def test_first_run_writes_outputs_and_unknown_labels(env):
    session, md, ss = env.run()
    assert md.read_text().startswith("# meeting.wav の文字起こし")
    assert [s.display_label for s in session.speakers.values()] == ["話者1", "話者2"]
    assert env.calls == 1
    assert (md, ss) == (env.audio.with_suffix(".md"), env.audio.with_suffix(".session.json"))
    assert load_session(ss).avalon_response == env.response
    assert session.created_at == NOW and session.model_id == "fake"
    assert session.audio_path == str(env.audio) and session.warnings == []


def test_speaker_states_hold_voiceprint_details(env):
    session, _, _ = env.run()
    spk_0, spk_1 = session.speakers["spk_0"], session.speakers["spk_1"]
    assert (spk_0.number, spk_1.number) == (1, 2)
    assert (spk_0.status, spk_0.name, spk_0.score, spk_0.scores) == ("unknown", None, None, {})
    assert spk_0.centroid == pytest.approx(unit(0).tolist())
    assert spk_1.centroid == pytest.approx(unit(1).tolist())
    assert spk_0.consistency == pytest.approx(1.0)
    assert spk_0.total_seconds == pytest.approx(7.8) and spk_0.segments == [pytest.approx((1.1, 8.9))]
    assert spk_0.db_sample_ids == [] and env.db.people() == {}


def test_second_meeting_auto_names_after_label(env):
    s1, _, ss = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")])
    env.db.save()
    s2, _, _ = env.run(audio=env.other_meeting_same_voices)
    assert s2.speakers["spk_0"].status == "auto" and s2.speakers["spk_0"].name == "田中"
    assert len(s2.speakers["spk_0"].db_sample_ids) == 1
    assert s2.speakers["spk_0"].scores == {"田中": pytest.approx(1.0)}
    assert s2.speakers["spk_1"].status == "unknown"
    assert s2.speakers["spk_1"].scores == {"田中": pytest.approx(0.0)}
    assert [s.source for s in env.db.people()["田中"]] == ["label", "auto"]


def test_rerun_uses_cache_and_reapplies_labels(env):
    s1, _, ss = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")])
    save_session(s1, ss)
    env.db.save()
    s2, md, _ = env.run()
    assert env.calls == 1  # API は呼ばれない
    assert s2.speakers["spk_0"].status == "confirmed"
    assert len(env.db.people()["田中"]) == 1  # 二重追加されない
    assert s2.speakers["spk_0"].db_sample_ids == [env.db.people()["田中"][0].id]
    assert s2.speakers["spk_1"].status == "unknown"
    assert "**[00:00:01] 田中**" in md.read_text()


def test_rerun_persists_db_and_session(env):
    label_first_run(env)
    s2, _, ss = env.run()
    reloaded = SpeakerDB.load(env.db_path, "fake", 4, 20)
    assert [x.id for x in reloaded.people()["田中"]] == s2.speakers["spk_0"].db_sample_ids
    assert load_session(ss).speakers["spk_0"].status == "confirmed"


def test_force_api_discards_labels(env):
    label_first_run(env)
    s2, md, _ = env.run(force_api=True)
    assert env.calls == 2 and {"code": "labels_discarded"} in s2.warnings
    assert s2.speakers["spk_0"].status == "unknown"
    assert "田中" not in env.db.people()
    assert "前回の名前付けは破棄されました" in md.read_text()


def test_force_api_without_confirmed_labels_does_not_warn(env):
    env.run()
    s2, _, _ = env.run(force_api=True)
    assert env.calls == 2 and s2.warnings == []


def test_moved_audio_does_not_self_match(env, tmp_path):
    s1, _, _ = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")])
    env.db.save()
    moved = tmp_path / "moved" / "meeting.wav"
    moved.parent.mkdir()
    shutil.copy(env.audio, moved)
    s2, _, _ = env.run(audio=moved)
    assert s2.speakers["spk_0"].status == "unknown"  # 自分自身の声紋は取り消し済み


def test_changed_audio_at_same_path_is_not_served_from_cache(env):
    env.run()
    shutil.copy(env.other_meeting_same_voices, env.audio)
    s2, _, ss = env.run()
    assert env.calls == 2
    assert load_session(ss).audio_sha256 == s2.audio_sha256


def test_corrupt_previous_session_is_ignored_with_a_warning(env, capsys):
    _, _, ss = env.run()
    ss.write_text("{ not json", encoding="utf-8")
    capsys.readouterr()
    s2, _, _ = env.run()
    captured = capsys.readouterr()
    assert env.calls == 2 and captured.out == ""
    assert captured.err.count("警告") == 1 and str(ss) in captured.err
    assert load_session(ss).audio_sha256 == s2.audio_sha256


def test_too_long_audio_rejected_before_upload(env, monkeypatch):
    monkeypatch.setattr(pipeline, "probe_duration", lambda p: 14400.1)
    with pytest.raises(AppError, match="4 時間"):
        env.run()
    assert env.calls == 0
    assert not env.audio.with_suffix(".md").exists()


def test_exactly_four_hours_is_accepted(env, monkeypatch):
    monkeypatch.setattr(pipeline, "probe_duration", lambda p: 14400.0)
    env.run()
    assert env.calls == 1


def test_missing_audio_is_a_usage_error(env, tmp_path):
    with pytest.raises(UsageError, match="音声ファイルが見つかりません"):
        env.run(audio=tmp_path / "nope.wav")
    assert env.calls == 0


def test_silent_recording(env):
    env.response = raw_diarized([], duration=30.0)
    session, md, _ = env.run()
    assert session.speakers == {} and "話者: なし" in md.read_text()


def test_speaker_limit_warning(env):
    env.response["warnings"] = ["speaker_limit_reached"]
    assert {"code": "speaker_limit_reached"} in env.run()[0].warnings


def test_unknown_avalon_warning_is_ignored(env):
    env.response["warnings"] = ["something_new"]
    assert env.run()[0].warnings == []


def test_speaker_with_too_little_speech_is_insufficient(env):
    words = clean_words("spk_0", 1.0, 9.0) + clean_words("spk_1", 11.0, 12.5)
    env.response = raw_diarized(words, duration=DURATION)
    env.db.add_sample("田中", unit(1), "other-sha", "label")
    session, _, _ = env.run()
    st = session.speakers["spk_1"]
    assert (st.status, st.name, st.score, st.scores) == ("insufficient", None, None, {})
    assert (st.centroid, st.consistency, st.db_sample_ids) == (None, None, [])
    assert st.display_label == "話者2"
    assert session.speakers["spk_0"].scores == {"田中": pytest.approx(0.0)}


def test_mixed_voices_raise_low_consistency_warning(env):
    # One diarized "speaker" that actually switches between the 150 Hz and 250 Hz voices.
    words = clean_words("spk_0", 1.0, 9.0) + clean_words("spk_0", 11.0, 19.0)
    env.response = raw_diarized(words, duration=DURATION)
    env.settings = replace(env.settings, min_consistency=0.8)
    session, _, _ = env.run()
    assert session.speakers["spk_0"].consistency == pytest.approx(0.7071, abs=0.01)
    assert session.warnings == [{"code": "low_consistency", "speaker": "spk_0"}]


def test_progress_goes_to_stderr_only(env, capsys):
    env.run()
    first = capsys.readouterr()
    assert first.out == "" and "文字起こし中" in first.err and "声紋を照合中" in first.err
    env.run()
    second = capsys.readouterr()
    assert second.out == "" and "キャッシュを再利用します" in second.err


def test_failed_transcription_leaves_db_untouched(env):
    s1, _, _ = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")])
    env.db.save()
    before = env.db_path.read_text(encoding="utf-8")

    def broken(mp3, opts):
        raise AppError("boom")

    with pytest.raises(AppError, match="boom"):
        run_transcribe(
            env.audio,
            TranscribeOptions(force_api=True),
            settings=env.settings,
            db=env.db,
            embedder=FreqEmbedder(),
            transcriber=broken,
            now=lambda: NOW,
        )
    assert env.db_path.read_text(encoding="utf-8") == before
