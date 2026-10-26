import copy
import os
import shutil
import wave
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from conftest import raw_diarized, unit

from kikiwake import pipeline
from kikiwake.audio import SAMPLE_RATE, sha256_file
from kikiwake.config import Settings
from kikiwake.errors import AppError, UsageError
from kikiwake.labeling import apply_labels
from kikiwake.pipeline import TranscribeOptions, recorded_at_for, run_transcribe
from kikiwake.session import load_session, save_session
from kikiwake.speaker_db import SpeakerDB, backup_path, locked

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
        self.calls = 0

    @property
    def db(self) -> SpeakerDB:
        """The DB as saved on disk."""
        return SpeakerDB.load(self.db_path, "fake", 4, self.settings.max_samples_per_person)

    @contextmanager
    def open_db(self):
        with locked(self.db_path):
            yield self.db

    def seed(self, name, embedding, sha="other-sha", source="label"):
        """Save a sample from some other recording to the DB."""
        with self.open_db() as db:
            db.add_sample(name, embedding, sha, source)
            db.save()

    def label(self, session_path, pairs):
        """Name speakers of a saved session, as the `label` command does."""
        with self.open_db() as db:
            session = load_session(session_path)
            apply_labels(session, db, pairs)
            db.save()
            save_session(session, session_path)
        return session

    def transcribe(self, mp3, opts):
        assert mp3.suffix == ".mp3" and mp3.stat().st_size > 0
        self.calls += 1
        return copy.deepcopy(self.response)

    def run(self, audio=None, *, transcriber=None, embedder=None, **opts):
        return run_transcribe(
            audio or self.audio,
            TranscribeOptions(**opts),
            settings=self.settings,
            open_db=self.open_db,
            embedder=embedder or FreqEmbedder(),
            transcriber=transcriber or self.transcribe,
            now=lambda: NOW,
        )


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def label_first_run(env):
    """Run once and name spk_0 「田中」 (as the `label` command does)."""
    _, md, ss = env.run()
    return env.label(ss, [("spk_0", "田中")]), md, ss


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
    label_first_run(env)
    s2, _, _ = env.run(audio=env.other_meeting_same_voices)
    assert s2.speakers["spk_0"].status == "auto" and s2.speakers["spk_0"].name == "田中"
    assert len(s2.speakers["spk_0"].db_sample_ids) == 1
    assert s2.speakers["spk_0"].scores == {"田中": pytest.approx(1.0)}
    assert s2.speakers["spk_1"].status == "unknown"
    assert s2.speakers["spk_1"].scores == {"田中": pytest.approx(0.0)}
    assert [s.source for s in env.db.people()["田中"]] == ["label", "auto"]


def test_rerun_uses_cache_and_reapplies_labels(env):
    label_first_run(env)
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


def moved_copy(env, tmp_path):
    moved = tmp_path / "moved" / "meeting.wav"
    moved.parent.mkdir()
    shutil.copy(env.audio, moved)
    return moved


def test_moved_audio_does_not_self_match_but_keeps_its_labels(env, tmp_path):
    label_first_run(env)
    s2, md, _ = env.run(audio=moved_copy(env, tmp_path))
    assert env.calls == 2  # no session next to the moved file, so the API is called again
    spk_0 = s2.speakers["spk_0"]
    assert spk_0.scores == {}  # 自分自身の声紋は照合前に取り消し済み
    assert (spk_0.status, spk_0.name) == ("confirmed", "田中")  # the label is carried over by voice
    (sample,) = env.db.people()["田中"]
    assert sample.source == "label" and spk_0.db_sample_ids == [sample.id]
    assert s2.speakers["spk_1"].status == "unknown" and s2.warnings == []
    assert "**[00:00:01] 田中**" in md.read_text()


def test_moved_audio_reports_labels_it_cannot_carry_over(env, tmp_path):
    label_first_run(env)
    env.label(env.audio.with_suffix(".session.json"), [("spk_1", "佐藤")])
    # This time spk_1 says too little to get a voiceprint, and spk_0's voice is nothing like 佐藤's.
    env.response = raw_diarized(
        clean_words("spk_0", 1.0, 9.0) + clean_words("spk_1", 11.0, 12.5), duration=DURATION
    )
    s2, md, _ = env.run(audio=moved_copy(env, tmp_path))
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "田中")
    assert (s2.speakers["spk_1"].status, s2.speakers["spk_1"].name) == ("insufficient", None)
    assert s2.warnings == [{"code": "labels_lost", "names": ["佐藤"]}]
    assert "前回の名前付け（佐藤）を引き継げませんでした。" in md.read_text()
    assert list(env.db.people()) == ["田中"]


def test_moved_audio_gives_one_speaker_only_one_carried_over_label(env, tmp_path):
    label_first_run(env)
    env.label(env.audio.with_suffix(".session.json"), [("spk_1", "佐藤")])
    # Now both voices come back as one speaker, close enough (0.71) to each label.
    env.response = raw_diarized(
        clean_words("spk_0", 1.0, 9.0) + clean_words("spk_0", 11.0, 19.0), duration=DURATION
    )
    s2, _, _ = env.run(audio=moved_copy(env, tmp_path))
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "田中")
    assert s2.warnings == [{"code": "labels_lost", "names": ["佐藤"]}]


def test_labels_are_carried_over_after_a_failed_first_try_at_the_new_place(env, tmp_path):
    label_first_run(env)
    moved = moved_copy(env, tmp_path)

    class Failing(FreqEmbedder):
        def embed(self, wave):
            raise AppError("声紋の計算に失敗")

    with pytest.raises(AppError):
        env.run(audio=moved, embedder=Failing())  # leaves only the paid response next to it
    s2, _, _ = env.run(audio=moved)
    assert env.calls == 2
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "田中")
    assert s2.warnings == []


def test_cached_rerun_follows_a_rename(env):
    label_first_run(env)
    with env.open_db() as db:
        db.rename("田中", "鈴木")
        db.save()
    s2, md, _ = env.run()
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "鈴木")
    assert list(env.db.people()) == ["鈴木"]
    assert [x.id for x in env.db.people()["鈴木"]] == s2.speakers["spk_0"].db_sample_ids
    assert "**[00:00:01] 鈴木**" in md.read_text()


def test_cached_rerun_does_not_bring_back_a_deleted_person(env, capsys):
    label_first_run(env)
    with env.open_db() as db:
        db.delete("田中")
        db.save()
    capsys.readouterr()
    s2, _, _ = env.run()
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("unknown", None)
    assert env.db.people() == {} and s2.warnings == []
    assert "警告" not in capsys.readouterr().err


def test_cached_rerun_keeps_the_name_of_a_speaker_without_a_voiceprint(env, capsys):
    env.response = raw_diarized(
        clean_words("spk_0", 1.0, 9.0) + clean_words("spk_1", 11.0, 12.5), duration=DURATION
    )
    _, _, ss = env.run()
    env.label(ss, [("spk_1", "佐藤")])
    capsys.readouterr()
    s2, _, _ = env.run()
    assert (s2.speakers["spk_1"].status, s2.speakers["spk_1"].name) == ("confirmed", "佐藤")
    assert "佐藤 は声紋が不足しているため" in capsys.readouterr().err


def test_cached_rerun_keeps_a_label_whose_sample_was_evicted(env):
    env.settings = replace(env.settings, max_samples_per_person=1)
    label_first_run(env)
    env.run(audio=env.other_meeting_same_voices)  # its auto sample of 田中 evicts the label sample
    s2, _, _ = env.run()
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "田中")


def another_recording(out_dir) -> str:
    return (
        f"{out_dir} には別の録音（同じ名前）の記録があります。"
        f'--output-dir で別のフォルダを指定してください（例: "{out_dir} (2)"）'
    )


@pytest.mark.parametrize("force_api", [False, True])
def test_changed_audio_at_same_path_does_not_overwrite_its_record(env, force_api):
    _, md, ss = env.run()
    before = (ss.read_bytes(), md.read_bytes())
    shutil.copy(env.other_meeting_same_voices, env.audio)  # another recording under the same name
    with pytest.raises(UsageError) as e:
        env.run(force_api=force_api)
    assert str(e.value) == another_recording(env.audio.parent)
    assert env.calls == 1  # refused before any API call
    assert (ss.read_bytes(), md.read_bytes()) == before


def test_same_stem_in_the_same_output_dir_is_refused_and_another_dir_works(env, tmp_path):
    records = tmp_path / "records"
    first = tmp_path / "a" / "meeting.wav"
    second = tmp_path / "b" / "meeting.wav"  # another meeting, the same file name
    for path, source in ((first, env.audio), (second, env.other_meeting_same_voices)):
        path.parent.mkdir()
        shutil.copy(source, path)
    _, _, ss = env.run(audio=first, output_dir=records / "meeting")
    with pytest.raises(UsageError) as e:
        env.run(audio=second, output_dir=records / "meeting")
    assert str(e.value) == another_recording(records / "meeting")
    assert env.calls == 1 and load_session(ss).audio_sha256 == sha256_file(first)

    s2, _, ss2 = env.run(audio=second, output_dir=records / "meeting (2)")
    assert ss2 == records / "meeting (2)" / "meeting.session.json"
    assert env.calls == 2 and s2.audio_sha256 == sha256_file(second)


def test_same_recording_again_in_the_same_output_dir_reuses_the_record(env, tmp_path):
    out = tmp_path / "records" / "meeting"
    env.run(output_dir=out)
    copy_elsewhere = moved_copy(env, tmp_path)  # same content, another place
    env.run(audio=copy_elsewhere, output_dir=out)
    assert env.calls == 1


def test_corrupt_previous_session_is_ignored_with_a_warning(env, capsys):
    _, _, ss = env.run()
    for broken in (ss, backup_path(ss)):  # its backup too
        broken.write_text("{ not json", encoding="utf-8")
    capsys.readouterr()
    s2, _, _ = env.run()
    captured = capsys.readouterr()
    assert env.calls == 2 and captured.out == ""
    assert captured.err.count("警告") == 1 and str(ss) in captured.err
    assert load_session(ss).audio_sha256 == s2.audio_sha256


def test_torn_previous_session_is_read_from_its_backup_and_not_bought_again(env, capsys):
    label_first_run(env)
    ss = env.audio.with_suffix(".session.json")
    ss.write_bytes(ss.read_bytes()[:100])  # cut short, as by a kill midway through a save
    capsys.readouterr()
    s2, _, _ = env.run()
    assert env.calls == 1
    assert (s2.speakers["spk_0"].status, s2.speakers["spk_0"].name) == ("confirmed", "田中")
    assert "警告: meeting.session.json が壊れていたため控えから読み込みました" in capsys.readouterr().err
    assert load_session(ss) == s2  # repaired by the save


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
    env.seed("田中", unit(1))
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
    label_first_run(env)
    before = env.db_path.read_text(encoding="utf-8")

    def broken(mp3, opts):
        raise AppError("boom")

    with pytest.raises(AppError, match="boom"):
        env.run(transcriber=broken, force_api=True)
    assert env.db_path.read_text(encoding="utf-8") == before


def test_label_made_while_the_api_is_in_flight_is_kept(env):
    _, _, other = env.run(audio=env.other_meeting_same_voices)

    def slow_api(mp3, opts):  # meanwhile, a `label` command names a speaker of the other meeting
        env.label(other, [("spk_1", "佐藤")])
        return env.transcribe(mp3, opts)

    session, _, _ = env.run(transcriber=slow_api)
    assert list(env.db.people()) == ["佐藤"]
    assert (session.speakers["spk_1"].status, session.speakers["spk_1"].name) == ("auto", "佐藤")


def test_label_made_while_a_cached_rerun_computes_voiceprints_is_kept(env):
    _, _, ss = env.run()
    labelled = []

    class LabellingMidway(FreqEmbedder):
        def embed(self, wave):
            if not labelled:  # meanwhile, a `label` command names spk_1 of this very session
                labelled.append(env.label(ss, [("spk_1", "佐藤")]))
            return super().embed(wave)

    s2, _, _ = env.run(embedder=LabellingMidway())
    assert env.calls == 1 and labelled
    assert (s2.speakers["spk_1"].status, s2.speakers["spk_1"].name) == ("confirmed", "佐藤")
    assert [x.id for x in env.db.people()["佐藤"]] == s2.speakers["spk_1"].db_sample_ids
    assert load_session(ss).speakers["spk_1"].status == "confirmed"


def test_paid_response_that_fails_to_parse_is_kept_and_not_bought_again(env):
    env.response = {"turns": "not a list"}
    with pytest.raises(AppError, match="応答形式"):
        env.run()
    kept = load_session(env.audio.with_suffix(".session.json"))
    assert kept.avalon_response == {"turns": "not a list"}
    assert (kept.speakers, kept.warnings, kept.audio_sha256) == ({}, [], sha256_file(env.audio))
    with pytest.raises(AppError, match="応答形式"):
        env.run()
    assert env.calls == 1  # reused from the kept session (only --force-api buys it again)


@pytest.mark.parametrize("failure", [AppError("声紋の計算に失敗"), KeyboardInterrupt()])
def test_paid_response_is_kept_when_a_later_step_fails(env, failure):
    class Failing(FreqEmbedder):
        def embed(self, wave):
            raise failure

    with pytest.raises(type(failure)):
        env.run(embedder=Failing())
    assert load_session(env.audio.with_suffix(".session.json")).avalon_response == env.response
    assert not env.db_path.exists() and not env.audio.with_suffix(".md").exists()

    session, md, _ = env.run()
    assert env.calls == 1
    assert [s.display_label for s in session.speakers.values()] == ["話者1", "話者2"]
    assert md.read_text().startswith("# meeting.wav の文字起こし")


OPTIONS_IGNORED = (
    "保存済みの文字起こしを再利用するため --language/--prompt/--max-speakers は反映されません"
    "（反映するには --force-api）\n"
)


@pytest.mark.parametrize(
    "options",
    [
        dict(language="ja"),
        dict(prompt="固有名詞"),
        dict(max_speakers=2),
        dict(language="ja", prompt="固有名詞", max_speakers=2),
    ],
    ids=["language", "prompt", "max_speakers", "all"],
)
def test_cached_rerun_says_once_that_api_options_are_not_applied(env, capsys, options):
    env.run(**options)
    assert OPTIONS_IGNORED not in capsys.readouterr().err  # the API got them
    env.run(**options)
    assert capsys.readouterr().err.count(OPTIONS_IGNORED) == 1
    env.run()
    assert OPTIONS_IGNORED not in capsys.readouterr().err
    env.run(force_api=True, **options)
    assert OPTIONS_IGNORED not in capsys.readouterr().err
    assert env.calls == 2


def test_session_is_saved_before_the_db(env, monkeypatch):
    label_first_run(env)

    def full_disk(self):
        raise OSError("disk full")

    monkeypatch.setattr(SpeakerDB, "save", full_disk)
    with pytest.raises(OSError, match="disk full"):
        env.run(audio=env.other_meeting_same_voices)
    kept = load_session(env.other_meeting_same_voices.with_suffix(".session.json"))
    assert (kept.speakers["spk_0"].status, kept.speakers["spk_0"].name) == ("auto", "田中")
    assert not env.other_meeting_same_voices.with_suffix(".md").exists()


def test_session_and_db_are_saved_before_the_markdown(env):
    label_first_run(env)
    env.other_meeting_same_voices.with_suffix(".md").mkdir()  # cannot be written
    with pytest.raises(OSError):
        env.run(audio=env.other_meeting_same_voices)
    kept = load_session(env.other_meeting_same_voices.with_suffix(".session.json"))
    assert kept.speakers["spk_0"].status == "auto"
    assert [s.source for s in env.db.people()["田中"]] == ["label", "auto"]


MTIME = 1_790_000_000


def mtime_iso(timestamp: float = MTIME) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def test_voice_memo_stem_gives_recorded_at(tmp_path):
    p = tmp_path / "20261016 125520-357FE498.m4a"
    p.write_bytes(b"x")
    assert recorded_at_for(p).startswith("2026-10-16T12:55:20")
    assert recorded_at_for(p) == datetime(2026, 10, 16, 12, 55, 20).astimezone().isoformat(timespec="seconds")


@pytest.mark.parametrize(
    "name",
    [
        "meeting.m4a",
        "20261399 999999-x.m4a",
        "20260230 100000-x.m4a",  # no 30 February
        "20261016 240000-x.m4a",  # no hour 24
        "20261016_125520-x.m4a",  # not the Voice Memos separator
        "メモ 20261016 125520.m4a",  # the date is not at the start
        "会議 10.26.m4a",
    ],
)
def test_other_stems_use_mtime(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")
    os.utime(p, (MTIME, MTIME))
    assert recorded_at_for(p) == mtime_iso()


def test_pipeline_stores_recorded_at(env):
    os.utime(env.audio, (MTIME, MTIME))
    session, _, ss = env.run()
    assert session.recorded_at == mtime_iso()
    assert load_session(ss).recorded_at == mtime_iso()


def test_pipeline_stores_recorded_at_in_the_early_save_too(env):
    os.utime(env.audio, (MTIME, MTIME))
    env.response = {"turns": "not a list"}  # fails after the paid response is kept
    with pytest.raises(AppError, match="応答形式"):
        env.run()
    kept = load_session(env.audio.with_suffix(".session.json"))
    assert kept.speakers == {} and kept.recorded_at == mtime_iso()


def test_pipeline_reads_recorded_at_from_a_voice_memo_name(env, tmp_path):
    memo = tmp_path / "20261016 125520-357FE498.wav"
    shutil.copy(env.audio, memo)
    os.utime(memo, (MTIME, MTIME))  # the name wins over the modification time
    session, _, ss = env.run(audio=memo)
    expected = datetime(2026, 10, 16, 12, 55, 20).astimezone().isoformat(timespec="seconds")
    assert session.recorded_at == expected and load_session(ss).recorded_at == expected


def test_cached_rerun_decides_recorded_at_again(env):
    os.utime(env.audio, (MTIME, MTIME))
    env.run()
    later = MTIME + 86_400
    os.utime(env.audio, (later, later))  # same content, so the cache is reused
    session, _, ss = env.run()
    assert env.calls == 1
    assert session.recorded_at == mtime_iso(later) == load_session(ss).recorded_at
