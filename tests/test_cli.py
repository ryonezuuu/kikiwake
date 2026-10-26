import copy
import json
import re
import shutil
import wave
import warnings
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import raw_diarized, unit

from kikiwake import cli
from kikiwake.audio import SAMPLE_RATE
from kikiwake.avalon import API_KEY_ENVS, API_KEY_FILENAME
from kikiwake.cli import main
from kikiwake.config import Settings
from kikiwake.errors import AppError, UsageError
from kikiwake.session import load_session, save_session
from kikiwake.speaker_db import SpeakerDB

real_make_transcriber = cli.make_transcriber
ISO_SECONDS_WITH_OFFSET = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d")

DURATION = 30.0
# (speaker, first word start, end, tone frequency): clean speech, 0.5 s per word, no gaps or overlap.
REGIONS = [("spk_0", 0.0, 8.0, 150.0), ("spk_1", 10.0, 18.0, 250.0)]


class FreqEmbedder:
    """Embeds by pitch: a peak below 200 Hz is ``unit(0)``, otherwise ``unit(1)``."""

    model_id = "fake"
    dim = 4

    def embed(self, wav: np.ndarray) -> np.ndarray:
        peak = np.argmax(np.abs(np.fft.rfft(wav))) * SAMPLE_RATE / len(wav)
        return unit(0 if peak < 200 else 1)


def write_meeting(path: Path, amplitude: float = 0.3) -> None:
    t = np.arange(round(DURATION * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for _, start, end, freq in REGIONS:
        inside = (t >= start) & (t < end)
        signal[inside] = amplitude * np.sin(2 * np.pi * freq * t[inside])
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


class CliEnv:
    """A tmp data home plus ``meeting.wav``, with the embedder and the Avalon call replaced by fakes."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.home = tmp_path / "home"
        self.audio = tmp_path / "meeting.wav"
        write_meeting(self.audio)
        self.session_path = self.audio.with_suffix(".session.json")
        self.markdown_path = self.audio.with_suffix(".md")
        self.png_path = self.audio.with_name("meeting.voices.png")
        self.response = meeting_response()
        self.calls = []  # the TranscribeOptions of every API call

    def transcriber(self, mp3, opts):
        assert mp3.suffix == ".mp3" and mp3.stat().st_size > 0
        self.calls.append(opts)
        return copy.deepcopy(self.response)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    env = CliEnv(tmp_path)
    monkeypatch.setenv("KIKIWAKE_HOME", str(env.home))
    monkeypatch.setattr(cli, "make_embedder", lambda home: FreqEmbedder())
    monkeypatch.setattr(cli, "make_transcriber", lambda settings, **kwargs: env.transcriber)
    return env


def no_model(monkeypatch):
    """Fail the test if a command loads the embedding model."""

    def forbidden(home):
        raise AssertionError("the embedding model must not be loaded")

    monkeypatch.setattr(cli, "make_embedder", forbidden)


def transcribe(cli_env, capsys, *extra) -> dict:
    assert main(["transcribe", str(cli_env.audio), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_transcribe_json_then_label(cli_env, capsys):
    assert main(["transcribe", str(cli_env.audio), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [s["label"] for s in summary["speakers"]] == ["話者1", "話者2"]
    assert main(["label", summary["session_path"], "spk_0=田中", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["speakers"][0]["label"] == "田中"
    assert "**[00:00:00] 田中**" in Path(summary["markdown_path"]).read_text()


def db_names(capsys) -> list[str]:
    capsys.readouterr()
    assert main(["speakers", "list", "--json"]) == 0
    return [p["name"] for p in json.loads(capsys.readouterr().out)]


def test_label_made_while_transcribe_waits_for_the_api_is_kept(cli_env, capsys):
    other = cli_env.tmp_path / "other.wav"  # the same two voices in another meeting
    write_meeting(other, amplitude=0.2)
    assert main(["transcribe", str(other)]) == 0
    api = cli_env.transcriber

    def slow_api(mp3, opts):  # while this call is "in flight", the user names a speaker of the other meeting
        assert main(["label", str(other.with_suffix(".session.json")), "spk_1=佐藤"]) == 0
        capsys.readouterr()  # that command's own output
        return api(mp3, opts)

    cli_env.transcriber = slow_api
    summary = transcribe(cli_env, capsys)
    assert db_names(capsys) == ["佐藤"]  # not overwritten by a snapshot taken before the label
    assert load_session(other.with_suffix(".session.json")).speakers["spk_1"].status == "confirmed"
    spk_1 = summary["speakers"][1]
    assert (spk_1["label"], spk_1["status"]) == ("佐藤", "auto")  # matched against the fresh DB


def track_db(monkeypatch) -> list[str]:
    """Record taking the DB lock, loading and saving the DB, in order."""
    events: list[str] = []
    real_locked, real_load, real_save = cli.locked, SpeakerDB.load, SpeakerDB.save

    @contextmanager
    def tracked_locked(path):
        with real_locked(path):
            events.append("lock")
            yield
            events.append("unlock")

    def tracked_load(cls, *args, **kwargs):
        events.append("load")
        return real_load(*args, **kwargs)

    def tracked_save(self):
        events.append("save")
        real_save(self)

    monkeypatch.setattr(cli, "locked", tracked_locked)
    monkeypatch.setattr(SpeakerDB, "load", classmethod(tracked_load))
    monkeypatch.setattr(SpeakerDB, "save", tracked_save)
    return events


def test_commands_that_change_the_db_hold_its_lock_from_load_to_save(cli_env, monkeypatch):
    main(["transcribe", str(cli_env.audio)])
    events = track_db(monkeypatch)
    for argv in (
        ["label", str(cli_env.session_path), "spk_0=田中"],
        ["speakers", "rename", "田中", "鈴木"],
        ["speakers", "delete", "鈴木"],
    ):
        events.clear()
        assert main(argv) == 0
        assert events == ["lock", "load", "save", "unlock"], argv


def test_transcribe_checks_the_db_early_but_changes_it_only_under_the_lock(cli_env, monkeypatch):
    events = track_db(monkeypatch)
    api = cli_env.transcriber

    def tracked_api(mp3, opts):
        events.append("api")
        return api(mp3, opts)

    cli_env.transcriber = tracked_api
    assert main(["transcribe", str(cli_env.audio)]) == 0
    assert events == ["load", "api", "lock", "load", "save", "unlock"]


def test_label_bad_key_exit_2(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    assert main(["label", str(cli_env.session_path), "nope=田中"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("エラー: ") and "話者1" in captured.err


def test_label_arg_without_equals(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    assert main(["label", str(cli_env.session_path), "田中"]) == 2
    assert "田中" in capsys.readouterr().err


def test_label_changes_nothing_when_one_pair_is_bad(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    before = cli_env.session_path.read_text(encoding="utf-8")
    assert main(["label", str(cli_env.session_path), "spk_0=田中", "nope=鈴木"]) == 2
    assert cli_env.session_path.read_text(encoding="utf-8") == before
    assert "田中" not in (cli_env.home / "db.json").read_text(encoding="utf-8")


def test_missing_api_key_exit_2(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_transcriber", real_make_transcriber)
    for k in API_KEY_ENVS:
        monkeypatch.delenv(k, raising=False)
    assert main(["transcribe", str(cli_env.audio)]) == 2
    captured = capsys.readouterr()
    assert "AQUAVOICE_API" in captured.err and captured.out == ""
    assert str(cli_env.home / API_KEY_FILENAME) in captured.err  # where to put the key file
    assert not cli_env.session_path.exists()


def test_transcribe_reads_the_key_file_in_kikiwake_home(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_transcriber", real_make_transcriber)
    for k in API_KEY_ENVS:
        monkeypatch.delenv(k, raising=False)
    cli_env.home.mkdir()
    (cli_env.home / API_KEY_FILENAME).write_text("file-key-123\n", encoding="utf-8")
    built = []

    class FakeClient:  # stands in for AvalonClient, so the real API is never reached
        def __init__(self, key, settings):
            built.append(key)

        def transcribe(self, mp3, **kwargs):
            return copy.deepcopy(cli_env.response)

    monkeypatch.setattr(cli, "AvalonClient", FakeClient)
    assert main(["transcribe", str(cli_env.audio), "--json"]) == 0
    captured = capsys.readouterr()
    assert built == ["file-key-123"]
    assert "file-key-123" not in captured.out + captured.err
    assert cli_env.session_path.exists()


def test_cached_rerun_needs_no_api_key(cli_env, monkeypatch, capsys):
    main(["transcribe", str(cli_env.audio)])
    monkeypatch.setattr(cli, "make_transcriber", real_make_transcriber)
    for k in API_KEY_ENVS:
        monkeypatch.delenv(k, raising=False)
    assert main(["transcribe", str(cli_env.audio)]) == 0
    assert len(cli_env.calls) == 1


def test_speakers_list_rename_delete(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    capsys.readouterr()
    assert main(["speakers", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [(p["name"], p["samples"]) for p in listed] == [("田中", 1)]
    assert ISO_SECONDS_WITH_OFFSET.fullmatch(listed[0]["updated_at"])
    assert main(["speakers", "rename", "田中", "鈴木"]) == 0
    assert capsys.readouterr().out == ""
    assert main(["speakers", "list", "--json"]) == 0
    assert [p["name"] for p in json.loads(capsys.readouterr().out)] == ["鈴木"]
    assert main(["speakers", "delete", "佐藤"]) == 2
    assert "佐藤" in capsys.readouterr().err
    assert main(["speakers", "delete", "鈴木"]) == 0
    assert capsys.readouterr().out == ""
    assert main(["speakers", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_speakers_rename_json(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    capsys.readouterr()
    assert main(["speakers", "rename", "田中", "鈴木", "--json"]) == 0
    expected = {"renamed": {"from": "田中", "to": "鈴木"}}
    assert capsys.readouterr().out == json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    assert main(["speakers", "list", "--json"]) == 0
    assert [p["name"] for p in json.loads(capsys.readouterr().out)] == ["鈴木"]
    assert main(["speakers", "rename", "田中", "佐藤", "--json"]) == 2  # errors are unchanged
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("エラー: ")


def test_speakers_delete_json(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    capsys.readouterr()
    assert main(["speakers", "delete", "田中", "--json"]) == 0
    expected = {"deleted": "田中"}
    assert capsys.readouterr().out == json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    assert main(["speakers", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["speakers", "delete", "田中", "--json"]) == 2  # errors are unchanged
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("エラー: ")


@pytest.mark.parametrize(
    ("new", "message"),
    [("鈴木?", "？"), ("鈴木？", "？"), ("  ", "名前が空です")],
    ids=["half-width-question-mark", "full-width-question-mark", "blank"],
)
def test_speakers_rename_rejects_names_label_would_reject(cli_env, capsys, new, message):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    assert main(["speakers", "rename", "田中", new]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert db_names(capsys) == ["田中"]


def test_speakers_rename_strips_the_new_name(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    capsys.readouterr()
    assert main(["speakers", "rename", "田中", " 山田 太郎\t", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"renamed": {"from": "田中", "to": "山田 太郎"}}
    assert db_names(capsys) == ["山田 太郎"]


def test_speakers_list_human_format(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中", "spk_1=鈴木"])
    capsys.readouterr()
    assert main(["speakers", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert re.fullmatch(r"田中  サンプル 1 件  最終更新 \S+", lines[0])
    assert ISO_SECONDS_WITH_OFFSET.fullmatch(lines[0].rsplit(" ", 1)[1])
    assert lines[1].startswith("鈴木  サンプル 1 件  最終更新 ")


def test_speakers_list_of_empty_db_prints_nothing(cli_env, capsys):
    assert main(["speakers", "list"]) == 0
    assert capsys.readouterr().out == ""


def test_speakers_rename_to_existing_name_exit_2(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中", "spk_1=鈴木"])
    capsys.readouterr()
    assert main(["speakers", "rename", "田中", "鈴木"]) == 2
    assert "鈴木" in capsys.readouterr().err


def test_json_stdout_is_pure_json(cli_env, capsys):
    main(["transcribe", str(cli_env.audio), "--json"])
    captured = capsys.readouterr()
    json.loads(captured.out)  # 進捗ログが混ざっていないこと
    assert captured.out.endswith("}\n") and "文字起こし中" in captured.err
    assert "田中" not in captured.out and "話者1" in captured.out  # ensure_ascii=False


def test_human_output_is_the_markdown_path_only(cli_env, capsys):
    assert main(["transcribe", str(cli_env.audio)]) == 0
    assert capsys.readouterr().out == f"{cli_env.markdown_path}\n"
    assert cli_env.markdown_path.read_text().startswith("# meeting.wav の文字起こし")
    assert main(["label", str(cli_env.session_path), "spk_0=田中"]) == 0
    assert capsys.readouterr().out == f"{cli_env.markdown_path}\n"


def test_viz_flag_writes_png(cli_env):
    assert main(["transcribe", str(cli_env.audio), "--viz"]) == 0
    assert cli_env.audio.with_name("meeting.voices.png").exists()


def test_viz_path_in_summary_follows_the_png(cli_env, capsys):
    assert transcribe(cli_env, capsys)["viz_path"] is None
    summary = transcribe(cli_env, capsys, "--viz")
    assert summary["viz_path"] == str(cli_env.png_path)
    assert transcribe(cli_env, capsys)["viz_path"] == str(cli_env.png_path)  # still there


def test_transcribe_passes_options_and_cached_rerun_skips_api(cli_env, tmp_path, capsys):
    out_dir = tmp_path / "out"
    summary = transcribe(
        cli_env, capsys, "--output-dir", str(out_dir), "--language", "ja",
        "--prompt", "固有名詞", "--max-speakers", "3",
    )  # fmt: skip
    (opts,) = cli_env.calls
    assert (opts.language, opts.prompt, opts.max_speakers) == ("ja", "固有名詞", 3)
    assert (opts.output_dir, opts.force_api) == (out_dir, False)
    assert summary["markdown_path"] == str(out_dir / "meeting.md")
    assert summary["session_path"] == str(out_dir / "meeting.session.json")
    assert not cli_env.session_path.exists()

    transcribe(cli_env, capsys, "--output-dir", str(out_dir))
    assert len(cli_env.calls) == 1
    transcribe(cli_env, capsys, "--output-dir", str(out_dir), "--force-api")
    assert len(cli_env.calls) == 2 and cli_env.calls[1].force_api is True


def test_another_recording_with_the_same_name_exit_2(cli_env, tmp_path, capsys):
    out_dir = tmp_path / "records" / "meeting"
    transcribe(cli_env, capsys, "--output-dir", str(out_dir))
    other = tmp_path / "other" / "meeting.wav"
    other.parent.mkdir()
    write_meeting(other, amplitude=0.2)
    assert main(["transcribe", str(other), "--output-dir", str(out_dir), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and len(cli_env.calls) == 1
    assert captured.err.endswith(
        f"エラー: {out_dir} には別の録音（同じ名前）の記録があります。"
        f'--output-dir で別のフォルダを指定してください（例: "{out_dir} (2)"）\n'
    )


def test_relative_paths_are_stored_absolute(cli_env, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["transcribe", "meeting.wav", "--output-dir", "out", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    session_path = tmp_path / "out" / "meeting.session.json"
    assert summary["session_path"] == str(session_path)
    assert load_session(session_path).audio_path == str(cli_env.audio)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert main(["viz", "../out/meeting.session.json"]) == 0  # finds the audio from any cwd
    assert capsys.readouterr().out == f"{tmp_path / 'out' / 'meeting.voices.png'}\n"


def test_home_is_expanded_in_paths(cli_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["transcribe", "~/meeting.wav", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["markdown_path"] == str(cli_env.markdown_path)


def test_session_created_at_is_local_iso8601_in_seconds(cli_env):
    main(["transcribe", str(cli_env.audio)])
    assert ISO_SECONDS_WITH_OFFSET.fullmatch(load_session(cli_env.session_path).created_at)


def test_times_follow_the_timezone_setting(cli_env, capsys):
    cli_env.home.mkdir()
    (cli_env.home / "config.toml").write_text('timezone = "Pacific/Honolulu"\n', encoding="utf-8")
    main(["transcribe", str(cli_env.audio)])
    main(["label", str(cli_env.session_path), "spk_0=田中"])
    session = load_session(cli_env.session_path)
    assert session.created_at.endswith("-10:00") and session.recorded_at.endswith("-10:00")
    capsys.readouterr()
    assert main(["speakers", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["updated_at"].endswith("-10:00")


def test_unknown_timezone_setting_exit_2(cli_env, capsys):
    cli_env.home.mkdir()
    (cli_env.home / "config.toml").write_text('timezone = "Asia/Tokio"\n', encoding="utf-8")
    assert main(["transcribe", str(cli_env.audio)]) == 2
    err = capsys.readouterr().err
    assert "timezone" in err and "Asia/Tokio" in err and cli_env.calls == []


def test_viz_command(cli_env, monkeypatch, capsys):
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    no_model(monkeypatch)
    assert main(["viz", str(cli_env.session_path)]) == 0
    assert capsys.readouterr().out == f"{cli_env.png_path}\n"
    assert cli_env.png_path.read_bytes().startswith(b"\x89PNG")
    assert main(["viz", str(cli_env.session_path), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["viz_path"] == str(cli_env.png_path)
    assert summary["session_path"] == str(cli_env.session_path)


def test_viz_without_the_original_audio_exit_1(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    cli_env.audio.unlink()
    assert main(["viz", str(cli_env.session_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"エラー: 元の音声ファイルが見つかりません: {cli_env.audio}\n"
    assert not cli_env.png_path.exists()


VM_AUDIO = "/sessions/brave-otter-1234/mnt/kikiwake/meeting.wav"  # a path of another conversation's VM


def archived_with_vm_path(cli_env, tmp_path, capsys) -> tuple[Path, Path]:
    """Transcribe into ``<root>/records/meeting/``, then store the audio as a VM path that no longer exists.

    Returns ``(root, session_path)``.
    """
    root = tmp_path / "kikiwake"
    session_path = root / "records" / "meeting" / "meeting.session.json"
    transcribe(cli_env, capsys, "--output-dir", str(session_path.parent))
    save_session(replace(load_session(session_path), audio_path=VM_AUDIO), session_path)
    return root, session_path


def put_copy(root: Path, folder: str, source: Path) -> Path:
    target = root / folder / "meeting.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    return target


def test_viz_finds_the_recording_in_the_records_folder_by_name_and_content(cli_env, tmp_path, capsys):
    root, session_path = archived_with_vm_path(cli_env, tmp_path, capsys)
    decoy = tmp_path / "decoy.wav"
    write_meeting(decoy, amplitude=0.2)
    put_copy(root, "a other meeting", decoy)  # same name, other content, and found first
    put_copy(root, "recordings/2026", cli_env.audio)
    cli_env.audio.unlink()
    assert main(["viz", str(session_path)]) == 0
    png = session_path.with_name("meeting.voices.png")
    assert capsys.readouterr().out == f"{png}\n" and png.read_bytes().startswith(b"\x89PNG")
    assert load_session(session_path).audio_path == VM_AUDIO  # the session is not changed


def test_viz_does_not_take_the_recording_from_records_or_kikiwake_home(cli_env, tmp_path, capsys):
    root, session_path = archived_with_vm_path(cli_env, tmp_path, capsys)
    put_copy(root, "records/meeting", cli_env.audio)
    put_copy(root, ".kikiwake", cli_env.audio)
    assert main(["viz", str(session_path)]) == 1
    assert capsys.readouterr().err == f"エラー: 元の音声ファイルが見つかりません: {VM_AUDIO}\n"


def test_viz_outside_a_records_folder_does_not_search(cli_env, tmp_path, capsys):
    out = tmp_path / "out"
    transcribe(cli_env, capsys, "--output-dir", str(out))
    session_path = out / "meeting.session.json"
    save_session(replace(load_session(session_path), audio_path=VM_AUDIO), session_path)
    put_copy(tmp_path, "elsewhere", cli_env.audio)
    assert main(["viz", str(session_path)]) == 1
    assert "元の音声ファイルが見つかりません" in capsys.readouterr().err


def test_label_redraws_the_picture_from_the_found_recording(cli_env, tmp_path, capsys):
    root, session_path = archived_with_vm_path(cli_env, tmp_path, capsys)
    put_copy(root, "recordings", cli_env.audio)
    png = session_path.with_name("meeting.voices.png")
    png.write_bytes(b"stale")
    assert main(["label", str(session_path), "spk_0=田中", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["viz_path"] == str(png) and VIZ_SKIPPED not in captured.err
    assert png.read_bytes().startswith(b"\x89PNG")


def test_label_regenerates_png_only_when_it_exists(cli_env, monkeypatch, capsys):
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    no_model(monkeypatch)
    assert main(["label", str(cli_env.session_path), "spk_0=田中", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["viz_path"] is None
    assert not cli_env.png_path.exists()

    cli_env.png_path.write_bytes(b"stale")
    assert main(["label", str(cli_env.session_path), "spk_1=鈴木", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["viz_path"] == str(cli_env.png_path)
    assert cli_env.png_path.read_bytes().startswith(b"\x89PNG")


VIZ_SKIPPED = "警告: 可視化をスキップしました: "


def test_transcribe_viz_failure_is_only_a_warning(cli_env, monkeypatch, capsys):
    def broken(*args):
        raise AppError("描画できません")

    monkeypatch.setattr(cli, "render_viz", broken)
    assert main(["transcribe", str(cli_env.audio), "--viz", "--json"]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["viz_path"] is None and summary["speakers"]
    assert f"{VIZ_SKIPPED}描画できません\n" in captured.err
    assert load_session(cli_env.session_path).speakers and cli_env.markdown_path.exists()


def test_label_viz_failure_is_only_a_warning(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    cli_env.png_path.write_bytes(b"stale")
    cli_env.audio.unlink()  # the picture cannot be redrawn without it
    capsys.readouterr()
    assert main(["label", str(cli_env.session_path), "spk_0=田中", "--json"]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["viz_path"] is None and summary["speakers"][0]["label"] == "田中"
    assert f"{VIZ_SKIPPED}元の音声ファイルが見つかりません: {cli_env.audio}\n" in captured.err
    assert load_session(cli_env.session_path).speakers["spk_0"].status == "confirmed"
    assert db_names(capsys) == ["田中"]


def test_label_viz_write_error_is_only_a_warning(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    cli_env.png_path.mkdir()  # an existing "picture" that cannot be overwritten
    capsys.readouterr()
    assert main(["label", str(cli_env.session_path), "spk_0=田中", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["viz_path"] is None and VIZ_SKIPPED in captured.err


def test_transcribe_redraws_an_existing_png(cli_env, capsys):
    transcribe(cli_env, capsys, "--viz")
    cli_env.png_path.write_bytes(b"stale")
    assert transcribe(cli_env, capsys)["viz_path"] == str(cli_env.png_path)
    assert cli_env.png_path.read_bytes().startswith(b"\x89PNG")


def test_label_warning_goes_to_stderr(cli_env, capsys):
    words = clean_words("spk_0", 0.0, 8.0) + clean_words("spk_1", 10.0, 11.5)
    cli_env.response = raw_diarized(words, duration=DURATION)
    main(["transcribe", str(cli_env.audio)])
    capsys.readouterr()
    assert main(["label", str(cli_env.session_path), "spk_1=佐藤"]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{cli_env.markdown_path}\n"
    assert "佐藤 は声紋が不足しているため、DB には登録しません" in captured.err
    assert "**[00:00:10] 佐藤**" in cli_env.markdown_path.read_text()


def test_label_with_missing_session_exit_2(cli_env, capsys):
    assert main(["label", str(cli_env.session_path), "spk_0=田中"]) == 2
    assert "セッションを読み込めません" in capsys.readouterr().err


def test_label_writes_next_to_the_session_file(cli_env, tmp_path, capsys):
    out_dir = tmp_path / "out"
    summary = transcribe(cli_env, capsys, "--output-dir", str(out_dir))
    assert main(["label", summary["session_path"], "spk_0=田中"]) == 0
    assert "**[00:00:00] 田中**" in (out_dir / "meeting.md").read_text()
    assert not cli_env.markdown_path.exists()


def test_commands_other_than_transcribe_do_not_load_the_model(cli_env, monkeypatch, capsys):
    main(["transcribe", str(cli_env.audio)])
    no_model(monkeypatch)
    assert main(["label", str(cli_env.session_path), "spk_0=田中"]) == 0
    assert main(["speakers", "list"]) == 0
    assert main(["speakers", "rename", "田中", "鈴木"]) == 0
    assert main(["speakers", "delete", "鈴木"]) == 0
    assert main(["viz", str(cli_env.session_path)]) == 0


def test_index_command_json(cli_env, tmp_path, monkeypatch, capsys):
    root = tmp_path / "kikiwake"
    meeting_dir = root / "records" / "meeting"
    assert main(["transcribe", str(cli_env.audio), "--output-dir", str(meeting_dir)]) == 0
    session_path = meeting_dir / "meeting.session.json"
    assert main(["label", str(session_path), "spk_0=田中"]) == 0
    no_model(monkeypatch)
    capsys.readouterr()

    assert main(["index", str(root), "--json"]) == 0

    captured = capsys.readouterr()
    index_path = (root / "index.md").resolve()
    assert json.loads(captured.out) == {"index_path": str(index_path), "meetings": 1, "skipped": []}
    assert captured.err == ""
    text = index_path.read_text(encoding="utf-8")
    assert re.fullmatch(
        r"# 会議の記録\n\n\| 日時 \| 長さ \| 参加者 \| 要点 \| 文字起こし \|\n\|---\|---\|---\|---\|---\|\n"
        r"\| \d{4}-\d\d-\d\d \d\d:\d\d \| 0 分 \| 田中・ほか1名 \| （要点なし） \| "
        r"\[開く\]\(<records/meeting/meeting\.md>\) \|\n",
        text,
    )

    (root / "records" / "broken").mkdir()
    (root / "records" / "broken" / "x.session.json").write_text("{", encoding="utf-8")
    assert main(["index", str(root), "--json"]) == 0  # a broken record neither stops it nor breaks the JSON
    captured = capsys.readouterr()
    assert json.loads(captured.out)["skipped"] == ["records/broken/x.session.json"]
    assert "警告: 読めない記録を飛ばしました: records/broken/x.session.json" in captured.err


def test_index_command_prints_the_path(cli_env, tmp_path, capsys):
    root = tmp_path / "kikiwake"
    root.mkdir()
    assert main(["index", str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{(root / 'index.md').resolve()}\n"
    assert "（まだ記録がありません）" in (root / "index.md").read_text(encoding="utf-8")


def test_index_missing_root_exit_2(cli_env, tmp_path, capsys):
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("", encoding="utf-8")
    for bad in (tmp_path / "nope", not_a_directory):
        assert main(["index", str(bad), "--json"]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("エラー: ") and str(bad) in captured.err
    assert not (tmp_path / "nope").exists() and not_a_directory.read_text() == ""


def test_missing_audio_exit_2(cli_env, tmp_path, capsys):
    assert main(["transcribe", str(tmp_path / "nope.wav")]) == 2
    assert "音声ファイルが見つかりません" in capsys.readouterr().err


def test_missing_audio_is_found_before_the_model_loads(cli_env, tmp_path, monkeypatch, capsys):
    no_model(monkeypatch)
    for missing in (tmp_path / "nope.wav", tmp_path):  # nothing there / not a file
        assert main(["transcribe", str(missing)]) == 2
        assert capsys.readouterr().err == f"エラー: 音声ファイルが見つかりません: {missing}\n"
    assert cli_env.calls == []


def test_missing_ffmpeg_is_found_before_the_model_loads(cli_env, monkeypatch, capsys):
    no_model(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert main(["transcribe", str(cli_env.audio)]) == 2
    assert "brew install ffmpeg" in capsys.readouterr().err


def test_os_errors_are_reported_without_a_traceback(cli_env, tmp_path, capsys):
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("", encoding="utf-8")
    assert main(["transcribe", str(cli_env.audio), "--output-dir", str(not_a_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    last = captured.err.splitlines()[-1]
    assert last.startswith("エラー: ") and str(not_a_dir) in last
    assert "Traceback" not in captured.err


def test_broken_config_exit_2(cli_env, capsys):
    cli_env.home.mkdir()
    (cli_env.home / "config.toml").write_text("auto_threshold = 'high'", encoding="utf-8")
    assert main(["speakers", "list"]) == 2
    assert "auto_threshold" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "9", "-1", "many"])
def test_max_speakers_must_be_1_to_8(cli_env, capsys, value):
    assert main(["transcribe", str(cli_env.audio), "--max-speakers", value]) == 2
    assert "--max-speakers" in capsys.readouterr().err
    assert cli_env.calls == []


@pytest.mark.parametrize("value", ["1", "8"])
def test_max_speakers_bounds_are_accepted(cli_env, value):
    assert main(["transcribe", str(cli_env.audio), "--max-speakers", value]) == 0
    assert cli_env.calls[0].max_speakers == int(value)


def test_argparse_errors_return_2_instead_of_exiting(cli_env, capsys):
    assert main(["transcribe"]) == 2
    assert "AUDIO" in capsys.readouterr().err
    assert main(["bogus"]) == 2
    assert main(["speakers"]) == 2
    assert main(["label", str(cli_env.session_path)]) == 2


def test_help_returns_0(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "kikiwake" in out and "transcribe" in out and "speakers" in out
    assert main(["transcribe", "--help"]) == 0
    assert "--max-speakers" in capsys.readouterr().out


def test_no_subcommand_prints_help_to_stderr_and_returns_2(capsys):
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage: kikiwake" in captured.err


def test_make_transcriber_asks_for_the_key_only_on_first_call(monkeypatch, tmp_path):
    built = []
    calls = []

    class FakeClient:
        def __init__(self, key, settings):
            built.append((key, settings))

        def transcribe(self, mp3, **kwargs):
            calls.append((mp3, kwargs))
            return {"ok": len(calls)}

    for k in API_KEY_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(cli, "AvalonClient", FakeClient)
    settings = Settings()
    transcriber = cli.make_transcriber(settings)  # no key needed yet
    assert built == []
    opts = cli.TranscribeOptions(language="ja", prompt="p", max_speakers=2)
    with pytest.raises(UsageError, match="AQUAVOICE_API"):
        transcriber(tmp_path / "a.mp3", opts)
    assert built == []

    monkeypatch.setenv("AQUA_API_KEY", "secret")
    assert transcriber(tmp_path / "a.mp3", opts) == {"ok": 1}
    assert transcriber(tmp_path / "b.mp3", opts) == {"ok": 2}
    assert built == [("secret", settings)]  # one client for every call
    assert calls[0] == (tmp_path / "a.mp3", {"language": "ja", "prompt": "p", "max_speakers": 2})


class _NoisyEmbedder:
    def __init__(self, savedir, message):
        self.savedir = savedir
        warnings.warn(message, UserWarning, stacklevel=2)


def test_make_embedder_hides_only_the_speechbrain_symlink_warning(monkeypatch, tmp_path):
    symlink = "The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`."
    monkeypatch.setattr(cli, "EcapaEmbedder", lambda savedir: _NoisyEmbedder(savedir, symlink))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cli.make_embedder(tmp_path).savedir == tmp_path / "models"

    monkeypatch.setattr(cli, "EcapaEmbedder", lambda savedir: _NoisyEmbedder(savedir, "something else"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="something else"):
            cli.make_embedder(tmp_path)
