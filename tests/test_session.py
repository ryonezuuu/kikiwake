import copy
import json
from dataclasses import replace

import pytest
from conftest import make_session, make_state

from kikiwake.errors import UsageError
from kikiwake.session import load_session, save_session, session_paths
from kikiwake.speaker_db import backup_path


def hand_edit(path, data) -> None:
    """Overwrite a saved session and its backup with ``data``, as if edited by hand."""
    for target in (path, backup_path(path)):
        target.write_text(json.dumps(data), encoding="utf-8")


def test_session_paths_keep_spaces_and_dots(tmp_path):
    md, ss, png = session_paths(tmp_path / "会議 10.26.m4a", None)
    assert (md.name, ss.name, png.name) == (
        "会議 10.26.md",
        "会議 10.26.session.json",
        "会議 10.26.voices.png",
    )
    assert md.parent == ss.parent == png.parent == tmp_path
    assert session_paths(tmp_path / "a.m4a", tmp_path / "out")[0] == tmp_path / "out" / "a.md"


def test_display_labels():
    assert make_state(status="auto", name="田中").display_label == "田中"
    assert make_state(status="confirmed", name="田中").display_label == "田中"
    assert make_state(status="candidate", name="田中").display_label == "田中？"
    assert make_state(status="unknown", number=2).display_label == "話者2"
    assert make_state(status="insufficient", number=3).display_label == "話者3"


def test_roundtrip(tmp_path):
    s = make_session(
        spk_0=dict(status="auto", name="田中", score=0.91, scores={"田中": 0.91}, db_sample_ids=["a", "b"]),
        spk_1=dict(status="insufficient", centroid=None, consistency=None, segments=[(0.5, 1.5), (3.0, 4.25)]),
    )
    s.warnings.append({"code": "low_consistency", "speaker": "spk_2"})
    path = tmp_path / "x.session.json"
    save_session(s, path)
    loaded = load_session(path)
    assert loaded == s
    assert loaded.speakers["spk_1"].segments == [(0.5, 1.5), (3.0, 4.25)]
    assert all(isinstance(seg, tuple) for seg in loaded.speakers["spk_1"].segments)


def test_saved_json_is_readable_and_has_display_label(tmp_path):
    s = make_session(spk_0=dict(status="candidate", name="田中"))
    path = tmp_path / "x.session.json"
    save_session(s, path)
    text = path.read_text(encoding="utf-8")
    assert "田中" in text and "\n  " in text  # ensure_ascii=False, indent=2
    data = json.loads(text)
    assert data["version"] == 1
    assert data["speakers"]["spk_0"]["display_label"] == "田中？"
    assert data["speakers"]["spk_1"]["display_label"] == "話者2"


def test_transcription_parses_avalon_response():
    s = make_session()
    assert s.transcription.speaker_ids == ("spk_0", "spk_1", "spk_2")
    assert copy.deepcopy(s) == s


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"version": 2}',
        '{"version": 1}',
        '{"version": 1, "speakers": 3}',
    ],
)
def test_load_rejects_bad_file(tmp_path, text):
    path = tmp_path / "x.session.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UsageError, match="x.session.json"):
        load_session(path)


def test_load_rejects_unknown_status(tmp_path):
    path = tmp_path / "x.session.json"
    save_session(make_session(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["speakers"]["spk_0"]["status"] = "bogus"
    hand_edit(path, data)
    with pytest.raises(UsageError, match="x.session.json"):
        load_session(path)


def test_load_missing_file(tmp_path):
    with pytest.raises(UsageError, match="nope.session.json"):
        load_session(tmp_path / "nope.session.json")


RECORDED_AT = "2026-10-16T12:55:20+09:00"


def test_recorded_at_roundtrips(tmp_path):
    s = replace(make_session(), recorded_at=RECORDED_AT)
    path = tmp_path / "x.session.json"
    save_session(s, path)
    assert json.loads(path.read_text(encoding="utf-8"))["recorded_at"] == RECORDED_AT
    assert load_session(path) == s


def test_recorded_at_defaults_to_none_and_is_always_written(tmp_path):
    s = make_session()
    assert s.recorded_at is None
    path = tmp_path / "x.session.json"
    save_session(s, path)
    assert json.loads(path.read_text(encoding="utf-8"))["recorded_at"] is None
    assert load_session(path).recorded_at is None


def test_session_without_recorded_at_still_loads(tmp_path):
    path = tmp_path / "x.session.json"
    save_session(replace(make_session(), recorded_at=RECORDED_AT), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["recorded_at"]  # a session.json written before the key existed
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_session(path) == make_session()


@pytest.mark.parametrize("value", [123, 1.5, True, [], {}], ids=repr)
def test_load_rejects_non_string_recorded_at(tmp_path, value):
    path = tmp_path / "x.session.json"
    save_session(make_session(), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["recorded_at"] = value
    hand_edit(path, data)
    with pytest.raises(UsageError, match="x.session.json"):
        load_session(path)


# --- the backup copy -----------------------------------------------------------------------

SESSION_BACKUP = ".x.session.json.bak"
SESSION_FALLBACK = "警告: x.session.json が壊れていたため控えから読み込みました\n"


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A saved session (``x.session.json``, with its backup) in the default write mode."""
    monkeypatch.delenv("KIKIWAKE_INPLACE_WRITES", raising=False)
    path = tmp_path / "x.session.json"
    save_session(make_session(spk_0=dict(status="confirmed", name="田中")), path)
    return path


def test_save_session_keeps_a_backup(saved):
    backup = saved.with_name(SESSION_BACKUP)
    assert backup.read_text(encoding="utf-8") == saved.read_text(encoding="utf-8")
    assert sorted(p.name for p in saved.parent.iterdir()) == [SESSION_BACKUP, "x.session.json"]


@pytest.mark.parametrize("torn", [b"", b'{"version": 1, "audio_pa', "田中".encode()[:-1], b"[]"])
def test_torn_session_is_read_from_the_backup(saved, capsys, torn):
    saved.write_bytes(torn)
    loaded = load_session(saved)
    assert loaded.speakers["spk_0"].name == "田中"
    assert capsys.readouterr().err == SESSION_FALLBACK
    save_session(loaded, saved)  # the next save repairs it
    assert load_session(saved) == loaded and capsys.readouterr().err == ""


def test_torn_backup_with_a_good_session_is_ignored(saved, capsys):
    saved.with_name(SESSION_BACKUP).write_bytes(b"{")
    assert load_session(saved).speakers["spk_0"].name == "田中"
    assert capsys.readouterr().err == ""


def test_torn_session_and_backup_raise_the_original_error(saved, capsys):
    saved.write_text("not json", encoding="utf-8")
    saved.with_name(SESSION_BACKUP).write_text("[]", encoding="utf-8")
    with pytest.raises(UsageError, match="x.session.json: セッションの形式が不正です: JSONDecodeError"):
        load_session(saved)
    assert capsys.readouterr().err == ""


def test_missing_session_does_not_read_the_backup(saved):
    saved.rename(saved.with_name("moved"))
    with pytest.raises(UsageError, match="セッションを読み込めません"):
        load_session(saved)


def test_session_of_another_version_does_not_fall_back(saved):
    data = json.loads(saved.read_text(encoding="utf-8"))
    data["version"] = 2
    saved.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(UsageError, match="未対応のセッションバージョン"):
        load_session(saved)
