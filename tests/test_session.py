import copy
import json
from dataclasses import replace

import pytest
from conftest import make_session, make_state

from kikiwake.errors import UsageError
from kikiwake.session import load_session, save_session, session_paths


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
    path.write_text(json.dumps(data), encoding="utf-8")
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
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(UsageError, match="x.session.json"):
        load_session(path)
