import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from kikiwake.errors import UsageError
from kikiwake.speaker_db import SpeakerDB, atomic_write_text, locked

DIM = 192


def new_db(tmp_path, max_samples=20, clock=None, model_id="m", dim=DIM):
    path = tmp_path / "db.json"
    if clock is None:
        return SpeakerDB.load(path, model_id, dim, max_samples)
    return SpeakerDB.load(path, model_id, dim, max_samples, now=iter(clock).__next__)


def unit(i):
    vec = np.zeros(DIM, dtype=np.float32)
    vec[i] = 1.0
    return vec


def test_add_save_load_roundtrip(tmp_path):
    db = new_db(tmp_path)
    sid = db.add_sample("田中", unit(0), "sha1", "label")
    db.save()
    db2 = new_db(tmp_path)
    s = db2.people()["田中"][0]
    assert s.id == sid and s.source == "label" and np.allclose(s.embedding, unit(0))
    assert json.loads((tmp_path / "db.json").read_text())["model_id"] == "m"


def test_max_samples_drops_oldest(tmp_path):
    db = new_db(tmp_path, max_samples=2, clock=["t1", "t2", "t3"])
    ids = [db.add_sample("田中", unit(0), f"s{i}", "auto") for i in range(3)]
    assert [s.id for s in db.people()["田中"]] == ids[1:]


def test_remove_samples_ignores_missing_and_drops_empty_person(tmp_path):
    db = new_db(tmp_path)
    sid = db.add_sample("田中", unit(0), "s", "auto")
    db.remove_samples([sid, "nope"])
    assert "田中" not in db.people()


def test_remove_by_session(tmp_path):
    db = new_db(tmp_path)
    a = db.add_sample("田中", unit(0), "A", "auto")
    db.add_sample("田中", unit(0), "B", "label")
    b = db.add_sample("佐藤", unit(1), "A", "label")
    removed = db.remove_by_session("A")
    assert {name: [(s.id, s.source) for s in samples] for name, samples in removed.items()} == {
        "田中": [(a, "auto")],
        "佐藤": [(b, "label")],
    }
    assert np.array_equal(removed["佐藤"][0].embedding, unit(1))
    assert list(db.people()) == ["田中"] and db.people()["田中"][0].session_sha256 == "B"


def test_model_mismatch(tmp_path):
    new_db(tmp_path).save()
    with pytest.raises(UsageError, match="other"):
        SpeakerDB.load(tmp_path / "db.json", "other", 192, 20)


def test_rename_and_delete(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "s", "auto")
    db.add_sample("佐藤", unit(1), "s", "auto")
    with pytest.raises(UsageError):
        db.rename("田中", "佐藤")
    db.rename("田中", "鈴木")
    db.delete("佐藤")
    assert list(db.people()) == ["鈴木"]
    with pytest.raises(UsageError):
        db.delete("佐藤")


def test_atomic_write_leaves_no_temp(tmp_path):
    atomic_write_text(tmp_path / "sub" / "f.json", "x")
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["f.json"]


# --- beyond the brief ---------------------------------------------------------------------


def test_load_missing_file_is_empty(tmp_path):
    db = new_db(tmp_path)
    assert db.people() == {} and db.embeddings() == {}
    assert not (tmp_path / "db.json").exists()  # load never writes


def test_dim_mismatch_names_both_dims(tmp_path):
    new_db(tmp_path).save()
    with pytest.raises(UsageError, match=r"192.*64|64.*192"):
        new_db(tmp_path, dim=64)


def test_model_mismatch_names_both_models_and_path(tmp_path):
    new_db(tmp_path, model_id="old-model").save()
    with pytest.raises(UsageError) as e:
        new_db(tmp_path, model_id="new-model")
    msg = str(e.value)
    assert "old-model" in msg and "new-model" in msg and str(tmp_path / "db.json") in msg


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"version": 2, "model_id": "m", "dim": 192, "people": {}}',
        '{"model_id": "m", "dim": 192, "people": {}}',
        '{"version": 1, "model_id": "m", "dim": 192}',
        '{"version": 1, "model_id": "m", "dim": 192, "people": {"田中": {"samples": []}}}',
        '{"version": 1, "model_id": "m", "dim": 192, "people": {"田中": '
        '{"created_at": "t", "samples": [{"id": "x"}]}}}',
        '{"version": 1, "model_id": "m", "dim": 192, "people": {"田中": '
        '{"created_at": "t", "samples": [{"id": "x", "session_sha256": "s", "source": "bogus", '
        '"created_at": "t", "embedding": [0.0]}]}}}',
    ],
)
def test_malformed_db_is_usage_error_naming_path(tmp_path, text):
    path = tmp_path / "db.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UsageError) as e:
        SpeakerDB.load(path, "m", DIM, 20)
    assert str(path) in str(e.value)


def test_unreadable_db_is_usage_error(tmp_path):
    path = tmp_path / "db.json"
    path.mkdir()  # reading a directory fails with OSError
    with pytest.raises(UsageError, match="db.json"):
        SpeakerDB.load(path, "m", DIM, 20)


def test_file_format_matches_spec(tmp_path):
    db = new_db(tmp_path, clock=["t0"])
    sid = db.add_sample("田中", unit(1), "sha1", "auto")
    db.save()
    text = (tmp_path / "db.json").read_text(encoding="utf-8")
    assert "田中" in text  # ensure_ascii=False
    data = json.loads(text)
    assert data["version"] == 1 and data["model_id"] == "m" and data["dim"] == DIM
    person = data["people"]["田中"]
    assert person["created_at"] == "t0"
    (sample,) = person["samples"]
    assert sample == {
        "id": sid,
        "session_sha256": "sha1",
        "source": "auto",
        "created_at": "t0",
        "embedding": unit(1).tolist(),
    }
    assert all(type(x) is float for x in sample["embedding"])


def test_loaded_embedding_is_float32_1d(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(3), "s", "auto")
    db.save()
    emb = new_db(tmp_path).people()["田中"][0].embedding
    assert emb.dtype == np.float32 and emb.shape == (DIM,)


def test_add_sample_returns_uuid4_and_does_not_check_dim(tmp_path):
    db = new_db(tmp_path)
    sid = db.add_sample("田中", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), "s", "auto")
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", sid)
    assert db.people()["田中"][0].embedding.shape == (4,)


def test_add_sample_stores_a_copy_of_the_embedding(tmp_path):
    db = new_db(tmp_path)
    vec = unit(0)
    db.add_sample("田中", vec, "s", "auto")
    vec[0] = 5.0
    assert db.people()["田中"][0].embedding[0] == 1.0


def test_add_sample_rejects_unknown_source(tmp_path):
    db = new_db(tmp_path)
    with pytest.raises(ValueError):
        db.add_sample("田中", unit(0), "s", "manual")
    assert db.people() == {}


def test_person_created_at_set_once(tmp_path):
    db = new_db(tmp_path, clock=["t1", "t2"])
    db.add_sample("田中", unit(0), "a", "auto")
    db.add_sample("田中", unit(0), "b", "auto")
    db.save()
    data = json.loads((tmp_path / "db.json").read_text(encoding="utf-8"))
    assert data["people"]["田中"]["created_at"] == "t1"
    assert [s["created_at"] for s in data["people"]["田中"]["samples"]] == ["t1", "t2"]


def test_default_now_is_local_iso8601_seconds(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "s", "auto")
    created = db.people()["田中"][0].created_at
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", created)


def test_eviction_ties_broken_by_insertion_order(tmp_path):
    db = new_db(tmp_path, max_samples=2, clock=["t", "t", "t"])
    ids = [db.add_sample("田中", unit(0), f"s{i}", "auto") for i in range(3)]
    assert [s.id for s in db.people()["田中"]] == ids[1:]


def test_eviction_uses_created_at_not_position(tmp_path):
    # A sample loaded from disk with an old created_at is evicted even if it is not first in the list.
    path = tmp_path / "db.json"
    sample = lambda sid, at: {  # noqa: E731
        "id": sid,
        "session_sha256": "s",
        "source": "auto",
        "created_at": at,
        "embedding": [1.0],
    }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "m",
                "dim": DIM,
                "people": {
                    "田中": {"created_at": "p", "samples": [sample("new", "2026-02"), sample("old", "2026-01")]}
                },
            }
        ),
        encoding="utf-8",
    )
    db = SpeakerDB.load(path, "m", DIM, 2, now=iter(["2026-03"]).__next__)
    db.add_sample("田中", unit(0), "s", "auto")
    ids = [s.id for s in db.people()["田中"]]
    assert "old" not in ids and ids[0] == "new" and len(ids) == 2


def test_people_and_embeddings_return_fresh_containers(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "s", "auto")
    people = db.people()
    people["田中"].clear()
    people["偽"] = []
    embs = db.embeddings()
    embs["田中"].clear()
    embs.clear()
    assert list(db.people()) == ["田中"] and len(db.people()["田中"]) == 1
    assert len(db.embeddings()["田中"]) == 1
    assert np.array_equal(db.embeddings()["田中"][0], unit(0))


def test_samples_keep_insertion_order(tmp_path):
    db = new_db(tmp_path)
    ids = [db.add_sample("田中", unit(i), f"s{i}", "auto") for i in range(3)]
    db.save()
    assert [s.id for s in new_db(tmp_path).people()["田中"]] == ids


def test_remove_samples_keeps_other_samples_of_the_person(tmp_path):
    db = new_db(tmp_path)
    a = db.add_sample("田中", unit(0), "s", "auto")
    b = db.add_sample("田中", unit(1), "s", "auto")
    db.remove_samples([a])
    assert [s.id for s in db.people()["田中"]] == [b]


def test_remove_by_session_with_no_match_returns_nothing(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "A", "auto")
    assert db.remove_by_session("Z") == {}
    assert list(db.people()) == ["田中"]


def test_rename_keeps_samples_and_created_at(tmp_path):
    db = new_db(tmp_path, clock=["t1"])
    sid = db.add_sample("田中", unit(0), "s", "auto")
    db.rename("田中", "鈴木")
    db.save()
    data = json.loads((tmp_path / "db.json").read_text(encoding="utf-8"))
    assert list(data["people"]) == ["鈴木"]
    assert data["people"]["鈴木"]["created_at"] == "t1"
    assert data["people"]["鈴木"]["samples"][0]["id"] == sid


def test_rename_errors_name_the_people(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "s", "auto")
    db.add_sample("佐藤", unit(1), "s", "auto")
    with pytest.raises(UsageError, match="山田"):
        db.rename("山田", "鈴木")
    with pytest.raises(UsageError, match="佐藤"):
        db.rename("田中", "佐藤")
    assert list(db.people()) == ["田中", "佐藤"]


def test_rename_to_same_name_is_noop_only_if_it_exists(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "s", "auto")
    db.rename("田中", "田中")
    assert list(db.people()) == ["田中"]
    with pytest.raises(UsageError, match="山田"):
        db.rename("山田", "山田")


def test_delete_missing_names_the_person(tmp_path):
    db = new_db(tmp_path)
    with pytest.raises(UsageError, match="山田"):
        db.delete("山田")


def test_atomic_write_replaces_existing_file(tmp_path):
    path = tmp_path / "f.json"
    path.write_text("old", encoding="utf-8")
    atomic_write_text(path, "新しい")
    assert path.read_text(encoding="utf-8") == "新しい"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_atomic_write_failure_keeps_old_file_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "f.json"
    path.write_text("old", encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk on fire"):
        atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_save_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "home" / "db.json"
    db = SpeakerDB.load(path, "m", DIM, 20)
    db.add_sample("田中", unit(0), "s", "auto")
    db.save()
    assert SpeakerDB.load(path, "m", DIM, 20).people().keys() == {"田中"}


WAITING = "別の kikiwake が DB を使用中のため待機しています…\n"


def src_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}


def test_locked_makes_the_lock_file_beside_the_db_and_releases_it(tmp_path, capsys):
    db_path = tmp_path / "new home" / "db.json"
    with locked(db_path):
        assert (tmp_path / "new home" / "db.json.lock").is_file()
    with pytest.raises(RuntimeError), locked(db_path):
        raise RuntimeError  # released on errors too
    with locked(db_path):
        pass
    assert capsys.readouterr().err == ""  # never had to wait
    assert not db_path.exists()


def test_locked_makes_another_process_wait_and_say_so(tmp_path):
    db_path = tmp_path / "db.json"
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from kikiwake.speaker_db import locked
        with locked(Path(sys.argv[1])):
            print("acquired")
        """
    )
    with locked(db_path):
        other = subprocess.Popen(
            [sys.executable, "-c", code, str(db_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=src_env(),
        )
        assert other.stderr.readline() == WAITING
        assert other.poll() is None  # still blocked while this process holds the lock
    out, err = other.communicate(timeout=60)
    assert (out, err, other.returncode) == ("acquired\n", "", 0)


def test_speaker_db_does_not_import_torch():
    code = "import sys, kikiwake.speaker_db; sys.exit('torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
