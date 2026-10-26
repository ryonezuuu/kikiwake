import numpy as np
import pytest

from kikiwake.config import Settings
from kikiwake.matcher import Match, assign, person_score, score_table

DIM = 192


def unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def mix(x):
    """Unit vector whose cosine similarity with ``unit(0)`` is ``x``."""
    return x * unit(0) + np.sqrt(1 - x * x) * unit(1)


def test_person_score_top3_mean():
    c = unit(0)
    embs = [unit(0), unit(0), mix(0.5), unit(1)]  # sims 1, 1, 0.5, 0
    assert person_score(c, embs, 3) == pytest.approx(2.5 / 3)
    assert person_score(c, [mix(0.5)], 3) == pytest.approx(0.5)


def test_person_score_is_a_real_cosine_for_unnormalised_inputs():
    score = person_score(3 * unit(0), [5 * mix(0.5)], 3)
    assert score == pytest.approx(0.5)
    assert type(score) is float


def test_score_table():
    centroids = {"spk_0": unit(0), "spk_1": unit(1)}
    people = {"田中": [unit(0)], "佐藤": [mix(0.5)], "鈴木": []}
    table = score_table(centroids, people, 3)
    assert list(table) == ["spk_0", "spk_1"]
    assert table["spk_0"] == {"田中": pytest.approx(1.0), "佐藤": pytest.approx(0.5)}
    assert table["spk_1"] == {"田中": pytest.approx(0.0), "佐藤": pytest.approx(np.sqrt(0.75))}
    assert all(type(v) is float for row in table.values() for v in row.values())


def test_score_table_without_people():
    assert score_table({"spk_0": unit(0)}, {}, 3) == {"spk_0": {}}


def test_thresholds():
    m = assign({"a": {"田中": 0.60}, "b": {"佐藤": 0.59}, "c": {"鈴木": 0.44}}, Settings())
    assert m["a"] == Match("田中", 0.60, "auto")
    assert m["b"] == Match("佐藤", 0.59, "candidate")
    assert m["c"] == Match(None, None, "unknown")


def test_candidate_lower_bound_is_inclusive():
    m = assign({"a": {"田中": 0.45}}, Settings())
    assert m["a"] == Match("田中", 0.45, "candidate")


def test_one_to_one_maximizes_total():
    m = assign({"a": {"田中": 0.9, "佐藤": 0.8}, "b": {"田中": 0.85, "佐藤": 0.1}}, Settings())
    assert (m["a"].name, m["b"].name) == ("佐藤", "田中")


def test_more_speakers_than_people():
    m = assign({"a": {"田中": 0.9}, "b": {"田中": 0.8}}, Settings())
    assert m["a"].name == "田中" and m["b"] == Match(None, None, "unknown")


def test_more_people_than_speakers():
    m = assign({"a": {"田中": 0.5, "佐藤": 0.7}}, Settings())
    assert m == {"a": Match("佐藤", 0.7, "auto")}


def test_speaker_missing_a_person_is_never_given_that_person():
    m = assign({"a": {"田中": 0.9}, "b": {}}, Settings())
    assert m["a"] == Match("田中", 0.9, "auto")
    assert m["b"] == Match(None, None, "unknown")


def test_empty_db():
    assert assign({"a": {}}, Settings()) == {"a": Match(None, None, "unknown")}
    assert assign({}, Settings()) == {}
