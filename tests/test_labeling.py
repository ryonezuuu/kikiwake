import copy

import numpy as np
import pytest
from conftest import make_session, unit

from kikiwake.errors import UsageError
from kikiwake.labeling import apply_label, apply_labels, resolve_key
from kikiwake.speaker_db import SpeakerDB


@pytest.fixture
def db(tmp_path):
    return SpeakerDB.load(tmp_path / "db.json", "fake", 4, 20)


def test_new_name_creates_person_and_confirms(db):
    s = make_session(spk_1=dict(status="unknown", number=2))
    apply_labels(s, db, [("話者2", "田中")])
    st = s.speakers["spk_1"]
    assert (st.status, st.name, st.display_label) == ("confirmed", "田中", "田中")
    assert [x.id for x in db.people()["田中"]] == st.db_sample_ids


def test_label_sample_holds_centroid_session_and_source(db):
    s = make_session()
    apply_labels(s, db, [("spk_1", "田中")])
    (sample,) = db.people()["田中"]
    assert sample.source == "label" and sample.session_sha256 == "sha"
    assert sample.embedding.dtype == np.float32 and np.allclose(sample.embedding, unit(1))


def test_relabel_replaces_own_sample(db):
    s = make_session()
    apply_labels(s, db, [("spk_0", "田中")])
    first = s.speakers["spk_0"].db_sample_ids
    apply_labels(s, db, [("spk_0", "田中")])
    second = s.speakers["spk_0"].db_sample_ids
    assert first != second and [x.id for x in db.people()["田中"]] == second


def test_score_is_kept(db):
    s = make_session(spk_0=dict(status="candidate", name="田中", score=0.6))
    apply_labels(s, db, [("spk_0", "田中")])
    assert s.speakers["spk_0"].score == 0.6


def test_correction_removes_wrong_auto_sample(db):
    wrong = db.add_sample("田中", unit(0), "sha", "auto")
    s = make_session(spk_0=dict(status="auto", name="田中", db_sample_ids=[wrong]))
    apply_labels(s, db, [("田中", "佐藤")])
    assert "田中" not in db.people() and s.speakers["spk_0"].name == "佐藤"


def test_merge_keeps_both(db):
    s = make_session(spk_0=dict(status="auto", name="田中"), spk_2=dict(status="unknown", number=3))
    apply_labels(s, db, [("spk_2", "田中")])
    assert s.speakers["spk_0"].display_label == s.speakers["spk_2"].display_label == "田中"
    assert s.speakers["spk_0"].status == "auto"


def test_label_clears_other_candidate(db):
    s = make_session(
        spk_0=dict(status="candidate", name="田中", score=0.6, number=1),
        spk_1=dict(status="unknown", number=2),
    )
    apply_labels(s, db, [("話者2", "田中")])
    other = s.speakers["spk_0"]
    assert other.display_label == "話者1"
    assert (other.status, other.name, other.score) == ("unknown", None, None)


def test_label_leaves_candidates_of_other_names(db):
    s = make_session(spk_0=dict(status="candidate", name="佐藤", score=0.6))
    apply_labels(s, db, [("話者2", "田中")])
    assert s.speakers["spk_0"].display_label == "佐藤？"


def test_insufficient_renames_without_db(db):
    s = make_session(spk_2=dict(status="insufficient", centroid=None, number=3))
    warnings = apply_labels(s, db, [("話者3", "鈴木")])
    assert s.speakers["spk_2"].display_label == "鈴木" and db.people() == {} and "鈴木" in warnings[0]
    assert warnings == ["鈴木 は声紋が不足しているため、DB には登録しません"]
    assert s.speakers["spk_2"].status == "confirmed" and s.speakers["spk_2"].db_sample_ids == []


def test_insufficient_also_clears_other_candidate(db):
    s = make_session(
        spk_0=dict(status="candidate", name="鈴木"),
        spk_2=dict(status="insufficient", centroid=None, number=3),
    )
    apply_labels(s, db, [("話者3", "鈴木")])
    assert s.speakers["spk_0"].display_label == "話者1"


def test_sufficient_label_has_no_warning(db):
    assert apply_labels(make_session(), db, [("話者1", "田中")]) == []


def test_halfwidth_question_mark_key(db):
    s = make_session(spk_0=dict(status="candidate", name="田中"))
    apply_labels(s, db, [("田中?", "田中")])
    assert s.speakers["spk_0"].status == "confirmed"


def test_fullwidth_question_mark_key(db):
    s = make_session(spk_0=dict(status="candidate", name="田中"))
    assert resolve_key(s, "田中？") == "spk_0"


def test_halfwidth_question_mark_inside_name(db):
    s = make_session(spk_0=dict(status="confirmed", name="A?B"))
    assert resolve_key(s, "A?B") == "spk_0"


def test_speaker_id_wins_over_label(db):
    s = make_session(spk_0=dict(status="confirmed", name="spk_1"))
    assert resolve_key(s, "spk_1") == "spk_1"


def test_ambiguous_key(db):
    s = make_session(spk_0=dict(status="auto", name="田中"), spk_1=dict(status="confirmed", name="田中"))
    with pytest.raises(UsageError, match="spk_0") as e:
        apply_labels(s, db, [("田中", "佐藤")])
    assert "spk_1" in str(e.value) and "話者 ID" in str(e.value)
    assert resolve_key(s, "spk_1") == "spk_1"


def test_unknown_key_lists_valid_keys(db):
    s = make_session(spk_0=dict(status="auto", name="田中"))
    with pytest.raises(UsageError) as e:
        resolve_key(s, "nope")
    message = str(e.value)
    for valid in ("spk_0", "spk_1", "spk_2", "田中", "話者2", "話者3"):
        assert valid in message
    assert "話者1" not in message


def test_name_is_stripped(db):
    s = make_session()
    apply_labels(s, db, [("話者1", "  田中\t")])
    assert s.speakers["spk_0"].name == "田中" and list(db.people()) == ["田中"]


def test_multiple_pairs(db):
    s = make_session()
    apply_labels(s, db, [("話者1", "田中"), ("spk_2", "佐藤")])
    assert [st.display_label for st in s.speakers.values()] == ["田中", "話者2", "佐藤"]
    assert set(db.people()) == {"田中", "佐藤"}


def test_invalid_input_changes_nothing(db):
    s = make_session()
    before = copy.deepcopy(s)
    for pairs in (
        [("話者1", "田中"), ("nope", "佐藤")],
        [("話者1", "")],
        [("話者1", "   ")],
        [("話者1", "田中？")],
        [("話者1", "田中?")],
        [("話者1", "田中"), ("話者2", "佐藤?")],
        [("話者1", "田中"), ("spk_0", "佐藤")],
    ):
        with pytest.raises(UsageError):
            apply_labels(s, db, pairs)
    assert s == before and db.people() == {}


def test_two_keys_for_one_speaker(db):
    s = make_session()
    with pytest.raises(UsageError, match="spk_0"):
        apply_labels(s, db, [("話者1", "田中"), ("spk_0", "佐藤")])


def test_apply_does_not_persist(db, tmp_path):
    apply_labels(make_session(), db, [("話者1", "田中")])
    assert not (tmp_path / "db.json").exists()


def test_apply_label_single(db):
    s = make_session()
    assert apply_label(s, db, "spk_1", "田中") == []
    assert s.speakers["spk_1"].display_label == "田中"
    with pytest.raises(UsageError):
        apply_label(s, db, "spk_9", "佐藤")
