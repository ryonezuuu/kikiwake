import pytest
from conftest import raw_diarized

from kikiwake.errors import AppError
from kikiwake.models import Word, parse_diarized


def test_parse_orders_speakers_by_first_seen():
    raw = raw_diarized([("spk_1", 0.0, 0.5, 0.9, False), ("spk_0", 1.0, 1.5, 0.9, False)])
    raw["speakers"].reverse()
    assert parse_diarized(raw).speaker_ids == ("spk_1", "spk_0")


def test_parse_turns_and_words():
    tr = parse_diarized(raw_diarized([("spk_0", 0.0, 0.5, 0.8, False), ("spk_0", 0.6, 1.0, 0.9, True)]))
    assert len(tr.turns) == 1 and tr.turns[0].words[1] == Word("w", 0.6, 1.0, 0.9, True)


def test_parse_empty_transcript():
    tr = parse_diarized(raw_diarized([], duration=30.0))
    assert tr.turns == () and tr.speaker_ids == () and tr.duration == 30.0


def test_parse_warnings():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)], warnings=["speaker_limit_reached"])
    assert parse_diarized(raw).warnings == ("speaker_limit_reached",)


def test_parse_missing_field():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)]); del raw["turns"]
    with pytest.raises(AppError, match="turns"):
        parse_diarized(raw)


def test_parse_accepts_int_numbers_as_float():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 1, 1, False)]))
    word = tr.turns[0].words[0]
    assert word == Word("w", 0.0, 1.0, 1.0, False)
    assert all(isinstance(v, float) for v in (word.start, word.end, word.speaker_confidence))
    assert isinstance(tr.duration, float)


def test_parse_wrong_type_names_nested_field():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)])
    raw["turns"][0]["words"][0]["start"] = "0.0"
    with pytest.raises(AppError, match=r"turns\[0\]\.words\[0\]\.start"):
        parse_diarized(raw)


def test_parse_rejects_bool_as_number():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)])
    raw["duration"] = True
    with pytest.raises(AppError, match="duration"):
        parse_diarized(raw)
