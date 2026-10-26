from dataclasses import replace

import numpy as np
import pytest
from conftest import FakeEmbedder, raw_diarized

from speaker_transcribe.config import Settings
from speaker_transcribe.models import parse_diarized
from speaker_transcribe.voiceprint import build_voiceprint, select_segments, split_chunks

S = Settings()


def seg(tr_words, sid="spk_0", **kw):
    out = select_segments(parse_diarized(raw_diarized(tr_words)), sid, replace(S, **kw))
    return [(round(a, 6), round(b, 6)) for a, b in out]  # 浮動小数の誤差を吸収


def test_joins_close_words_and_trims_edges():
    words = [("spk_0", 0.0, 1.0, 0.9, False), ("spk_0", 1.2, 2.0, 0.9, False)]
    assert seg(words) == [(0.1, 1.9)]


def test_gap_splits_segments():
    words = [("spk_0", 0.0, 2.0, 0.9, False), ("spk_0", 2.4, 4.5, 0.9, False)]
    assert seg(words) == [(0.1, 1.9), (2.5, 4.4)]


def test_overlap_and_low_confidence_break_runs():
    words = [("spk_0", 0, 2, 0.9, False), ("spk_0", 2.0, 2.2, 0.9, True), ("spk_0", 2.2, 4.4, 0.9, False),
             ("spk_0", 4.4, 4.6, 0.5, False)]
    assert seg(words) == [(0.1, 1.9), (2.3, 4.3)]


def test_other_speaker_breaks_run():
    words = [("spk_0", 0, 2, 0.9, False), ("spk_1", 2.0, 2.1, 0.9, False), ("spk_0", 2.1, 4.1, 0.9, False)]
    assert seg(words) == [(0.1, 1.9), (2.2, 4.0)]


def test_short_segments_dropped():
    assert seg([("spk_0", 0.0, 1.6, 0.9, False)]) == []  # 1.4 s after trim


def test_total_capped_longest_first_and_truncated():
    words = [("spk_0", 0, 30.2, 0.9, False), ("spk_1", 30.2, 31, 0.9, False),
             ("spk_0", 31, 51.2, 0.9, False), ("spk_1", 51.2, 52, 0.9, False),
             ("spk_0", 52, 72.2, 0.9, False)]
    assert seg(words) == [(0.1, 30.1), (31.1, 51.1), (52.1, 62.1)]  # 最後の 20 秒は先頭 10 秒に切り詰め


def test_all_overlap_speaker_is_insufficient():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 10, 0.9, True)]))
    vp = build_voiceprint(np.zeros(160000, np.float32), tr, "spk_0", FakeEmbedder(), S)
    assert vp.centroid is None and vp.segments == [] and vp.total_seconds == 0.0


def test_under_four_seconds_is_insufficient():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 3.9, 0.9, False)]))
    assert build_voiceprint(np.zeros(80000, np.float32), tr, "spk_0", FakeEmbedder(), S).centroid is None


def test_split_chunks():
    assert split_chunks([(0.0, 7.0)], S) == [(0.0, 3.0), (3.0, 7.0)]  # remainder 1.0 merged
    assert split_chunks([(0.0, 7.5)], S) == [(0.0, 3.0), (3.0, 6.0), (6.0, 7.5)]
    assert split_chunks([(0.0, 2.0)], S) == [(0.0, 2.0)]


def test_centroid_normalized_and_consistency():
    wave = np.full(16000 * 10, 0.5, np.float32)
    tr = parse_diarized(raw_diarized([("spk_0", 0, 10, 0.9, False)]))
    vp = build_voiceprint(wave, tr, "spk_0", FakeEmbedder(), S)
    assert np.linalg.norm(vp.centroid) == pytest.approx(1.0) and vp.consistency == pytest.approx(1.0)


def test_words_are_ordered_by_start_across_turns():
    raw = raw_diarized([("spk_0", 0, 2, 0.9, False), ("spk_1", 2.0, 2.1, 0.9, False), ("spk_0", 2.1, 4.1, 0.9, False)])
    raw["turns"].reverse()
    out = select_segments(parse_diarized(raw), "spk_0", S)
    assert [(round(a, 6), round(b, 6)) for a, b in out] == [(0.1, 1.9), (2.2, 4.0)]


def test_no_segments_is_insufficient_even_with_zero_minimum():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 10, 0.9, True)]))
    vp = build_voiceprint(np.zeros(160000, np.float32), tr, "spk_0", FakeEmbedder(), replace(S, min_total_per_speaker=0.0))
    assert vp.centroid is None and vp.consistency is None
