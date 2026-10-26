import json
from pathlib import Path

from conftest import session_from_words

from kikiwake.render import (
    build_summary,
    format_ts,
    render_markdown,
    warning_messages,
)

TITLE = "# meeting.m4a の文字起こし\n\n"


def _meeting():
    return session_from_words(
        [
            ("spk_0", 12.0, 13.0, 0.9, False),
            ("spk_0", 13.0, 14.0, 0.9, False),
            ("spk_1", 45.0, 46.0, 0.9, True),
            ("spk_1", 46.0, 47.0, 0.9, True),
            ("spk_1", 47.0, 48.0, 0.9, False),
            ("spk_0", 50.0, 51.0, 0.9, False),
        ],
        duration=3120.0,
        states={
            "spk_0": dict(status="auto", name="田中"),
            "spk_1": dict(status="unknown", number=2),
        },
    )


def test_format_ts():
    assert format_ts(0) == "00:00:00" and format_ts(3725.9) == "01:02:05"


def test_markdown_golden():
    assert render_markdown(_meeting()) == (
        "# meeting.m4a の文字起こし\n\n"
        "- 録音: 52 分 / 話者: 田中・話者2\n\n"
        "> ⚠ 名前が未確定の話者が 1 人います。\n\n"
        "**[00:00:12] 田中**\nww\n\n"
        "**[00:00:45] 話者2**\nwww ※同時発話あり\n\n"
        "**[00:00:50] 田中**\nw\n"
    )


def test_merged_speakers_form_one_paragraph():
    # spk_0 and spk_2 are both confirmed as 田中 and speak back to back.
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, False), ("spk_2", 1, 2, 0.9, False), ("spk_1", 2, 3, 0.9, False)],
        states={
            "spk_0": dict(status="confirmed", name="田中"),
            "spk_1": dict(status="auto", name="佐藤"),
            "spk_2": dict(status="auto", name="田中", number=3),
        },
    )
    md = render_markdown(s)
    assert md.count("] 田中**") == 1
    assert "**[00:00:00] 田中**\nww\n" in md
    assert "- 録音: 0 分 / 話者: 田中・佐藤\n" in md


def test_paragraph_texts_are_joined_by_language():
    words = [("spk_0", 0, 1, 0.9, False), ("spk_1", 1, 2, 0.9, False), ("spk_2", 2, 3, 0.9, False)]
    same = {"spk_0": dict(status="auto", name="田中"), "spk_1": dict(status="auto", name="田中"),
            "spk_2": dict(status="auto", name="田中")}
    assert "田中**\nwww\n" in render_markdown(session_from_words(words, states=same))
    assert "田中**\nw w w\n" in render_markdown(session_from_words(words, states=same, language="en"))
    assert "田中**\nwww\n" in render_markdown(session_from_words(words, states=same, language="ja-JP"))


def test_empty_transcript():
    s = session_from_words([], duration=30.0, states={})
    assert render_markdown(s) == "# meeting.m4a の文字起こし\n\n- 録音: 0 分 / 話者: なし\n"


def test_header_uses_basename_and_distinct_labels():
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, False), ("spk_1", 1, 2, 0.9, False)],
        audio_path="/rec/会議 10.26.m4a",
        states={"spk_0": dict(status="auto", name="田中"), "spk_1": dict(status="auto", name="田中")},
    )
    md = render_markdown(s)
    assert md.startswith("# 会議 10.26.m4a の文字起こし\n\n- 録音: 0 分 / 話者: 田中\n")


def test_overlap_mark_needs_strict_majority():
    half = session_from_words(
        [("spk_0", 0, 1, 0.9, True), ("spk_0", 1, 2, 0.9, False)],
    )
    assert "※" not in render_markdown(half)
    majority = session_from_words(
        [("spk_0", 0, 1, 0.9, True), ("spk_0", 1, 2, 0.9, True), ("spk_0", 2, 3, 0.9, False)],
    )
    assert render_markdown(majority).endswith("\nwww ※同時発話あり\n")


def test_overlap_counts_words_across_merged_turns():
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, True), ("spk_1", 1, 2, 0.9, False), ("spk_2", 2, 4, 0.9, True)],
        states={k: dict(status="auto", name="田中") for k in ("spk_0", "spk_1", "spk_2")},
    )
    assert "www ※同時発話あり" in render_markdown(s)


def test_turn_without_words_gets_no_overlap_mark():
    s = session_from_words([("spk_0", 0, 1, 0.9, True)])
    s.avalon_response["turns"][0]["words"] = []
    assert render_markdown(s).endswith("**[00:00:00] 話者1**\nw\n")


def test_blank_turn_is_skipped():
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, False), ("spk_1", 1, 2, 0.9, False), ("spk_0", 2, 3, 0.9, False)],
    )
    s.avalon_response["turns"][1]["text"] = "  \n"
    # the blank turn vanishes, so the surrounding spk_0 turns become one paragraph
    assert render_markdown(s).endswith("\n\n**[00:00:00] 話者1**\nww\n")


def test_turn_text_is_stripped():
    s = session_from_words([("spk_0", 0, 1, 0.9, False)])
    s.avalon_response["turns"][0]["text"] = " こんにちは\n"
    assert render_markdown(s).endswith("**[00:00:00] 話者1**\nこんにちは\n")


def test_unlisted_turn_speaker_uses_raw_id():
    s = session_from_words([("spk_0", 0, 1, 0.9, False), ("ghost", 1, 2, 0.9, False)])
    del s.speakers["ghost"]
    md = render_markdown(s)
    assert "**[00:00:01] ghost**\nw\n" in md
    assert "話者: 話者1\n" in md


def test_markdown_lists_each_warning_on_its_own_line():
    s = _meeting()
    s.warnings = [{"code": "labels_discarded"}]
    assert (
        "> ⚠ 名前が未確定の話者が 1 人います。\n"
        "> ⚠ API を呼び直したため、前回の名前付けは破棄されました。\n\n"
        "**[00:00:12] 田中**"
    ) in render_markdown(s)


def _warned():
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, False), ("spk_1", 1, 2, 0.9, False)],
        states={"spk_0": dict(status="auto", name="田中"), "spk_1": dict(status="unknown", number=2)},
    )
    s.warnings = [
        {"code": "labels_lost", "names": ["佐藤", "山田 太郎"]},
        {"code": "low_consistency", "speaker": "spk_0"},
        {"code": "speaker_limit_reached"},
        {"code": "labels_discarded"},
    ]
    return s


def test_warning_order():
    assert warning_messages(_warned()) == [
        "名前が未確定の話者が 1 人います。",
        "9 人以上いた可能性があり、一部の発言は別の話者に割り当てられています。",
        "田中 の声に複数人が混ざっている可能性があります。",
        "API を呼び直したため、前回の名前付けは破棄されました。",
        "前回の名前付け（佐藤・山田 太郎）を引き継げませんでした。",
    ]


def test_warning_order_ignores_entry_order():
    s = _warned()
    s.warnings.reverse()
    assert warning_messages(s) == warning_messages(_warned())


def test_warning_counts_unconfirmed_and_skips_when_none():
    s = session_from_words(
        [("spk_0", 0, 1, 0.9, False), ("spk_1", 1, 2, 0.9, False), ("spk_2", 2, 3, 0.9, False)],
        states={
            "spk_0": dict(status="candidate", name="田中"),
            "spk_1": dict(status="insufficient"),
            "spk_2": dict(status="confirmed", name="佐藤"),
        },
    )
    assert warning_messages(s) == ["名前が未確定の話者が 2 人います。"]
    s.speakers["spk_0"].status = "auto"
    s.speakers["spk_1"].status = "confirmed"
    assert warning_messages(s) == []


def test_low_consistency_keeps_appearance_order_and_raw_id_for_unknown_speaker():
    s = _warned()
    s.warnings = [
        {"code": "low_consistency", "speaker": "spk_1"},
        {"code": "low_consistency", "speaker": "spk_9"},
        {"code": "low_consistency", "speaker": "spk_0"},
    ]
    assert warning_messages(s)[1:] == [
        "話者2 の声に複数人が混ざっている可能性があります。",
        "spk_9 の声に複数人が混ざっている可能性があります。",
        "田中 の声に複数人が混ざっている可能性があります。",
    ]


def _summary_session():
    s = session_from_words(
        [
            ("spk_0", 0.0, 1.0, 0.9, False),
            ("spk_1", 45.0, 46.0, 0.9, True),
            ("spk_0", 50.0, 51.0, 0.9, False),
        ],
        states={
            "spk_0": dict(status="auto", name="田中", score=0.72),
            "spk_1": dict(status="unknown", number=2),
        },
    )
    return s


def test_summary_examples():
    s = _summary_session()
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    unknown = next(x for x in sm["speakers"] if x["id"] == "spk_1")
    assert unknown["label"] == "話者2" and unknown["examples"][0]["time"] == "00:00:45"
    assert "examples" not in next(x for x in sm["speakers"] if x["id"] == "spk_0")
    assert sm["viz_path"] is None and sm["markdown_path"] == "m.md"
    assert sm["recorded_at"] is None  # an old session without it


def test_summary_shape():
    s = _summary_session()
    s.warnings = [{"code": "labels_discarded"}]
    s.recorded_at = "2026-10-26T10:20:27+09:00"
    sm = build_summary(s, Path("out/m.md"), Path("out/m.session.json"), Path("out/m.voices.png"))
    assert sm == {
        "markdown_path": "out/m.md",
        "session_path": "out/m.session.json",
        "viz_path": "out/m.voices.png",
        "recorded_at": "2026-10-26T10:20:27+09:00",
        "speakers": [
            {"id": "spk_0", "label": "田中", "status": "auto", "score": 0.72},
            {
                "id": "spk_1",
                "label": "話者2",
                "status": "unknown",
                "score": None,
                "examples": [{"time": "00:00:45", "text": "w"}],
            },
        ],
        "warnings": ["名前が未確定の話者が 1 人います。", "API を呼び直したため、前回の名前付けは破棄されました。"],
    }
    assert json.loads(json.dumps(sm, ensure_ascii=False)) == sm


def test_summary_examples_are_longest_first_and_capped_at_three():
    words = []
    for i, n in enumerate([1, 3, 2, 3, 4]):
        t = 10.0 * i
        words += [("spk_1", t + k * 0.1, t + k * 0.1 + 0.1, 0.9, False) for k in range(n)]
        words.append(("spk_0", t + 5, t + 6, 0.9, False))
    s = session_from_words(words, states={"spk_1": dict(status="candidate", name="佐藤", score=0.5)})
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    cand = next(x for x in sm["speakers"] if x["id"] == "spk_1")
    assert cand["label"] == "佐藤？"
    # lengths 4, 3 (at 10 s), 3 (at 30 s): ties go to the earlier turn
    assert cand["examples"] == [
        {"time": "00:00:40", "text": "wwww"},
        {"time": "00:00:10", "text": "www"},
        {"time": "00:00:30", "text": "www"},
    ]


def test_summary_examples_for_every_unconfirmed_status():
    s = session_from_words(
        [(f"spk_{i}", float(i), i + 1.0, 0.9, False) for i in range(5)],
        states={
            "spk_0": dict(status="auto", name="a"),
            "spk_1": dict(status="candidate", name="b"),
            "spk_2": dict(status="unknown"),
            "spk_3": dict(status="insufficient"),
            "spk_4": dict(status="confirmed", name="e"),
        },
    )
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    assert [("examples" in x) for x in sm["speakers"]] == [False, True, True, True, False]


def test_example_truncated_to_60():
    s = _summary_session()
    s.avalon_response["turns"][1]["text"] = "あ" * 80
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    unknown = next(x for x in sm["speakers"] if x["id"] == "spk_1")
    assert unknown["examples"][0]["text"] == "あ" * 60 + "…"
    s.avalon_response["turns"][1]["text"] = "あ" * 60
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    unknown = next(x for x in sm["speakers"] if x["id"] == "spk_1")
    assert unknown["examples"][0]["text"] == "あ" * 60
