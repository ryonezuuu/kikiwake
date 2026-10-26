import tempfile
from pathlib import Path

import pytest
from conftest import session_from_words

from kikiwake import archive
from kikiwake.archive import build_index, participants_text, summary_line, write_index
from kikiwake.session import Session, save_session

HEADER = "# 会議の記録\n\n| 日時 | 長さ | 参加者 | 要点 | 文字起こし |\n|---|---|---|---|---|\n"
NO_SUMMARY = "（要点なし）"
THREE_SPEAKERS = [
    ("spk_0", 0, 5, 0.9, False),
    ("spk_1", 5, 10, 0.9, False),
    ("spk_2", 10, 15, 0.9, False),
]


@pytest.fixture(autouse=True)
def _temp_file_mode(monkeypatch):
    """Start every test in the default (temp file + replace) mode; tests opt in to in-place."""
    monkeypatch.delenv("KIKIWAKE_INPLACE_WRITES", raising=False)


def meeting(
    stem: str,
    recorded_at: str | None,
    *,
    created_at: str = "2026-10-26T12:00:00+09:00",
    duration: float = 1800.0,
    states: dict[str, dict] | None = None,
) -> Session:
    session = session_from_words(
        THREE_SPEAKERS, duration=duration, states=states, audio_path=f"/audio/{stem}.m4a"
    )
    session.recorded_at = recorded_at
    session.created_at = created_at
    return session


def put(root: Path, stem: str, session: Session) -> Path:
    """Save ``session`` as ``root/records/<stem>/<stem>.session.json``; return the folder."""
    folder = root / "records" / stem
    save_session(session, folder / f"{stem}.session.json")
    return folder


def rows(text: str) -> list[str]:
    """The table rows (everything after the 4-line header), in order."""
    return text.splitlines()[4:]


def whens(text: str) -> list[str]:
    """The first column of every table row, in order."""
    return [row.split(" | ")[0].removeprefix("| ") for row in rows(text)]


def test_index_golden(tmp_path):
    voice_memo = "20261026 102027-FA0D0F9E"
    named = {
        "spk_0": dict(status="auto", name="田中"),
        "spk_1": dict(status="confirmed", name="佐藤"),
    }
    folder = put(
        tmp_path,
        voice_memo,
        meeting(voice_memo, "2026-10-26T10:20:27+09:00", duration=2460.0, states=named),
    )
    (folder / "summary.md").write_text(
        "# 時刻同期の議論（2026-10-26 10:20）\n\n要点: 時刻同期の前提を見直す\n\n## 決まったこと\n- なし\n",
        encoding="utf-8",
    )
    put(tmp_path, "weekly", meeting("weekly", "2026-10-20T09:00:00+09:00"))  # older, no summary.md

    text, count, skipped = build_index(tmp_path)

    assert text == (
        HEADER
        + "| 2026-10-26 10:20 | 41 分 | 田中・佐藤・ほか1名 | 時刻同期の前提を見直す"
        + " | [開く](<records/20261026 102027-FA0D0F9E/20261026 102027-FA0D0F9E.md>) |\n"
        + f"| 2026-10-20 09:00 | 30 分 | 3名（名前未確定） | {NO_SUMMARY}"
        + " | [開く](<records/weekly/weekly.md>) |\n"
    )
    assert (count, skipped) == (2, [])


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("要点: 時刻同期を見直す\n", "時刻同期を見直す"),
        ("要点：時刻同期を見直す\n", "時刻同期を見直す"),  # full-width colon
        ("   要点:\t 前に空白があり、後ろにも  \n", "前に空白があり、後ろにも"),
        ("# 題\n\n要点: 最初\n要点: 二つ目\n", "最初"),  # the first such line
        ("# 題\r\n要点: CRLF\r\n", "CRLF"),
        ("要点: " + "あ" * 60 + "\n", "あ" * 60),  # exactly 60 characters stays whole
        ("要点: " + "あ" * 61 + "\n", "あ" * 60 + "…"),
        ("要点: 速度|精度\n", "速度／精度"),  # would break the table
        ("# 題\n\n## 決まったこと\n- 要点: 箇条書きの中\n", NO_SUMMARY),  # not at the start of a line
        ("要点\n要点の整理: x\n", NO_SUMMARY),  # no colon right after 要点
        ("要点:   \n", NO_SUMMARY),  # nothing after the colon
        ("", NO_SUMMARY),
    ],
)
def test_summary_line_variants(tmp_path, content, expected):
    path = tmp_path / "summary.md"
    path.write_text(content, encoding="utf-8")
    assert summary_line(path) == expected


def test_summary_line_without_a_readable_file(tmp_path):
    assert summary_line(tmp_path / "summary.md") == NO_SUMMARY
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe\x00 not utf-8")
    assert summary_line(bad) == NO_SUMMARY
    assert summary_line(tmp_path) == NO_SUMMARY  # a directory


def test_participants_variants():
    def text(**states: dict) -> str:
        return participants_text(session_from_words(THREE_SPEAKERS, states=states))

    named = dict(status="auto", name="田中")
    assert text(spk_0=named, spk_1=dict(status="confirmed", name="佐藤")) == "田中・佐藤・ほか1名"
    everyone = dict(spk_0=named, spk_1=dict(status="confirmed", name="佐藤"), spk_2=dict(status="auto", name="鈴木"))
    assert text(**everyone) == "田中・佐藤・鈴木"
    assert text(spk_2=named) == "田中・ほか2名"
    # The same person on two speakers is listed once; the rest are counted.
    assert text(spk_0=named, spk_1=dict(status="confirmed", name="田中")) == "田中・ほか1名"
    # Guesses and unmatched voices are not names.
    assert text(spk_0=dict(status="candidate", name="田中"), spk_1=dict(status="insufficient")) == (
        "3名（名前未確定）"
    )
    assert text() == "3名（名前未確定）"
    assert text(spk_0=dict(status="auto", name="A|B")) == "A／B・ほか2名"  # would break the table


def test_participants_without_speakers():
    unfinished = session_from_words(THREE_SPEAKERS)  # the voiceprint step never ran
    unfinished.speakers = {}
    assert participants_text(unfinished) == "（処理未完了）"

    silent = session_from_words([])
    assert silent.speakers == {}
    assert participants_text(silent) == "0名"


def test_old_session_without_recorded_at_sorts_by_created_at(tmp_path):
    put(tmp_path, "recorded", meeting("recorded", "2026-10-20T09:00:00+09:00"))
    put(tmp_path, "old", meeting("old", None, created_at="2026-10-15T12:34:56+09:00"))
    put(tmp_path, "newest", meeting("newest", None, created_at="2026-10-26T08:00:00+09:00"))
    # created_at is later than all of these, but recorded_at is what counts when present.
    late_upload = meeting("earliest", "2026-10-01T07:30:00+09:00", created_at="2026-10-27T00:00:00+09:00")
    put(tmp_path, "earliest", late_upload)

    text, count, _ = build_index(tmp_path)

    assert count == 4
    assert whens(text) == [
        "2026-10-26 08:00",
        "2026-10-20 09:00",
        "2026-10-15 12:34",
        "2026-10-01 07:30",
    ]
    assert "(<records/old/old.md>)" in rows(text)[2]


def test_sort_follows_the_instant_and_shows_the_time_as_written(tmp_path):
    # 2026-10-26 00:00 UTC, written in +09:00 ...
    put(tmp_path, "tokyo", meeting("tokyo", "2026-10-26T09:00:00+09:00"))
    # ... and 01:00 UTC, written in -05:00: later, though it sorts earlier as text.
    put(tmp_path, "chicago", meeting("chicago", "2026-10-25T20:00:00-05:00"))

    text, _, _ = build_index(tmp_path)

    assert whens(text) == ["2026-10-25 20:00", "2026-10-26 09:00"]


def test_equal_times_are_ordered_by_link(tmp_path):
    for stem in ("b", "c", "a"):
        put(tmp_path, stem, meeting(stem, "2026-10-26T09:00:00+09:00"))

    text, _, _ = build_index(tmp_path)

    links = [row.split("(<")[1].split(">)")[0] for row in rows(text)]
    assert links == ["records/a/a.md", "records/b/b.md", "records/c/c.md"]


def test_unreadable_session_is_skipped_with_warning(tmp_path, capsys):
    put(tmp_path, "good", meeting("good", "2026-10-26T09:00:00+09:00"))
    bad_json = tmp_path / "records" / "broken json" / "broken json.session.json"
    bad_json.parent.mkdir(parents=True)
    bad_json.write_text("{not json", encoding="utf-8")
    no_response = meeting("no response", "2026-10-26T09:00:00+09:00")
    no_response.avalon_response = {}  # loads, but is not a transcription
    put(tmp_path, "no response", no_response)
    put(tmp_path, "bad times", meeting("bad times", "last Tuesday", created_at="yesterday"))
    put(tmp_path, "endless", meeting("endless", "2026-10-26T09:00:00+09:00", duration=float("inf")))
    (tmp_path / "records" / "not a file" / "x.session.json").mkdir(parents=True)  # a directory
    (tmp_path / "records" / "stray.session.json").write_text("{}", encoding="utf-8")  # not in a folder: ignored

    path, count, skipped = write_index(tmp_path)

    assert path == tmp_path / "index.md"
    assert count == 1
    assert skipped == [
        "records/bad times/bad times.session.json",
        "records/broken json/broken json.session.json",
        "records/endless/endless.session.json",  # OverflowError: any error skips only that record
        "records/no response/no response.session.json",
        "records/not a file/x.session.json",
    ]
    text = path.read_text(encoding="utf-8")
    assert len(rows(text)) == 1 and "records/good/good.md" in text
    err = capsys.readouterr().err
    assert err.splitlines() == [f"警告: 読めない記録を飛ばしました: {rel}" for rel in skipped]


def test_unexpected_error_in_one_record_does_not_stop_the_index(tmp_path, monkeypatch):
    put(tmp_path, "good", meeting("good", "2026-10-26T09:00:00+09:00"))
    put(tmp_path, "odd", meeting("odd", "2026-10-26T09:00:00+09:00"))
    real = archive.participants_text

    def flaky(session):
        if session.audio_path.endswith("odd.m4a"):
            raise RuntimeError("unexpected")
        return real(session)

    monkeypatch.setattr(archive, "participants_text", flaky)
    _, count, skipped = build_index(tmp_path)
    assert (count, skipped) == (1, ["records/odd/odd.session.json"])


@pytest.mark.parametrize("recorded_at", ["last Tuesday", ""])
def test_recorded_at_that_is_not_a_time_falls_back_to_created_at(tmp_path, recorded_at):
    put(tmp_path, "m", meeting("m", recorded_at, created_at="2026-10-15T12:34:56+09:00"))
    put(tmp_path, "later", meeting("later", "2026-10-20T09:00:00+09:00"))
    text, count, skipped = build_index(tmp_path)
    assert (count, skipped) == (2, [])
    assert whens(text) == ["2026-10-20 09:00", "2026-10-15 12:34"]


@pytest.mark.parametrize(
    ("stem", "target"),
    [
        ("速度|精度", "records/速度%7C精度/速度%7C精度.md"),
        ("a<b>c", "records/a%3Cb%3Ec/a%3Cb%3Ec.md"),
    ],
)
def test_stem_with_table_or_link_characters_does_not_break_the_row(tmp_path, stem, target):
    put(tmp_path, stem, meeting(stem, "2026-10-26T09:00:00+09:00"))
    text, _, _ = build_index(tmp_path)
    (row,) = rows(text)
    assert row.endswith(f" | [開く](<{target}>) |")
    assert row.count("|") == 6  # five cells


@pytest.mark.parametrize("make_records_dir", [False, True])
def test_empty_root_says_no_records(tmp_path, make_records_dir):
    if make_records_dir:
        (tmp_path / "records").mkdir()
        (tmp_path / "records" / "no-session").mkdir()
        (tmp_path / "records" / "no-session" / "summary.md").write_text("要点: x\n", encoding="utf-8")

    path, count, skipped = write_index(tmp_path)

    assert (count, skipped) == (0, [])
    assert path.read_text(encoding="utf-8") == HEADER + "| （まだ記録がありません） | | | | |\n"


def test_index_uses_inplace_writes_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKIWAKE_INPLACE_WRITES", "1")

    def no_temp(*args, **kwargs):
        pytest.fail("a temp file was created")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", no_temp)
    put(tmp_path, "m", meeting("m", "2026-10-26T09:00:00+09:00"))
    (tmp_path / "index.md").write_text("old", encoding="utf-8")

    path, count, _ = write_index(tmp_path)

    assert count == 1
    assert "records/m/m.md" in path.read_text(encoding="utf-8")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.md", "records"]
