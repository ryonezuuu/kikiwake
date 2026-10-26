"""The meeting archive: ``ROOT/index.md``, a table of every ``ROOT/records/<stem>/<stem>.session.json``.

The index is what an AI reads to recall past meetings, so it lists each one with its date, length,
participants, the one-line takeaway from ``summary.md`` and a link to the transcript.
"""

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kikiwake.session import Session, load_session, session_paths
from kikiwake.speaker_db import atomic_write_text

SUMMARY_FILENAME = "summary.md"
_SUMMARY_LINE = re.compile(r"^\s*要点[:：](.*)$")
_SUMMARY_CHARS = 60
_NO_SUMMARY = "（要点なし）"
_NAMED = ("auto", "confirmed")
_HEADER = "# 会議の記録\n\n| 日時 | 長さ | 参加者 | 要点 | 文字起こし |\n|---|---|---|---|---|\n"
_EMPTY_ROW = "| （まだ記録がありません） | | | | |\n"
# Characters of a file name that would end the table cell or the <...> link; percent-encoded.
_LINK_ESCAPES = str.maketrans({"|": "%7C", "<": "%3C", ">": "%3E"})


@dataclass(frozen=True)
class MeetingRow:
    when: str  # YYYY-MM-DD HH:MM
    minutes: int
    participants: str
    summary: str
    link: str  # the transcript, relative to the archive root


def _table_safe(text: str) -> str:
    """``text`` with the column separator replaced, so that it cannot break the table."""
    return text.replace("|", "／")


def summary_line(summary_md: Path) -> str:
    """The text after the first ``要点:`` (or ``要点：``) line of ``summary_md``, at most 60 characters.

    Longer text is cut to 60 characters and ends in ``…``. Without such a line, or a readable file,
    ``（要点なし）``.
    """
    try:
        lines = summary_md.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return _NO_SUMMARY
    for line in lines:
        found = _SUMMARY_LINE.match(line)
        if found is None:
            continue
        body = found.group(1).strip()
        if not body:
            return _NO_SUMMARY
        if len(body) > _SUMMARY_CHARS:
            body = body[:_SUMMARY_CHARS] + "…"
        return _table_safe(body)
    return _NO_SUMMARY


def participants_text(session: Session) -> str:
    """Who was there: named speakers (``auto`` / ``confirmed``), then a count of the others.

    ``田中・佐藤・ほか1名``; with nobody named ``3名（名前未確定）``. A session without speakers is
    ``（処理未完了）`` if it has a transcript (the voiceprint step never ran), else ``0名``.
    """
    if not session.speakers:
        return "（処理未完了）" if session.transcription.turns else "0名"
    named = [st for st in session.speakers.values() if st.status in _NAMED]
    names = list(dict.fromkeys(st.display_label for st in named))
    others = len(session.speakers) - len(named)
    if not names:
        return f"{others}名（名前未確定）"
    text = "・".join(names)
    if others:
        text += f"・ほか{others}名"
    return _table_safe(text)


def _meeting_time(session: Session) -> datetime:
    """``recorded_at``, or ``created_at`` where it is absent (older sessions) or not a time (hand-edited).

    Both are written with a UTC offset. ``ValueError`` if ``created_at`` is not a time either.
    """
    if session.recorded_at:
        try:
            return datetime.fromisoformat(session.recorded_at)
        except ValueError:
            pass
    return datetime.fromisoformat(session.created_at)


def _meeting(session_file: Path, root: Path) -> tuple[datetime, MeetingRow]:
    """One record's sort time and table row; raises if the record cannot be read."""
    session = load_session(session_file)
    minutes = round(session.transcription.duration / 60)
    when = _meeting_time(session)
    markdown_path = session_paths(Path(session.audio_path), session_file.parent)[0]
    row = MeetingRow(
        when=when.strftime("%Y-%m-%d %H:%M"),  # as written, in its own UTC offset
        minutes=minutes,
        participants=participants_text(session),
        summary=summary_line(session_file.parent / SUMMARY_FILENAME),
        link=markdown_path.relative_to(root).as_posix().translate(_LINK_ESCAPES),
    )
    # A time without an offset (hand-edited) counts as UTC for ordering only, so sorting never fails.
    return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)), row


def _format_row(row: MeetingRow) -> str:
    return (
        f"| {row.when} | {row.minutes} 分 | {row.participants} | {row.summary}"
        f" | [開く](<{row.link}>) |\n"
    )


def build_index(root: Path) -> tuple[str, int, list[str]]:
    """The text of ``root/index.md``, the number of meetings in it, and the records skipped.

    Meetings are newest first (ties by link). A record that cannot be read, for whatever reason, is
    skipped and listed by its path relative to ``root``: one bad record never stops the index.
    """
    found: list[tuple[datetime, MeetingRow]] = []
    skipped: list[str] = []
    for session_file in (root / "records").glob("*/*.session.json"):
        try:
            found.append(_meeting(session_file, root))
        except Exception:
            skipped.append(session_file.relative_to(root).as_posix())

    found.sort(key=lambda pair: pair[1].link)
    found.sort(key=lambda pair: pair[0], reverse=True)  # stable: equal times stay ordered by link
    body = "".join(_format_row(row) for _, row in found) or _EMPTY_ROW
    return _HEADER + body, len(found), sorted(skipped)


def write_index(root: Path) -> tuple[Path, int, list[str]]:
    """Write ``root/index.md``; return its path, the number of meetings and the records skipped.

    Each skipped record is also reported on stderr.
    """
    text, meetings, skipped = build_index(root)
    index_path = root / "index.md"
    atomic_write_text(index_path, text)
    for rel in skipped:
        print(f"警告: 読めない記録を飛ばしました: {rel}", file=sys.stderr)
    return index_path, meetings, skipped
