"""Render a session as Markdown (``<stem>.md``) and as the JSON summary an AI agent reads."""

from dataclasses import dataclass
from pathlib import Path

from speaker_transcribe.models import Turn, Word
from speaker_transcribe.session import Session

_UNCONFIRMED = ("candidate", "unknown", "insufficient")
_MAX_EXAMPLES = 3
_EXAMPLE_CHARS = 60


def format_ts(seconds: float) -> str:
    """``HH:MM:SS``, truncating fractions of a second."""
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def warning_messages(session: Session) -> list[str]:
    """User-facing notes, grouped by kind in a fixed order (unconfirmed, limit, consistency, discarded)."""
    messages: list[str] = []
    unconfirmed = sum(st.status in _UNCONFIRMED for st in session.speakers.values())
    if unconfirmed:
        messages.append(f"名前が未確定の話者が {unconfirmed} 人います。")
    codes = [w["code"] for w in session.warnings]
    if "speaker_limit_reached" in codes:
        messages.append("9 人以上いた可能性があり、一部の発言は別の話者に割り当てられています。")
    for w in session.warnings:
        if w["code"] == "low_consistency":
            st = session.speakers.get(w["speaker"])
            label = w["speaker"] if st is None else st.display_label
            messages.append(f"{label} の声に複数人が混ざっている可能性があります。")
    if "labels_discarded" in codes:
        messages.append("API を呼び直したため、前回の名前付けは破棄されました。")
    return messages


@dataclass
class _Paragraph:
    start: float
    label: str
    texts: list[str]
    words: list[Word]

    @property
    def mostly_overlapping(self) -> bool:
        return 2 * sum(w.overlap for w in self.words) > len(self.words)


def _paragraphs(session: Session, turns: tuple[Turn, ...]) -> list[_Paragraph]:
    """Non-blank turns in order, consecutive ones with the same display label merged."""
    paragraphs: list[_Paragraph] = []
    for turn in turns:
        text = turn.text.strip()
        if not text:
            continue
        st = session.speakers.get(turn.speaker)
        label = turn.speaker if st is None else st.display_label
        if paragraphs and paragraphs[-1].label == label:
            paragraphs[-1].texts.append(text)
            paragraphs[-1].words.extend(turn.words)
        else:
            paragraphs.append(_Paragraph(turn.start, label, [text], list(turn.words)))
    return paragraphs


def render_markdown(session: Session) -> str:
    tr = session.transcription
    joiner = "" if tr.language.startswith("ja") else " "
    labels = "・".join(dict.fromkeys(st.display_label for st in session.speakers.values())) or "なし"
    blocks = [
        f"# {Path(session.audio_path).name} の文字起こし",
        f"- 録音: {round(tr.duration / 60)} 分 / 話者: {labels}",
    ]
    warnings = warning_messages(session)
    if warnings:
        blocks.append("\n".join(f"> ⚠ {w}" for w in warnings))
    for p in _paragraphs(session, tr.turns):
        body = joiner.join(p.texts) + (" ※同時発話あり" if p.mostly_overlapping else "")
        blocks.append(f"**[{format_ts(p.start)}] {p.label}**\n{body}")
    return "\n\n".join(blocks) + "\n"


def _examples(turns: tuple[Turn, ...], speaker_id: str) -> list[dict[str, str]]:
    """The speaker's longest turns (earlier first on ties), cut to 60 characters."""
    spoken = [(t, t.text.strip()) for t in turns if t.speaker == speaker_id]
    spoken = [(t, text) for t, text in spoken if text]
    spoken.sort(key=lambda pair: (-len(pair[1]), pair[0].start))
    return [
        {
            "time": format_ts(t.start),
            "text": text[:_EXAMPLE_CHARS] + ("…" if len(text) > _EXAMPLE_CHARS else ""),
        }
        for t, text in spoken[:_MAX_EXAMPLES]
    ]


def build_summary(
    session: Session, markdown_path: Path, session_path: Path, viz_path: Path | None
) -> dict:
    """The ``--json`` payload for an AI agent: output paths, each speaker, and the warnings."""
    turns = session.transcription.turns
    speakers = []
    for sid, st in session.speakers.items():
        entry: dict = {"id": sid, "label": st.display_label, "status": st.status, "score": st.score}
        if st.status not in ("auto", "confirmed"):
            entry["examples"] = _examples(turns, sid)
        speakers.append(entry)
    return {
        "markdown_path": str(markdown_path),
        "session_path": str(session_path),
        "viz_path": None if viz_path is None else str(viz_path),
        "speakers": speakers,
        "warnings": warning_messages(session),
    }
