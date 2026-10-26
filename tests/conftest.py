from collections.abc import Sequence

import numpy as np


def raw_diarized(
    words: list[tuple[str, float, float, float, bool]],
    *,
    duration: float | None = None,
    warnings: Sequence[str] = (),
    language: str = "ja",
) -> dict:
    """Build an Avalon ``diarized_json`` payload from ``(speaker, start, end, confidence, overlap)`` words.

    Consecutive words by the same speaker are merged into one turn. Every word's
    text is ``"w"`` and a turn's text is the concatenation of its words' texts.
    """
    turns: list[dict] = []
    first_seen: dict[str, float] = {}
    for speaker, start, end, confidence, overlap in words:
        first_seen.setdefault(speaker, start)
        word = {
            "text": "w",
            "start": start,
            "end": end,
            "speaker_confidence": confidence,
            "overlap": overlap,
        }
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["words"].append(word)
            turns[-1]["end"] = end
            turns[-1]["text"] += word["text"]
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": word["text"],
                    "words": [word],
                }
            )
    if duration is None:
        duration = words[-1][2] if words else 0.0
    return {
        "task": "transcribe",
        "language": language,
        "duration": duration,
        "text": " ".join(t["text"] for t in turns),
        "speakers": [{"id": s, "first_seen": t} for s, t in first_seen.items()],
        "turns": turns,
        "warnings": list(warnings),
        "usage": {"type": "duration", "seconds": duration},
    }


class FakeEmbedder:
    """Deterministic stand-in for a speaker embedder: depends only on the mean of the waveform."""

    model_id = "fake"
    dim = 4

    def embed(self, wave: np.ndarray) -> np.ndarray:
        vec = np.array([wave.mean(), 1.0, 0.0, 0.0], dtype=np.float32)
        return vec / np.linalg.norm(vec)
