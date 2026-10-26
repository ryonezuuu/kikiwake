from dataclasses import dataclass

import numpy as np

from kikiwake.audio import cut
from kikiwake.config import Settings
from kikiwake.embedder import Embedder
from kikiwake.models import DiarizedTranscription

Segment = tuple[float, float]  # (start, end) in seconds

_LENGTH_DECIMALS = 6  # segment lengths closer than a microsecond count as equal


@dataclass
class Voiceprint:
    segments: list[Segment]
    total_seconds: float
    centroid: np.ndarray | None  # None: not enough clean speech to make a voiceprint
    consistency: float | None  # mean cosine similarity of the chunks to the centroid


def _clean_runs(tr: DiarizedTranscription, speaker_id: str, s: Settings) -> list[Segment]:
    """Join the speaker's clean words into runs, breaking at any other word or a gap of ``max_word_gap``."""
    words = sorted(
        ((word, turn.speaker) for turn in tr.turns for word in turn.words),
        key=lambda pair: pair[0].start,
    )
    runs: list[Segment] = []
    open_run: Segment | None = None
    for word, speaker in words:
        clean = (
            speaker == speaker_id
            and not word.overlap
            and word.speaker_confidence >= s.min_word_confidence
        )
        if clean and open_run is not None and word.start - open_run[1] < s.max_word_gap:
            open_run = (open_run[0], max(open_run[1], word.end))
            continue
        if open_run is not None:
            runs.append(open_run)
            open_run = None
        if clean:
            open_run = (word.start, word.end)
    if open_run is not None:
        runs.append(open_run)
    return runs


def select_segments(tr: DiarizedTranscription, speaker_id: str, s: Settings) -> list[Segment]:
    """Pick up to ``max_total_per_speaker`` seconds of the speaker's clean speech, in time order."""
    trimmed = [(a + s.edge_trim, b - s.edge_trim) for a, b in _clean_runs(tr, speaker_id, s)]
    candidates = [seg for seg in trimmed if seg[1] - seg[0] >= s.min_segment]
    # Longest first; lengths are rounded so float noise cannot break a tie, which goes to the earlier one.
    candidates.sort(key=lambda seg: (-round(seg[1] - seg[0], _LENGTH_DECIMALS), seg[0]))

    chosen: list[Segment] = []
    total = 0.0
    for start, end in candidates:
        if total + (end - start) <= s.max_total_per_speaker:
            chosen.append((start, end))
            total += end - start
            continue
        # Fill what is left with the start of this segment and stop (no later one could fit anyway).
        remaining = s.max_total_per_speaker - total
        if remaining >= s.min_segment:
            chosen.append((start, start + remaining))
        break
    return sorted(chosen)


def split_chunks(segments: list[Segment], s: Settings) -> list[Segment]:
    """Cut each segment into ``chunk_seconds`` pieces; a remainder under ``min_segment`` joins the last piece."""
    chunks: list[Segment] = []
    for start, end in segments:
        full = int((end - start) // s.chunk_seconds)
        pieces = [
            (start + i * s.chunk_seconds, start + (i + 1) * s.chunk_seconds) for i in range(full)
        ]
        tail_start = start + full * s.chunk_seconds
        if end - tail_start >= s.min_segment or (not pieces and end > tail_start):
            pieces.append((tail_start, end))
        elif pieces:
            pieces[-1] = (pieces[-1][0], end)
        chunks.extend(pieces)
    return chunks


def build_voiceprint(
    wave: np.ndarray,
    tr: DiarizedTranscription,
    speaker_id: str,
    embedder: Embedder,
    s: Settings,
) -> Voiceprint:
    segments = select_segments(tr, speaker_id, s)
    total = sum((b - a for a, b in segments), 0.0)
    if not segments or total < s.min_total_per_speaker:
        return Voiceprint(segments, total, None, None)

    chunks = split_chunks(segments, s)
    embeddings = np.stack([embedder.embed(cut(wave, a, b)) for a, b in chunks])
    mean = embeddings.mean(axis=0)
    centroid = mean / np.linalg.norm(mean)
    consistency = float(np.mean(embeddings @ centroid))
    return Voiceprint(segments, total, centroid, consistency)
