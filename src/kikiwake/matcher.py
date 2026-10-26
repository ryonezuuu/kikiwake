from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from kikiwake.config import Settings

# Stands in for a (speaker, person) pair with no score; far below any cosine, so it is never chosen over a real one.
_MISSING = -1e9


@dataclass(frozen=True)
class Match:
    name: str | None
    score: float | None
    status: Literal["auto", "candidate", "unknown"]


_UNKNOWN = Match(None, None, "unknown")


def person_score(centroid: np.ndarray, embeddings: list[np.ndarray], top_k: int) -> float:
    """Mean of the ``top_k`` highest cosine similarities between the centroid and the person's embeddings."""
    stacked = np.stack(embeddings).astype(np.float64)
    c = centroid.astype(np.float64)
    sims = stacked @ c / (np.linalg.norm(stacked, axis=1) * np.linalg.norm(c))
    return float(np.sort(sims)[::-1][:top_k].mean())


def score_table(
    centroids: dict[str, np.ndarray],
    people: dict[str, list[np.ndarray]],
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Score every speaker centroid against every person who has voiceprints: ``{speaker_id: {name: score}}``."""
    return {
        speaker_id: {
            name: person_score(centroid, embeddings, top_k)
            for name, embeddings in people.items()
            if embeddings
        }
        for speaker_id, centroid in centroids.items()
    }


def assign(scores: dict[str, dict[str, float]], s: Settings) -> dict[str, Match]:
    """Pair speakers with people one-to-one for the best total score, then grade each pair by threshold."""
    names = sorted({name for row in scores.values() for name in row})
    if not names:
        return {speaker_id: _UNKNOWN for speaker_id in scores}

    matrix = np.array(
        [[row.get(name, _MISSING) for name in names] for row in scores.values()], dtype=np.float64
    )
    rows, cols = linear_sum_assignment(matrix, maximize=True)
    picked = dict(zip(rows.tolist(), cols.tolist(), strict=True))

    result: dict[str, Match] = {}
    for i, speaker_id in enumerate(scores):
        j = picked.get(i)
        if j is None or matrix[i, j] == _MISSING or matrix[i, j] < s.candidate_threshold:
            result[speaker_id] = _UNKNOWN
            continue
        score = float(matrix[i, j])
        status = "auto" if score >= s.auto_threshold else "candidate"
        result[speaker_id] = Match(names[j], score, status)
    return result
