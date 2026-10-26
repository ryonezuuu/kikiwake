import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from speaker_transcribe.audio import to_wav16k
from speaker_transcribe.session import Session, SpeakerState


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


def say_wav(voice: str, text: str, path: Path) -> Path:
    """Synthesise ``text`` with macOS ``say`` and write it to ``path`` as 16 kHz mono WAV."""
    if shutil.which("say") is None:
        pytest.skip("macOS `say` is not available")
    aiff = path.with_suffix(".aiff")
    subprocess.run(
        ["say", "-v", voice, "-o", str(aiff), text],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    to_wav16k(aiff, path)
    return path


def unit(i: int, dim: int = 4) -> np.ndarray:
    """The ``i``-th basis vector: an L2-normalised embedding that is orthogonal to every other ``unit``."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[i] = 1.0
    return vec


def make_state(**overrides) -> SpeakerState:
    """A ``SpeakerState`` for ``spk_0`` (unknown, one 5 s segment, centroid ``unit(0)``); ``overrides`` replace fields."""
    fields = dict(
        id="spk_0",
        number=1,
        segments=[(0.0, 5.0)],
        total_seconds=5.0,
        centroid=unit(0).tolist(),
        consistency=1.0,
        scores={},
        status="unknown",
        name=None,
        score=None,
        db_sample_ids=[],
    )
    fields.update(overrides)
    return SpeakerState(**fields)


def make_session(**speaker_overrides: dict) -> Session:
    """A ``Session`` with three unknown speakers ``spk_0``..``spk_2`` (numbers 1..3, centroids ``unit(0..2)``).

    ``spk_1=dict(status="auto", name="田中")`` overrides fields of that speaker; ``centroid=None`` is allowed.
    """
    ids = ("spk_0", "spk_1", "spk_2")
    unexpected = set(speaker_overrides) - set(ids)
    if unexpected:
        raise TypeError(f"unknown speakers: {sorted(unexpected)}")
    speakers = {
        sid: make_state(
            **{
                "id": sid,
                "number": i + 1,
                "centroid": unit(i).tolist(),
                **speaker_overrides.get(sid, {}),
            }
        )
        for i, sid in enumerate(ids)
    }
    return Session(
        audio_path="meeting.m4a",
        audio_sha256="sha",
        created_at="2026-10-26T10:00:00+09:00",
        model_id="fake",
        avalon_response=raw_diarized(
            [
                ("spk_0", 0, 5, 0.9, False),
                ("spk_1", 5, 10, 0.9, False),
                ("spk_2", 10, 15, 0.9, False),
            ]
        ),
        speakers=speakers,
        warnings=[],
    )
