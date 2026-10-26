from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Turns a 16 kHz mono waveform into a fixed-size, L2-normalised speaker vector."""

    model_id: str
    dim: int

    def embed(self, wave: np.ndarray) -> np.ndarray: ...
