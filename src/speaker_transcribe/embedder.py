from pathlib import Path
from typing import Protocol

import numpy as np

from speaker_transcribe.audio import SAMPLE_RATE
from speaker_transcribe.errors import AppError

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DIM = 192

_MIN_SAMPLES = SAMPLE_RATE // 2  # 0.5 s; shorter clips give meaningless (or NaN) embeddings


class Embedder(Protocol):
    """Turns a 16 kHz mono waveform into a fixed-size, L2-normalised speaker vector."""

    model_id: str
    dim: int

    def embed(self, wave: np.ndarray) -> np.ndarray: ...


class EcapaEmbedder:
    """SpeechBrain ECAPA-TDNN speaker embedder (192-d, L2-normalised), run on CPU.

    torch and speechbrain are imported here rather than at module level so that
    ``MODEL_ID`` / ``DIM`` / ``Embedder`` can be used without loading them.
    """

    model_id = MODEL_ID
    dim = DIM

    def __init__(self, savedir: Path) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=str(savedir),
                run_opts={"device": "cpu"},
            )
        except Exception as e:
            raise AppError("埋め込みモデルのダウンロードに失敗しました。ネットワーク接続を確認してください") from e

    def embed(self, wave: np.ndarray) -> np.ndarray:
        if len(wave) < _MIN_SAMPLES:
            raise AppError("声紋の計算に必要な音声が短すぎます")
        with self._torch.no_grad():
            out = self._classifier.encode_batch(self._torch.from_numpy(wave)[None])
        vec = out.reshape(-1).numpy().astype(np.float32)
        return vec / np.linalg.norm(vec)
