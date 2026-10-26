import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import say_wav

import kikiwake
from kikiwake.audio import SAMPLE_RATE, load_wav
from kikiwake.embedder import DIM, MODEL_ID, EcapaEmbedder
from kikiwake.errors import AppError


def test_importing_embedder_does_not_load_torch_or_speechbrain():
    code = (
        "import sys; import kikiwake.embedder; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'speechbrain' not in sys.modules, 'speechbrain was imported'"
    )
    src_dir = Path(kikiwake.__file__).parent.parent
    env = {**os.environ, "PYTHONPATH": str(src_dir)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_constants():
    assert MODEL_ID == "speechbrain/spkrec-ecapa-voxceleb"
    assert DIM == 192
    assert EcapaEmbedder.model_id == MODEL_ID
    assert EcapaEmbedder.dim == DIM


def test_model_load_failure_names_the_cause(monkeypatch, tmp_path):
    from speechbrain.inference.speaker import EncoderClassifier

    def offline(*args, **kwargs):
        raise ConnectionError("Name or service not known")

    monkeypatch.setattr(EncoderClassifier, "from_hparams", offline)
    with pytest.raises(AppError) as excinfo:
        EcapaEmbedder(tmp_path)
    assert str(excinfo.value) == (
        "埋め込みモデルのダウンロードに失敗しました。ネットワーク接続を確認してください"
        "（ConnectionError: Name or service not known）"
    )


@pytest.fixture(scope="module")
def savedir(tmp_path_factory):
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="module")
def embedder(savedir):
    return EcapaEmbedder(savedir)


@pytest.fixture
def emb(embedder, tmp_path):
    counter = iter(range(1000))

    def _emb(voice: str, text: str) -> np.ndarray:
        wav = say_wav(voice, text, tmp_path / f"{voice}-{next(counter)}.wav")
        return embedder.embed(load_wav(wav))

    return _emb


@pytest.mark.model
def test_model_files_live_in_savedir(embedder, savedir):
    for name in ("embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.ckpt"):
        assert (savedir / name).is_file(), name
    assert (savedir / "embedding_model.ckpt").stat().st_size > 1_000_000
    assert not [p for p in savedir.rglob("*") if p.is_symlink()]


@pytest.mark.model
def test_shape_and_norm(tmp_path, embedder):
    v = embedder.embed(load_wav(say_wav("Kyoko", "本日の議題は予算についてです。", tmp_path / "a.wav")))
    assert v.shape == (192,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.model
def test_same_voice_closer_than_different(emb):
    k1 = emb("Kyoko", "本日の議題は予算についてです。")
    k2 = emb("Kyoko", "来週までに資料をまとめておきます。")
    g = emb("Grandpa", "来週までに資料をまとめておきます。")
    assert k1 @ k2 > k1 @ g


@pytest.mark.model
def test_too_short_wave_is_rejected(embedder):
    with pytest.raises(AppError, match="短すぎ"):
        embedder.embed(np.zeros(SAMPLE_RATE // 2 - 1, dtype=np.float32))
    with pytest.raises(AppError, match="短すぎ"):
        embedder.embed(np.zeros(0, dtype=np.float32))
