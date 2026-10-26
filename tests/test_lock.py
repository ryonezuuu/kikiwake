"""Static checks on ``uv.lock`` (no network): Linux must resolve to CPU-only PyTorch."""

import tomllib
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"
CPU_INDEX = "download.pytorch.org/whl/cpu"


def lock() -> dict:
    return tomllib.loads(LOCK.read_text(encoding="utf-8"))


def test_linux_torch_comes_from_the_cpu_index():
    pkgs = [p for p in lock()["package"] if p["name"] in ("torch", "torchaudio")]
    linux = [p for p in pkgs if CPU_INDEX in p["source"].get("registry", "")]
    assert {p["name"] for p in linux} == {"torch", "torchaudio"}
    assert all("+cpu" in p["version"] for p in linux)


def test_no_cuda_packages_in_the_lock():
    assert not [p["name"] for p in lock()["package"] if p["name"].startswith("nvidia-")]


def test_macos_torch_still_from_pypi():
    assert any(
        p["name"] == "torch" and "pypi.org" in p["source"].get("registry", "")
        for p in lock()["package"]
    )
