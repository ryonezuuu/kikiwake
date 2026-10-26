import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from speaker_transcribe.audio import (
    SAMPLE_RATE,
    cut,
    load_wav,
    probe_duration,
    require_ffmpeg,
    sha256_file,
    to_upload_mp3,
    to_wav16k,
)
from speaker_transcribe.errors import AppError, UsageError


def make_sine(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def ffprobe_stream(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {"channels": stream["channels"], "sample_rate": int(stream["sample_rate"])}


def write_pcm16(path: Path, *, channels: int, rate: int, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * channels * frames)


def test_roundtrip_wav_and_duration(tmp_path):
    src = tmp_path / "会議 10.26.m4a"
    make_sine(src, seconds=2)
    wav = tmp_path / "a.wav"
    to_wav16k(src, wav)
    w = load_wav(wav)
    assert w.dtype == np.float32 and abs(len(w) - 32000) < 400 and np.abs(w).max() <= 1.0
    assert probe_duration(src) == pytest.approx(2.0, abs=0.1)


def test_upload_mp3_is_mono_16k(tmp_path):
    src = tmp_path / "会議 10.26.m4a"
    make_sine(src, seconds=1)
    mp3 = tmp_path / "up load.mp3"
    to_upload_mp3(src, mp3)
    assert ffprobe_stream(mp3) == {"channels": 1, "sample_rate": 16000}


def make_m4a_with_cover_art(path: Path) -> None:
    cover = path.with_name("cover.png")
    run = lambda *args: subprocess.run(  # noqa: E731
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True, capture_output=True
    )
    run("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(cover))
    run(
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-i", str(cover),
        "-map", "0:a", "-map", "1:v", "-c:a", "aac", "-c:v", "mjpeg",
        "-disposition:v:0", "attached_pic", str(path),
    )  # fmt: skip


def stream_types(path: Path) -> list[str]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()


def test_conversions_drop_cover_art(tmp_path):
    src = tmp_path / "with cover.m4a"
    make_m4a_with_cover_art(src)
    assert sorted(stream_types(src)) == ["audio", "video"]
    to_upload_mp3(src, tmp_path / "up.mp3")
    to_wav16k(src, tmp_path / "a.wav")
    assert stream_types(tmp_path / "up.mp3") == ["audio"]
    assert stream_types(tmp_path / "a.wav") == ["audio"]


def test_cut_clamps():
    w = np.zeros(16000, dtype=np.float32)
    assert len(cut(w, 0.5, 2.0)) == 8000 and len(cut(w, -1.0, 0.25)) == 4000


def test_cut_empty_when_end_not_after_start():
    w = np.ones(16000, dtype=np.float32)
    for start, end in [(1.0, 0.5), (0.5, 0.5), (2.0, 3.0), (-2.0, -1.0)]:
        out = cut(w, start, end)
        assert len(out) == 0 and out.dtype == np.float32


def test_cut_uses_rounded_sample_positions():
    w = np.arange(SAMPLE_RATE, dtype=np.float32)
    out = cut(w, 0.25, 0.5)
    assert out[0] == 4000 and len(out) == 4000


def test_ffmpeg_failure_raises(tmp_path):
    bad = tmp_path / "x.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(AppError) as excinfo:
        to_wav16k(bad, tmp_path / "y.wav")
    assert "x.m4a" in str(excinfo.value)
    assert not isinstance(excinfo.value, UsageError)


def test_ffmpeg_missing_input_raises_app_error(tmp_path):
    with pytest.raises(AppError, match="nothing here.m4a"):
        to_upload_mp3(tmp_path / "nothing here.m4a", tmp_path / "y.mp3")


def test_probe_duration_failure_raises(tmp_path):
    bad = tmp_path / "x.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(AppError, match="x.m4a"):
        probe_duration(bad)
    with pytest.raises(AppError, match="gone.m4a"):
        probe_duration(tmp_path / "gone.m4a")


@pytest.mark.parametrize("output", [b"N/A\n", b"\n", b"nan\n"])
def test_probe_duration_non_numeric_output_raises(monkeypatch, tmp_path, output):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AppError, match="音声の長さ"):
        probe_duration(tmp_path / "a.m4a")


def test_load_wav_rejects_wrong_format(tmp_path):
    stereo = tmp_path / "stereo.wav"
    write_pcm16(stereo, channels=2, rate=SAMPLE_RATE)
    with pytest.raises(AppError, match="stereo.wav"):
        load_wav(stereo)

    wrong_rate = tmp_path / "rate.wav"
    write_pcm16(wrong_rate, channels=1, rate=44100)
    with pytest.raises(AppError, match="rate.wav"):
        load_wav(wrong_rate)

    not_wav = tmp_path / "bad.wav"
    not_wav.write_bytes(b"not a wav")
    with pytest.raises(AppError, match="bad.wav"):
        load_wav(not_wav)


def test_load_wav_scales_int16_to_unit_range(tmp_path):
    path = tmp_path / "ext.wav"
    samples = np.array([-32768, 0, 16384, 32767], dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())
    w = load_wav(path)
    assert w.dtype == np.float32
    assert w.tolist() == [-1.0, 0.0, 0.5, 32767 / 32768]


def test_require_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(UsageError, match="brew install ffmpeg"):
        require_ffmpeg()


def test_require_ffmpeg_present():
    require_ffmpeg()


def test_sha256(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_streams_multiple_blocks(tmp_path):
    p = tmp_path / "big"
    data = b"x" * (2 * 1024 * 1024 + 123)
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
