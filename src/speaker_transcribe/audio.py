import hashlib
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from speaker_transcribe.errors import AppError, UsageError

SAMPLE_RATE = 16000

_STDERR_TAIL_CHARS = 500
_HASH_BLOCK_BYTES = 1024 * 1024


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise UsageError(
            f"{' / '.join(missing)} が見つかりません。`brew install ffmpeg` などでインストールしてください"
        )


def _run(cmd: list[str], src: Path, what: str) -> subprocess.CompletedProcess[bytes]:
    """Run ``cmd`` without a shell; any failure becomes an ``AppError`` naming ``src``."""
    try:
        # stdin is closed so ffmpeg can never consume the caller's stdin.
        proc = subprocess.run(cmd, check=False, capture_output=True, stdin=subprocess.DEVNULL)
    except OSError as e:
        raise AppError(f"{cmd[0]} を実行できませんでした: {e}") from e
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace").strip()[-_STDERR_TAIL_CHARS:]
        raise AppError(f"{what}に失敗しました: {src.name}\n{tail}")
    return proc


def probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]  # fmt: skip
    proc = _run(cmd, path, "ffprobe による音声長の取得")
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    try:
        duration = float(text)
    except ValueError:
        duration = math.nan
    if not math.isfinite(duration):
        raise AppError(f"音声の長さを取得できませんでした: {path.name}（ffprobe の出力: {text!r}）")
    return duration


def _convert(src: Path, dst: Path, codec_args: list[str]) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        *codec_args,
        str(dst),
    ]  # fmt: skip
    _run(cmd, src, "ffmpeg による音声変換")


def to_upload_mp3(src: Path, dst: Path) -> None:
    _convert(src, dst, ["-c:a", "libmp3lame", "-b:a", "64k"])


def to_wav16k(src: Path, dst: Path) -> None:
    _convert(src, dst, ["-c:a", "pcm_s16le"])


def load_wav(path: Path) -> np.ndarray:
    """Load a 16-bit PCM mono 16 kHz WAV (as made by ``to_wav16k``) as float32 in [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as wf:
            channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if (channels, width, rate) != (1, 2, SAMPLE_RATE):
                raise AppError(
                    f"WAV の形式が想定外です: {path.name}"
                    f"（{channels} ch, {width * 8} bit, {rate} Hz。"
                    f"16 bit PCM / モノラル / {SAMPLE_RATE} Hz が必要です）"
                )
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise AppError(f"WAV を読み込めませんでした: {path.name}（{e}）") from e
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def cut(wave: np.ndarray, start: float, end: float) -> np.ndarray:
    """Return ``wave[start:end]`` (seconds), clamped to the signal; empty if ``end <= start``."""
    first = min(max(round(start * SAMPLE_RATE), 0), len(wave))
    last = min(max(round(end * SAMPLE_RATE), 0), len(wave))
    return wave[first:last] if last > first else wave[:0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()
