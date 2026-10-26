"""Voice visualisation (``<stem>.voices.png``): pitch, long-term average spectrum and a voiceprint map."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from speaker_transcribe.audio import SAMPLE_RATE, cut
from speaker_transcribe.session import Session, SpeakerState

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Segment = tuple[float, float]  # (start, end) in seconds

_JAPANESE_FONTS = ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP")
_NO_DATA = "データ不足"

_F0_MIN, _F0_MAX = 60, 500  # Hz
_F0_BINS = np.arange(_F0_MIN, _F0_MAX + 5, 10)
_PYIN_FRAME = 1024  # samples; librosa's default (2048) can warn at 16 kHz
_MIN_PITCH_SAMPLES = 2 * _PYIN_FRAME  # pyin needs a few frames to track a pitch

_N_FFT = 1024
_HOP = 512
_POWER_FLOOR = 1e-12  # keeps log10 finite for silence
_RANK_TOLERANCE = 1e-9  # a principal component this much weaker than the first carries only rounding noise

# Categorical colours in fixed order: the i-th speaker is drawn in the i-th colour in every panel.
_SPEAKER_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)  # fmt: skip
_PERSON_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
)  # fmt: skip


def speaker_f0(wave: np.ndarray, segments: Sequence[Segment]) -> np.ndarray:
    """Voiced fundamental frequencies (Hz) found in ``segments`` of ``wave``; empty if there are none."""
    import librosa

    found = [np.empty(0)]
    for start, end in segments:
        chunk = cut(wave, start, end)
        if len(chunk) < _MIN_PITCH_SAMPLES:
            continue
        f0, voiced, _ = librosa.pyin(
            chunk, fmin=_F0_MIN, fmax=_F0_MAX, sr=SAMPLE_RATE, frame_length=_PYIN_FRAME
        )
        found.append(f0[voiced & np.isfinite(f0)])
    return np.concatenate(found)


def _average_spectrum_db(wave: np.ndarray, segments: Sequence[Segment]) -> np.ndarray | None:
    """Time-averaged power (dB) per FFT bin over the joined ``segments``; ``None`` if they are too short."""
    import librosa

    chunks = [c for c in (cut(wave, start, end) for start, end in segments) if len(c) >= _N_FFT]
    if not chunks:
        return None
    power = np.abs(librosa.stft(np.concatenate(chunks), n_fft=_N_FFT, hop_length=_HOP)) ** 2
    return 10 * np.log10(power.mean(axis=1) + _POWER_FLOOR)


def project_2d(points: np.ndarray) -> np.ndarray:
    """Project the rows of ``points`` onto their top two principal components.

    A component without real variance (e.g. the second one of two points, or any of identical points) is zero.
    """
    centered = points - points.mean(axis=0)
    _, strength, components = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:2].T
    coords[:, strength[:2] <= strength[0] * _RANK_TOLERANCE] = 0.0
    return coords


def _font_family() -> list[str]:
    """The first installed Japanese font, then the generic fallback.

    Listing a family that is not installed, or one with no regular-weight face, makes matplotlib log a warning,
    so only one installed family (preferably with a regular face) is named.
    """
    from matplotlib import font_manager

    faces = font_manager.fontManager.ttflist
    installed = {font.name for font in faces}
    regular = {font.name for font in faces if font.weight == 400}
    usable = [name for name in _JAPANESE_FONTS if name in installed]
    usable.sort(key=lambda name: name not in regular)  # stable: keeps the preference order otherwise
    return usable[:1] + ["sans-serif"]


def _show_no_data(ax: "Axes") -> None:
    ax.text(0.5, 0.5, _NO_DATA, transform=ax.transAxes, ha="center", va="center", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_pitch(ax: "Axes", speakers: list[SpeakerState], colors: list[str], wave: np.ndarray) -> None:
    ax.set_title("ピッチ分布")
    drawn = False
    for st, color in zip(speakers, colors, strict=True):
        f0 = speaker_f0(wave, st.segments)
        if f0.size:
            ax.hist(f0, bins=_F0_BINS, histtype="step", linewidth=1.8, color=color, label=st.display_label)
            drawn = True
    if not drawn:
        _show_no_data(ax)
        return
    ax.set_xlabel("基本周波数 (Hz)")
    ax.set_ylabel("フレーム数")
    ax.legend()


def _draw_spectrum(ax: "Axes", speakers: list[SpeakerState], colors: list[str], wave: np.ndarray) -> None:
    ax.set_title("長時間平均スペクトル")
    freqs = np.fft.rfftfreq(_N_FFT, 1 / SAMPLE_RATE)
    drawn = False
    for st, color in zip(speakers, colors, strict=True):
        spectrum = _average_spectrum_db(wave, st.segments)
        if spectrum is not None:
            ax.plot(freqs, spectrum, linewidth=1.5, color=color, label=st.display_label)
            drawn = True
    if not drawn:
        _show_no_data(ax)
        return
    ax.set_xlim(0, SAMPLE_RATE / 2)
    ax.set_xlabel("周波数 (Hz)")
    ax.set_ylabel("パワー (dB)")
    ax.legend()


def _draw_voiceprints(
    ax: "Axes",
    speakers: list[SpeakerState],
    colors: list[str],
    people: dict[str, list[np.ndarray]],
) -> None:
    ax.set_title("声紋マップ")
    centroids = [np.asarray(st.centroid, dtype=np.float64) for st in speakers]
    samples = [np.asarray(sample, dtype=np.float64) for per_person in people.values() for sample in per_person]
    if not speakers or len(centroids) + len(samples) < 2:
        _show_no_data(ax)
        return

    coords = project_2d(np.stack(centroids + samples))
    speaker_xy, sample_xy = coords[: len(centroids)], coords[len(centroids) :]
    first = 0
    for i, (name, per_person) in enumerate(people.items()):
        mine = sample_xy[first : first + len(per_person)]
        first += len(per_person)
        if len(mine):
            ax.scatter(
                mine[:, 0], mine[:, 1], s=18, alpha=0.65,
                color=_PERSON_COLORS[i % len(_PERSON_COLORS)], label=name,
            )  # fmt: skip
    for st, color, (x, y) in zip(speakers, colors, speaker_xy, strict=True):
        ax.scatter(x, y, marker="*", s=300, color=color, edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(st.display_label, (x, y), textcoords="offset points", xytext=(8, 6))
    ax.margins(0.12)  # room for the labels beside the stars
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    if samples:
        ax.legend(title="DB の登録者")


def render_viz(
    session: Session, wave: np.ndarray, people: dict[str, list[np.ndarray]], out_path: Path
) -> None:
    """Write the 3-panel voice picture of ``session`` (speakers with a voiceprint) to ``out_path`` as a PNG."""
    # Imported here so the CLI's other commands stay fast; Agg keeps matplotlib from opening a window.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speakers = [st for st in session.speakers.values() if st.centroid is not None]
    colors = [_SPEAKER_COLORS[i % len(_SPEAKER_COLORS)] for i in range(len(speakers))]
    with plt.rc_context({"font.family": _font_family()}):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120, layout="constrained")
        try:
            _draw_pitch(axes[0], speakers, colors, wave)
            _draw_spectrum(axes[1], speakers, colors, wave)
            _draw_voiceprints(axes[2], speakers, colors, people)
            fig.savefig(out_path)
        finally:
            plt.close(fig)
