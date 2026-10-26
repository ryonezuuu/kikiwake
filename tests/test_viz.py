import matplotlib
import numpy as np
from conftest import make_session, session_from_words, unit
from matplotlib.colors import to_rgb

from speaker_transcribe.audio import SAMPLE_RATE
from speaker_transcribe.viz import _SPEAKER_COLORS, _draw_voiceprints, project_2d, render_viz, speaker_f0

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sine_wave(freq: float, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def two_speaker_session_with_sine_waves():
    """A 20 s recording: spk_0 is a 150 Hz tone on 0-10 s, spk_1 a 250 Hz tone on 10-20 s."""
    wave = np.concatenate([sine_wave(150, 10), sine_wave(250, 10)])
    session = session_from_words(
        [("spk_0", 0, 10, 0.9, False), ("spk_1", 10, 20, 0.9, False)],
        duration=20.0,
        states={
            "spk_0": dict(
                segments=[(0.5, 9.5)],
                centroid=unit(0).tolist(),
                status="auto",
                name="田中",
            ),
            "spk_1": dict(segments=[(10.5, 19.5)], centroid=unit(1).tolist()),
        },
    )
    return session, wave


def session_with_only_insufficient():
    return make_session(
        spk_0=dict(centroid=None, status="insufficient"),
        spk_1=dict(centroid=None, status="insufficient"),
        spk_2=dict(centroid=None, status="insufficient"),
    )


def test_writes_png(tmp_path):
    session, wave = two_speaker_session_with_sine_waves()
    out = tmp_path / "m.voices.png"
    render_viz(session, wave, {"田中": [unit(0), unit(1)]}, out)
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_no_eligible_speakers(tmp_path):
    render_viz(session_with_only_insufficient(), np.zeros(16000, np.float32), {}, tmp_path / "x.png")
    assert (tmp_path / "x.png").exists()


def test_no_eligible_speakers_ignores_the_db(tmp_path):
    out = tmp_path / "x.png"
    render_viz(session_with_only_insufficient(), np.zeros(16000, np.float32), {"田中": [unit(0), unit(1)]}, out)
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_silence_and_unseen_segments_still_render(tmp_path):
    # Silent speech, a slice shorter than the FFT window, and a slice beyond the end of the audio.
    session = make_session(spk_0=dict(segments=[(0.0, 1.0), (1.0, 1.01), (50.0, 60.0)]))
    out = tmp_path / "x.png"
    render_viz(session, np.zeros(2 * SAMPLE_RATE, np.float32), {}, out)
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_merged_speakers_with_the_same_name_and_identical_centroids(tmp_path):
    session = make_session(
        spk_0=dict(status="auto", name="田中"),
        spk_1=dict(status="auto", name="田中", centroid=unit(0).tolist()),
    )
    out = tmp_path / "x.png"
    render_viz(session, np.zeros(2 * SAMPLE_RATE, np.float32), {"田中": [unit(0)]}, out)
    assert out.read_bytes()[:8] == PNG_MAGIC


def test_speaker_f0_follows_the_pitch_of_the_voice():
    _, wave = two_speaker_session_with_sine_waves()
    low = speaker_f0(wave, [(0.5, 9.5)])
    high = speaker_f0(wave, [(10.5, 19.5)])
    assert len(low) > 100 and len(high) > 100
    assert abs(np.median(low) - 150) < 5
    assert abs(np.median(high) - 250) < 5


def test_speaker_f0_skips_silence_and_slices_that_are_too_short():
    wave = np.concatenate([np.zeros(SAMPLE_RATE, np.float32), sine_wave(150, 1)])
    assert speaker_f0(wave, [(0.0, 1.0)]).size == 0  # silence is unvoiced
    assert speaker_f0(wave, [(1.0, 1.05)]).size == 0  # shorter than two pyin frames
    assert speaker_f0(wave, [(5.0, 6.0)]).size == 0  # outside the audio
    assert speaker_f0(wave, [(0.0, 1.0), (1.2, 1.8)]).size > 0  # the voiced slice still counts


def test_project_2d_separates_orthogonal_points():
    points = np.stack([unit(0), unit(1), unit(2)])
    coords = project_2d(points)
    assert coords.shape == (3, 2)
    dists = [np.linalg.norm(coords[i] - coords[j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    assert np.allclose(dists, np.sqrt(2))  # three equidistant points span the plane exactly


def test_project_2d_leaves_no_noise_in_directions_without_variance():
    points = np.array([[0.3, 0.7, 0.1, 0.2], [0.9, 0.1, 0.4, 0.5]])  # two points span a line, not a plane
    coords = project_2d(points)
    assert np.isclose(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(points[0] - points[1]))
    assert np.array_equal(coords[:, 1], np.zeros(2))


def test_project_2d_puts_identical_points_at_the_origin():
    coords = project_2d(np.stack([unit(0), unit(0)]))
    assert np.array_equal(coords, np.zeros((2, 2)))


def test_voiceprint_map_colours_db_people_by_identity():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session = make_session(
        spk_0=dict(status="auto", name="田中"),
        spk_1=dict(status="candidate", name="佐藤"),  # a guess: drawn as 佐藤？, so no shared colour
    )
    speakers = list(session.speakers.values())
    colors = list(_SPEAKER_COLORS[:3])
    people = {"佐藤": [unit(1)], "鈴木": [unit(3)], "田中": [unit(0), unit(0)]}
    fig, ax = plt.subplots()
    try:
        _draw_voiceprints(ax, speakers, colors, people)
        dots = {c.get_label(): c for c in ax.collections if c.get_label() in people}
    finally:
        plt.close(fig)

    def rgb(name):
        return tuple(dots[name].get_facecolor()[0][:3])

    assert rgb("田中") == to_rgb(colors[0])  # the same person as the 田中 star
    speaker_rgbs = {to_rgb(c) for c in _SPEAKER_COLORS}
    assert rgb("佐藤") not in speaker_rgbs and rgb("鈴木") not in speaker_rgbs
    assert all(d.get_alpha() < 1 for d in dots.values())
