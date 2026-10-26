"""End-to-end: two fake meetings voiced by macOS ``say``, the real Avalon API and the real embedding model.

Run with ``AQUAVOICE_API`` set: ``uv run pytest -m e2e -v -s``.
"""

import json
import subprocess
from pathlib import Path

import pytest
from conftest import say_wav

from speaker_transcribe.audio import SAMPLE_RATE
from speaker_transcribe.cli import main

pytestmark = pytest.mark.e2e

# The Japanese Grandpa and Sandy voices must be addressed by their full names:
# plain "Grandpa" / "Sandy" select the English voices.
KYOKO = "Kyoko"
GRANDPA = "Grandpa (日本語（日本）)"
SANDY = "Sandy (日本語（日本）)"

GAP_SECONDS = 0.7

# Meeting A: Kyoko -> Grandpa -> Sandy -> Kyoko -> Grandpa -> Sandy (each voice twice).
MEETING_A = [
    (KYOKO, "皆さん、お疲れ様です。それでは来週の定例会議を始めます。まず先月の売上の報告からお願いします。"),
    (GRANDPA, "はい、承知しました。先月の売上は前の年より約一割増えました。関西地方が特に好調でした。"),
    (SANDY, "ありがとうございます。次に広告の予算について相談させてください。来月からテレビで宣伝を始めます。"),
    (KYOKO, "分かりました。予算の上限は三百万円までですので、媒体ごとの金額を表にまとめてください。"),
    (GRANDPA, "了解です。工場の生産計画も見直しておきます。部品の注文は、今週中に出す予定です。"),
    (SANDY, "最後に、お客様の問い合わせ窓口についてです。電話の受付時間を、午後六時まで広げたいと考えています。"),
]

# Meeting B: Grandpa -> Sandy -> Kyoko -> Grandpa -> Sandy -> Kyoko, with different sentences.
MEETING_B = [
    (GRANDPA, "お待たせしました。今日は新しいオフィスへの引っ越しについて話し合います。駅から歩いて五分の便利な建物です。"),
    (SANDY, "私も下見に行ってきました。窓が大きくて明るい部屋でしたが、会議室が少し狭いです。"),
    (KYOKO, "引っ越しの日は来月の最終週を考えています。荷物の整理は各自で進めて、不要な書類は処分してください。"),
    (GRANDPA, "パソコンの配線は、業者に頼むことにしました。見積もりが届きましたので、後ほどメールで共有します。"),
    (SANDY, "新しい住所の連絡も忘れずに行いましょう。名刺と封筒の印刷は、案内状と一緒に注文します。"),
    (KYOKO, "では以上で会議を終わります。次回は引っ越しが終わってから、新しい会議室で集まりましょう。"),
]


def concat_with_gaps(parts: list[Path], out: Path) -> Path:
    """Join the 16 kHz mono WAVs ``parts`` into ``out`` with ``GAP_SECONDS`` of silence between them."""
    silence = out.with_name(f"{out.stem}_silence.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-t", str(GAP_SECONDS),
        "-c:a", "pcm_s16le",
        str(silence),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)

    inputs: list[str] = []
    for i, part in enumerate(parts):
        if i:
            inputs += ["-i", str(silence)]
        inputs += ["-i", str(part)]
    n = 2 * len(parts) - 1
    graph = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", graph,
        "-map", "[out]",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL)
    return out


def make_meeting(utterances: list[tuple[str, str]], directory: Path, stem: str) -> Path:
    """``directory/stem.wav``: the utterances voiced in order, with the pieces kept in ``directory/stem_parts``."""
    parts_dir = directory / f"{stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        say_wav(voice, text, parts_dir / f"{i}.wav") for i, (voice, text) in enumerate(utterances)
    ]
    return concat_with_gaps(parts, directory / f"{stem}.wav")


def make_meeting_a(tmp_path: Path) -> Path:
    return make_meeting(MEETING_A, tmp_path, "meeting_a")


def make_meeting_b(tmp_path: Path) -> Path:
    return make_meeting(MEETING_B, tmp_path, "meeting_b")


def test_names_carry_over_between_meetings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPEAKER_TRANSCRIBE_HOME", str(tmp_path / "home"))
    a, b = make_meeting_a(tmp_path), make_meeting_b(tmp_path)

    assert main(["transcribe", str(a), "--language", "ja", "--json"]) == 0
    sa = json.loads(capsys.readouterr().out)
    with capsys.disabled():
        print("\nmeeting A:", json.dumps(sa["speakers"], ensure_ascii=False, indent=2))
    assert len(sa["speakers"]) == 3, sa

    assert main(["label", sa["session_path"], "spk_0=京子", "spk_1=祖父", "spk_2=サンディ", "--json"]) == 0
    capsys.readouterr()

    assert main(["transcribe", str(b), "--language", "ja", "--viz", "--json"]) == 0
    sb = json.loads(capsys.readouterr().out)
    with capsys.disabled():
        print("\nmeeting B:", json.dumps(sb["speakers"], ensure_ascii=False, indent=2))
        print("voices picture:", sb["viz_path"])
    assert [(s["label"], s["status"]) for s in sb["speakers"]] == [
        ("祖父", "auto"),
        ("サンディ", "auto"),
        ("京子", "auto"),
    ], sb
    assert Path(sb["viz_path"]).is_file()
