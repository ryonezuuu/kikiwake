# kikiwake（聞き分け）

会議・打ち合わせの録音を、**話者名付き**で文字起こしする CLI です。

- 文字起こしと録音内の話者分離は [Aqua Voice の Avalon API](https://aquavoice.com/avalon-api/docs) に任せます。
- 話者ごとの声を手元で**声紋**（話者埋め込みベクトル）にし、ローカルの声紋 DB と照合します。
  一度名前を教えた人は、別の会議の録音でも自動で同じ名前が付きます。
- AI エージェント用のスキル（[`skill/SKILL.md`](skill/SKILL.md)）付き。AI チャットに音声ファイルを渡すと、
  文字起こし → 未確定の話者を質問 → 名前を登録、までを AI が進めます。

```markdown
**[00:00:12] 田中**
では定例を始めます。まず先週の件ですが…

**[00:00:45] 話者1**
はい、その件は… ※同時発話あり
```

## しくみ

```
音声 ─▶ Avalon API（diarize）── 話者ごとの発言区間・単語の時刻
     ─▶ 重なりのない、はっきりした区間だけを選ぶ（1 人あたり最大 60 秒）
     ─▶ SpeechBrain ECAPA-TDNN で声紋ベクトルに変換
     ─▶ 声紋 DB と照合（コサイン類似度、1 対 1 の最適割り当て）
     ─▶ <stem>.md / <stem>.session.json（/ <stem>.voices.png）
```

| 類似度 | 表示 | 扱い |
|---|---|---|
| 0.60 以上 | `田中` | 自動で確定し、声紋を DB に追加 |
| 0.45〜0.60 | `田中？` | 候補。確認されるまで DB には追加しない |
| 0.45 未満 | `話者N` | 名前を付けたら DB に登録 |

声紋に使える声が 4 秒未満の人（相づちだけ、など）は「声紋不足」として照合しません。

## 必要なもの

- macOS（Apple Silicon で動作確認。GPU 不要）
- [uv](https://docs.astral.sh/uv/)（Python 3.12 を自動で用意します）
- `ffmpeg` / `ffprobe`（`brew install ffmpeg`）
- Aqua Voice の API キー

## インストール

```bash
git clone https://github.com/ryonezuuu/kikiwake.git
cd kikiwake
uv tool install .
```

コードを更新したら `uv tool install --reinstall .` で入れ直します。

API キーはシェルの設定（`~/.zshrc` など）に書きます。**`export` が必要です**（無いと `kikiwake` に渡りません）。

```bash
echo 'export AQUAVOICE_API="ここにキー"' >> ~/.zshrc
source ~/.zshrc
```

`AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY` の順に読みます。キーはログや出力に書き出しません。

初回の実行時に声紋モデル（約 90 MB）をダウンロードします。

## 使い方

### 1. 文字起こし

```bash
kikiwake transcribe 会議.m4a --language ja
```

`会議.md`（読む用）と `会議.session.json`（作業データ）が音声と同じ場所にできます。

| オプション | 説明 |
|---|---|
| `--output-dir DIR` | 出力先（既定は音声と同じディレクトリ） |
| `--language CODE` | 言語コード（既定は自動判定） |
| `--prompt TEXT` | 固有名詞・用語のヒント |
| `--max-speakers N` | 話者数の上限（1〜8） |
| `--viz` | 声の可視化画像 `会議.voices.png` も作る |
| `--force-api` | 保存済みの結果があっても API を呼び直す |
| `--json` | 標準出力に機械可読の JSON だけを出す（AI 向け） |

同じ音声をもう一度処理すると、保存済みの Avalon の結果を再利用します（API は呼ばれず、料金もかかりません）。

### 2. 名前を付ける

```bash
kikiwake label 会議.session.json spk_0=田中 spk_1=佐藤
```

- KEY には話者 ID（`spk_0`）か表示ラベル（`話者1`、`田中？`）を使えます。
- 間違えたら付け直すだけです（誤って登録した声紋は取り消されます）。
- 同じ人が 2 つの ID に分かれていたら、両方に同じ名前を付けると統合されます。

### 3. 声紋 DB の管理

```bash
kikiwake speakers list
kikiwake speakers rename 田中 田中太郎
kikiwake speakers delete 佐藤
```

### 4. 可視化

```bash
kikiwake viz 会議.session.json
```

話者ごとのピッチ（声の高さ）分布、長時間平均スペクトル、声紋の 2 次元マップを 1 枚の画像にします。

## AI スキルとして使う

Claude Code などの AI エージェントにスキルとして読み込ませます。

```bash
ln -s "$PWD/skill" ~/.claude/skills/kikiwake
```

AI チャットに音声ファイルを渡して「文字起こしして」と頼むと、AI が `kikiwake` を実行し、
名前の分からない話者について「00:12:30 の『予算は来月までに…』と話している人は誰ですか？」のように尋ねます。

## データの置き場所

| もの | 場所 |
|---|---|
| 声紋 DB | `~/.local/share/kikiwake/db.json` |
| 設定（任意） | `~/.local/share/kikiwake/config.toml` |
| 声紋モデル | `~/.local/share/kikiwake/models/` |

`KIKIWAKE_HOME` で場所を変えられます。閾値などは `config.toml` で上書きできます（例: `auto_threshold = 0.65`）。

文字起こし・作業データ・声紋 DB は個人情報を含むので、本人だけが読める権限で保存します。git にコミットしないでください。

## 制限

- 1 回の録音で区別できる話者は 8 人まで（Avalon の上限）。録音は 4 時間まで。
- 話者の境目で、発言の末尾が隣の人に付くことがあります（Avalon の話者分離の結果をそのまま使うため）。
- 似た声質の人を Avalon が 1 人にまとめてしまうことがあります。
- 閾値は初期値です。実際の会議で使いながら調整してください。

## 開発

```bash
uv sync
uv run pytest            # 単体テスト（ネット接続なし）
uv run pytest -m model   # 実際の声紋モデルを使うテスト（モデルをダウンロード）
uv run pytest -m e2e     # 実際の Avalon API を使う通しのテスト（API 料金がかかります）
```

設計書は [`docs/superpowers/specs/`](docs/superpowers/specs/)、実装計画は [`docs/superpowers/plans/`](docs/superpowers/plans/) にあります
（開発時の名前 `speaker-transcribe` のまま残しています）。
