# speaker-transcribe 設計書

- 作成日: 2026-10-26
- 状態: レビュー待ち

## 1. 目的と前提

### 目的

会議・打ち合わせ（3〜8 人程度）の録音ファイルを、**話者名付きの文字起こし**にする。
話者の名前は初回は「話者1」「話者2」と仮置きし、ユーザーが一度名前を教えると声紋を記憶して、
次回以降の会議では自動で名前が付く。

最終的な利用形態は **AI エージェントのスキル**。AI チャットに音声ファイルを渡すと、AI が CLI を実行し、
未確定の話者がいればチャットで「この発言をした人は誰ですか？」と尋ね、答えを CLI に反映する。

### 成功基準

1. 音声ファイル 1 本からコマンド 1 回で、時刻・話者ラベル付きの Markdown が得られる。
2. 一度名前を付けた人は、別の会議の録音でも（同程度の録音環境なら）自動で同じ名前が付く。
3. 自動判定の誤りを `label` コマンドで訂正でき、誤った声紋が DB に残らない。
4. 名前付けや再出力のために Avalon API を再度呼ばない（課金は録音 1 本につき 1 回）。
5. AI がスキル経由で、人間の介入なしに `transcribe` → 質問 → `label` を進められる。

### 前提・制約

- 実行環境: macOS（ユーザーの Mac）。GPU 不要。
- Python 3.11 以上、`uv` で環境管理。`ffmpeg` / `ffprobe` が PATH にあること。
- API キーは環境変数から読む。優先順: `AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY`
  （既存の yt-transcribe スキルと共通）。キーはログ・出力に一切書かない。
- 文字起こしと**録音内の話者分離**は Avalon API（`avalon-v1.5`, `diarize=true`）に任せる。
  本プロジェクトが自作するのは**録音をまたいだ話者の同定**（声紋照合）と、その可視化。

### Avalon API について確認済みの事実（2026-10-26 時点の公式ドキュメント / OpenAPI）

- 非同期ジョブ: `POST /v1/audio/transcription-jobs`（同期版と同じ multipart フィールド）、
  `GET /v1/audio/transcription-jobs/{job_id}`（`status`: `queued` / `succeeded` / `failed` / `cancelled` / `abandoned`）、
  `GET /v1/audio/transcription-jobs/{job_id}/result`（未完了時は `409 job_not_complete`）。
  `Idempotency-Key` ヘッダで安全に再送できる。
- 同期版 `POST /v1/audio/transcriptions` は最大 210 秒しか待たないため使わない。
- 上限: 256 MiB、デコード後 4 時間。
- `diarize=true` + `response_format=diarized_json` の応答（`DiarizedTranscription`）:
  - `text`, `language`, `duration`, `speakers[{id, first_seen}]`, `warnings[]`（`speaker_limit_reached`）, `usage`
  - `turns[{speaker, start, end, text, words[{text, start, end, speaker_confidence, overlap}]}]`
- 話者 ID（`spk_0`, `spk_1`, …）は**その文字起こし内で初めて話した順**の番号であり、録音をまたいで一致しない。
- 1 回の文字起こしで話者は最大 8 人。`max_speakers`（1〜8）で上限を下げられる。

## 2. 全体構成

```
音声ファイル
  ├─▶ audio      ffmpeg で (a) アップロード用 64kbps モノラル MP3、(b) 照合用 16kHz モノラル WAV を作る
  ├─▶ avalon     非同期ジョブで diarize → DiarizedTranscription（型付きデータ）
  ├─▶ voiceprint 話者ごとにきれいな区間を選び、声紋ベクトル（重心）を作る
  ├─▶ matcher    声紋 DB と照合し、spk_N → 名前 / 候補 / 話者N を決める
  ├─▶ speaker_db 自動確定分の声紋を DB に追加
  └─▶ render     <stem>.md と <stem>.session.json を書く（--viz なら viz も実行）

label <session.json> 話者1=田中
  └─▶ session を更新 → speaker_db を更新（追加・取り消し）→ render で再出力
```

### モジュール

パッケージ名 `speaker_transcribe`（`src/` レイアウト）。各モジュールは単独でテスト可能にする。

| モジュール | 責務 | 主な依存 |
|---|---|---|
| `config` | 設定値（閾値など）の既定値と、`config.toml` による上書き | 標準ライブラリ |
| `audio` | ffmpeg による変換、長さの取得、区間の波形切り出し | ffmpeg, numpy |
| `avalon` | API 呼び出し（ジョブ投入・ポーリング・結果取得・リトライ）と応答のパース | httpx |
| `voiceprint` | 区間選択、小片分割、埋め込み計算、重心と一貫性スコア | numpy, `Embedder` |
| `embedder` | `Embedder` インターフェースと SpeechBrain ECAPA-TDNN 実装 | speechbrain, torch |
| `speaker_db` | 人と声紋サンプルの永続化、追加・取り消し・改名・削除 | numpy |
| `matcher` | 類似度行列、最適割り当て、判定状態の決定 | numpy, scipy |
| `session` | `session.json` の読み書きとデータ構造 | 標準ライブラリ |
| `render` | Markdown 出力、AI 向け JSON サマリ | 標準ライブラリ |
| `viz` | ピッチ分布・長時間平均スペクトル・声紋 2D マップの描画 | librosa, matplotlib |
| `cli` | サブコマンドの定義と終了コード | argparse |

`Embedder` は `embed(waveform_16k_mono: np.ndarray) -> np.ndarray`（L2 正規化済み）と
`model_id: str`、`dim: int` を持つ。初期実装は `speechbrain/spkrec-ecapa-voxceleb`（192 次元）。
初回実行時に Hugging Face から自動ダウンロードし、データディレクトリ配下にキャッシュする。

### データの置き場所

- データディレクトリ: `~/.local/share/speaker-transcribe/`（環境変数 `SPEAKER_TRANSCRIBE_HOME` で変更可）
  - `db.json` … 声紋 DB
  - `config.toml` … 任意。設定値の上書き
  - `models/` … 埋め込みモデルのキャッシュ
- 出力: 既定は入力音声と同じディレクトリ（`--output-dir` で変更）
  - `<stem>.md`, `<stem>.session.json`, `<stem>.voices.png`（可視化時のみ）

## 3. CLI

コマンド名は `speaker-transcribe`。全サブコマンドが `--json` を受け付け、指定時は標準出力に機械可読の JSON
だけを出す（進捗ログは常に標準エラー）。

### `transcribe <audio>`

| オプション | 既定 | 説明 |
|---|---|---|
| `--output-dir DIR` | 入力と同じ | 出力先 |
| `--language CODE` | なし（自動判定） | Avalon に渡す ISO 639-1 コード |
| `--prompt TEXT` | なし | 固有名詞・用語のヒント |
| `--max-speakers N` | なし | Avalon の `max_speakers` |
| `--viz` | off | 処理後に `viz` を実行 |
| `--force-api` | off | キャッシュがあっても API を呼び直す |
| `--json` | off | AI 向けサマリを出力 |

動作:
1. 入力ファイルの SHA-256 を計算。既存の `<stem>.session.json` があり、ハッシュが一致すれば、
   保存済みの Avalon 応答を再利用する（`--force-api` 時を除く）。
2. 既存 session が DB に追加していた声紋サンプル（`auto` / `label` の両方）は、照合の前に**すべて取り消す**
   （再実行での二重追加と、自分自身との照合を防ぐ）。
3. 4 時間を超える音声はアップロード前にエラー。
4. 文字起こし → 声紋抽出 → 照合 → 自動確定分を DB に追加 → 出力。
5. キャッシュを再利用した場合（話者 ID が前回と同じ）は、前回 `confirmed` だった話者のラベルを
   照合後に `label` と同じ手順で**付け直す**。`--force-api` で API を呼び直した場合は話者 ID が変わり得るため
   付け直さず、「前回の名前付けは破棄されました」と警告する。

### `label <session.json> KEY=NAME [KEY=NAME ...]`

`KEY` には、話者 ID（`spk_2`）か、現在の表示ラベル（`話者1`、`田中`、`田中？`）を指定できる。
表示ラベルが複数の話者に一致する場合（統合後の `田中` など）はエラーとし、話者 ID の指定を求める。

| ケース | 動作 |
|---|---|
| `NAME` が DB に無い | 人を新規作成し、この話者の声紋を追加 |
| `NAME` が DB にあり、この会議では未割り当て | 声紋を追加。この話者が別人に自動割り当てされていた場合は、その自動追加サンプルを先に取り消す |
| `NAME` がこの会議の別の話者に `auto` / `confirmed` で割り当て済み | 同一人物とみなして**統合**。両方の表示を `NAME` にし、声紋も追加 |
| `NAME` がこの会議の別の話者の `candidate` | その別の話者は `unknown`（`話者N`）に戻し、この話者に `NAME` を付ける |
| この話者が「声紋不足」 | 表示名だけ変える。DB には追加せず、警告を出す |

ラベルを付けた話者の状態は `confirmed` になる。最後に `<stem>.md` を再出力し、`<stem>.voices.png` があればそれも再生成する。

### `speakers list | rename OLD NEW | delete NAME`

DB の人の一覧（サンプル数、最終更新日時）、改名、削除を行う。

### `viz <session.json>`

`<stem>.voices.png` を生成する（§6）。

### 終了コード

`0` 成功 / `1` 実行時エラー（API 失敗など）/ `2` 使い方・環境エラー（API キー無し、ffmpeg 無し、引数不正、モデル不一致）。

## 4. 声紋の抽出

既定値はすべて `config` で上書きできる。値は**初期値**であり、実データで調整する前提。

### 4.1 区間の選択

話者ごとに、Avalon の `words` から次の条件で区間を作る。

1. `overlap == false` かつ `speaker_confidence >= 0.7`（`min_word_confidence`）の単語だけを使う。
2. 同じ話者の隣り合う採用単語の間隔が 0.3 秒未満（`max_word_gap`）なら連結して 1 区間にする。
   間に不採用の単語や他の話者の単語が挟まる場合は連結しない。
3. 区間の両端を 0.1 秒ずつ削る（`edge_trim`）。
4. 削った後の長さが 1.5 秒未満（`min_segment`）の区間は捨てる。
5. 長い区間から順に採用し、合計 60 秒（`max_total_per_speaker`）に達したら打ち切る。

採用区間の合計が 4 秒未満（`min_total_per_speaker`）の話者は**声紋不足**（`insufficient`）とし、照合も DB 追加もしない。

### 4.2 声紋ベクトル

1. 各区間を 3 秒（`chunk_seconds`）の小片に分割する。端数が 1.5 秒以上なら独立した小片、未満なら直前の小片に含める。
2. 各小片を `Embedder` でベクトル化（L2 正規化済み）。
3. 全小片の平均を取り、再び L2 正規化したものを、その会議でのその話者の**重心**とする。
4. **一貫性スコア** = 各小片と重心のコサイン類似度の平均。0.5 未満（`min_consistency`）なら
   「複数人が混ざっている可能性」の警告を session と Markdown に出す（判定自体は続行）。

## 5. 照合と声紋 DB

### 5.1 DB 形式（`db.json`）

```json
{
  "version": 1,
  "model_id": "speechbrain/spkrec-ecapa-voxceleb",
  "dim": 192,
  "people": {
    "田中": {
      "created_at": "2026-10-26T10:00:00+09:00",
      "samples": [
        {
          "id": "6f1c…（UUID）",
          "session_sha256": "…",
          "source": "auto | label",
          "created_at": "…",
          "embedding": [0.01, …]
        }
      ]
    }
  }
}
```

- 1 人あたりのサンプルは新しい順に最大 20 件（`max_samples_per_person`）。超えたら最古を削除。
- `model_id` / `dim` が現在の `Embedder` と一致しなければ終了コード 2 で停止する（古い声紋とは比較できないため）。
- 書き込みは一時ファイル → `os.replace` で原子的に行う。

### 5.2 類似度

話者 *s*（重心 *c*）と人 *p* の類似度 = *p* のサンプルと *c* のコサイン類似度の**上位 3 件の平均**
（サンプルが 3 件未満ならある分の平均）。

### 5.3 割り当て

1. 声紋不足でない話者 × DB の人で類似度行列を作る。
2. 類似度の合計が最大になるよう、`scipy.optimize.linear_sum_assignment` で 1 対 1 に割り当てる
   （同じ人が 2 人の話者に自動で割り当てられることはない）。
3. 割り当てられた組の類似度で状態を決める。

| 類似度 | 状態 | 表示 | DB への追加 |
|---|---|---|---|
| ≥ 0.60（`auto_threshold`） | `auto` | `田中` | 自動で追加（`source: "auto"`） |
| 0.45 以上 0.60 未満（`candidate_threshold`） | `candidate` | `田中？` | 追加しない（確認待ち） |
| < 0.45、または割り当てなし | `unknown` | `話者N` | 追加しない |
| 声紋不足 | `insufficient` | `話者N` | 追加しない |

`話者N` の番号は、`transcribe` 時点で全話者（状態を問わない）に初めて話した順で 1 から振り、session に保存して
以後変えない（`label` の途中で番号がずれて、AI やユーザーが参照したラベルが別人を指すことを防ぐ）。
`candidate` から `unknown` に戻った話者も、この番号を使う。

## 6. 出力

### 6.1 `<stem>.md`

```markdown
# meeting.m4a の文字起こし

- 録音: 52 分 / 話者: 田中・佐藤・田中？・話者1・話者2

> ⚠ 名前が未確定の話者が 3 人います。

**[00:00:12] 田中**
では定例を始めます。まず先週の件ですが…

**[00:00:45] 話者1**
はい、その件は… ※同時発話あり
```

- 表示ラベルが同じ連続ターンは 1 段落にまとめる（統合された話者も同じ扱い）。
- 段落内の単語の過半数が `overlap == true` なら末尾に「※同時発話あり」を付ける。
- 冒頭の注意書き: 未確定の話者（`candidate` / `unknown` / `insufficient`）の人数、
  `speaker_limit_reached`（「9 人以上いた可能性があり、一部の発言は別の話者に割り当てられています」）、一貫性の警告。

### 6.2 `<stem>.session.json`

- `version`, `audio_path`, `audio_sha256`, `created_at`, `model_id`
- `avalon_response`: Avalon の `diarized_json` 応答そのもの（キャッシュ）
- `speakers`: 話者 ID ごとに
  - `segments`（採用区間 `[start, end]` の配列）, `total_seconds`, `centroid`, `consistency`
  - `scores`（DB の人ごとの類似度。閾値調整の根拠に使う）
  - `status`（`auto` / `candidate` / `unknown` / `insufficient` / `confirmed`）, `name`, `display_label`
  - `db_sample_ids`（この session が DB に追加したサンプル ID。取り消しに使う）
- `warnings`

### 6.3 AI 向け JSON サマリ（`--json`）

```json
{
  "markdown_path": "…/meeting.md",
  "session_path": "…/meeting.session.json",
  "viz_path": null,
  "speakers": [
    {"id": "spk_0", "label": "田中", "status": "auto", "score": 0.72},
    {"id": "spk_2", "label": "田中？", "status": "candidate", "score": 0.51,
     "examples": [{"time": "00:12:30", "text": "予算は来月までに…"}]},
    {"id": "spk_3", "label": "話者1", "status": "unknown",
     "examples": [{"time": "00:00:45", "text": "はい、その件は…"}]}
  ],
  "warnings": []
}
```

`examples` は `auto` / `confirmed` 以外の話者に付け、その話者の長いターンから順に最大 3 件、本文は 60 文字で切る。

### 6.4 可視化（`<stem>.voices.png`）

横に 3 枚並べた 1 枚の画像。対象は声紋不足でない話者で、凡例には表示ラベルを使う。

1. **ピッチ分布**: 採用区間の基本周波数（`librosa.pyin`, 60〜500 Hz）のヒストグラム（話者ごとに色分け）。
2. **長時間平均スペクトル**: 採用区間のパワースペクトルの平均（dB、0〜8 kHz）。
3. **声紋 2D マップ**: この会議の話者の重心と、DB の全サンプルを PCA で 2 次元に投影した散布図。
   会議の話者は大きな印、DB のサンプルは人ごとの色の小さな点で描く。

## 7. エラー処理

| 状況 | 動作 |
|---|---|
| API キーが無い | どの環境変数を設定すべきかを示して終了コード 2 |
| ffmpeg / ffprobe が無い | `brew install ffmpeg` を案内して終了コード 2 |
| 音声が 4 時間超 | アップロードせず終了コード 1 |
| 429 / 5xx / ネットワーク断 | 指数バックオフ（2, 4, 8 秒…）で最大 5 回リトライ。ジョブ投入には毎回同じ `Idempotency-Key` を付ける |
| その他の 4xx | リトライせず、ステータスと API のエラー内容を表示して終了コード 1 |
| ジョブが `failed` / `cancelled` / `abandoned` | `error_code` を表示して終了コード 1 |
| ジョブが 60 分（`job_timeout`）で終わらない | ジョブ ID を表示して終了コード 1 |
| ポーリング間隔 | 5 秒から開始し、最大 15 秒まで伸ばす |
| モデルのダウンロード失敗 | ネットワーク確認を促して終了コード 1 |
| DB のモデル不一致 | §5.1 のとおり終了コード 2 |
| `label` の KEY が見つからない | 有効な KEY の一覧を表示して終了コード 2。session も DB も変更しない |

## 8. テスト方針

- **単体テスト（pytest、ネット接続なし）**
  - `voiceprint`: 手作りの `DiarizedTranscription` から、区間選択（信頼度・重なり・連結・端の削り・最短長・合計上限）を検証。
    `Embedder` は決定的な偽実装を使う。
  - `matcher`: 人工ベクトルで、閾値ごとの状態、1 対 1 割り当て、上位 3 件平均を検証。
  - `speaker_db`: 追加・上限 20 件・取り消し・改名・削除・モデル不一致・原子的書き込み。
  - `session` + `label`: 新規名・既存名・訂正（自動サンプルの取り消し）・統合・候補の解除・声紋不足・
    不正 KEY・曖昧な KEY・キャッシュ再利用時のサンプル取り消しとラベル付け直し。
  - `render`: 期待する Markdown / JSON と比較（段落まとめ、同時発話、注意書き）。
  - `avalon`: HTTP をモックし、ジョブ投入 → ポーリング → 結果取得、リトライ、失敗状態。
    応答のフィクスチャは本物の応答を一度保存したものを使う。
- **通しのテスト（手動、`-m e2e`）**: macOS の `say` コマンドで複数の日本語音声を使った疑似会議を 2 本作り、
  1 本目で名前を付けたあと、2 本目で同じ音声に同じ名前が自動で付くことを確認する。
  実 API と実モデルを使う。

## 9. AI スキル

リポジトリに `skill/SKILL.md` を置く（インストール方法は実装計画で決める）。内容:

1. 音声ファイルを受け取ったら `speaker-transcribe transcribe <file> --json` を実行する。
2. `speakers` に `auto` / `confirmed` 以外の話者がいれば、`examples` を引用してユーザーに名前を尋ねる。
   `candidate` は「〜さんで合っていますか？」と確認する。
3. 答えを `speaker-transcribe label <session> spk_N=NAME ... --json` で反映する（曖昧さを避けるため、KEY には話者 ID を使う）。
   わからないと言われた話者はそのままにする。`candidate` が合っていると言われた場合も、同じく `label` で確定させる。
4. 最後に Markdown のパスと要点をユーザーに伝える。

## 10. 対象外

リアルタイム文字起こし、GUI、9 人以上の区別、文字起こし本文の編集、4 時間を超える音声の分割、
複数ユーザーでの DB 共有。
