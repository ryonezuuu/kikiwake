---
name: speaker-transcribe
description: 会議・打ち合わせの音声ファイルを、話者名付きで文字起こしする（Avalon API + ローカル声紋 DB）。音声/動画ファイル（m4a, mp3, wav, mp4 など）を渡されて「文字起こし」「議事録」「誰が何を話したか」を求められたときに使う。
---

# speaker-transcribe

会議音声を Avalon API で文字起こし・話者分離し、ローカルの声紋 DB と照合して話者に名前を付ける CLI `speaker-transcribe` を使う。
名前を一度教えれば声紋 DB に溜まり、次回以降の会議では同じ人に自動で名前が付く。

## 前提

- `speaker-transcribe` が PATH 上にあること。無ければリポジトリのルートで `uv tool install .` を実行する。
  コードを変更したあとは `uv tool install --reinstall .` で入れ直す。リポジトリのルートは、このスキルの
  シンボリックリンク先の親ディレクトリ（`dirname "$(readlink ~/.claude/skills/speaker-transcribe)"`）。
- `ffmpeg` / `ffprobe` が必要（無ければ `brew install ffmpeg`）。
- Avalon の API キーは、ユーザーのシェル設定（`~/.zshrc` など）で `export AQUAVOICE_API=...` として設定された環境変数から読む
  （`AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY` の順）。`export` を付けないと子プロセスに渡らない。
  **API キーをチャットに貼らせない。** 設定されていなければ、設定方法だけを案内する。
- 初回実行では声紋モデル（約 90 MB）をダウンロードするので時間がかかる。ネットワークが必要。
- 長い録音は数分以上かかることがある。コマンドのタイムアウトを長めにするか、バックグラウンドで実行する。
- ユーザーの音声ファイル、`.md`、`.session.json`、`.voices.png` は個人情報を含む。git にコミットしない
  （音声が git 管理下のディレクトリにあるときは `--output-dir` で管理外に出力するか、`git add` しない）。

## 手順

### 1. 文字起こしする

```bash
speaker-transcribe transcribe "<音声ファイルのパス>" --json
```

- パスは引用符で囲む。標準出力には JSON だけが出る。進捗は標準エラーに出る。
- 言語が分かっているなら `--language ja`、固有名詞のヒントは `--prompt "..."`、人数が分かっているなら `--max-speakers N`（1〜8）。
- 出力先は既定で音声と同じ場所（`<stem>.md` と `<stem>.session.json`）。変えるときは `--output-dir DIR`。
- 同じ音声をもう一度処理すると保存済みの結果を再利用する（API は呼ばれない）。API から取り直すときだけ `--force-api`
  （話者 ID が変わり得るので、それまでの名前付けは破棄される）。
- 保存済みの結果を再利用するときは `--language` / `--prompt` / `--max-speakers` は**反映されない**
  （標準エラーに「保存済みの文字起こしを再利用するため … は反映されません（反映するには --force-api）」と出る）。
  反映させたいときだけ `--force-api` を付ける（API の料金がもう一度かかるので、ユーザーに確認してから）。
- 声の可視化画像（`<stem>.voices.png`）は、ユーザーが見たいと言ったときだけ `--viz` を付けて作る。あとから作るなら `speaker-transcribe viz "<session_path>"`。

**終了コードが 2 のとき**（API キー未設定、ffmpeg 未導入、引数不正、声紋 DB のモデル不一致）は、標準エラーの
`エラー: ...` をそのままユーザーに伝え、直し方を案内して止まる。勝手にリトライしない。
**終了コードが 1 のとき**（API 失敗、4 時間超の音声、モデルのダウンロード失敗など）も、エラー文をそのまま伝える。

### 2. 結果を読む

標準出力の JSON:

```json
{
  "markdown_path": "…/meeting.md",
  "session_path": "…/meeting.session.json",
  "viz_path": null,
  "speakers": [
    {"id": "spk_0", "label": "田中", "status": "auto", "score": 0.72},
    {"id": "spk_2", "label": "田中？", "status": "candidate", "score": 0.51,
     "examples": [{"time": "00:12:30", "text": "予算は来月までに…"}]},
    {"id": "spk_3", "label": "話者1", "status": "unknown", "score": null,
     "examples": [{"time": "00:00:45", "text": "はい、その件は…"}]}
  ],
  "warnings": []
}
```

`status` の意味:

| status | 意味 | 対応 |
|---|---|---|
| `auto` | 声紋 DB と一致して自動で名前が付いた | そのまま使う |
| `confirmed` | ユーザーが名前を確定済み | そのまま使う |
| `candidate` | 似ている人がいる（`label` は `田中？`） | 「〜さんで合っていますか？」と確認する |
| `unknown` | 該当する人がいない（`話者N`） | 名前を尋ねる |
| `insufficient` | 声紋に使える発話が少なすぎる（`話者N`） | 名前は付けられるが声紋 DB には登録されない。そう伝える |

`warnings` が空でなければ、ユーザーに伝える（9 人以上いた可能性、声の一貫性の警告、
前回の名前付けを引き継げなかった人など）。ただし「名前が未確定の話者が N 人います。」は、
その話者について手順 3 でユーザーに尋ねたあとは繰り返さなくてよい。

標準エラーに次の行が出ることがある。

- `警告: ...`（`transcribe` / `label` とも）: 処理は成功している。内容をユーザーに伝える。
  `警告: 可視化をスキップしました: ...` のときは画像だけ作れなかった（`viz_path` は `null`）。
- `別の speaker-transcribe が DB を使用中のため待機しています…`: 別の実行が声紋 DB を使い終わるのを待っているだけ。
  そのまま待つ（中断しない）。

### 3. 名前を尋ねて反映する

`auto` / `confirmed` 以外の話者がいるときだけ、ユーザーに尋ねる。

- `examples` の**時刻と発言を引用**して、誰の声かを尋ねる（例:「00:00:45 の『はい、その件は…』と話している人は誰ですか？」）。
- `candidate` は「00:12:30 の『予算は来月までに…』は田中さんで合っていますか？」のように確認する。
- 答えを `label` で反映する。**KEY には必ず話者 ID（`spk_N`）を使う**（表示ラベルは `label` のたびに変わるため）。
  複数人分をまとめて 1 回で渡せる。

```bash
speaker-transcribe label "<session_path>" spk_2=田中 spk_3=佐藤 --json
speaker-transcribe label "<session_path>" "spk_4=山田 太郎" --json   # 空白を含む名前は KEY=NAME ごと引用符で囲む
```

- `candidate` が合っていると言われたら、`？` を除いた名前で同じように `label` する（`spk_2=田中`）。違うと言われたら別の名前を尋ねる。
- ユーザーが**わからないと言った話者には `label` しない**。そのまま `話者N` として残す。名前を推測で付けない。
- 同じ人が 2 つの話者 ID に分かれていると分かったら、両方に同じ名前を付ける（自動的に統合される）。
- 名前は「さん」などを付けず、ユーザーが言った表記で統一する。迷ったら先に `speaker-transcribe speakers list --json` で登録済みの名前を確認する。
  名前は `？` / `?` で終われない。
- `label` は標準エラーに `警告: ...` を出すことがある（声紋不足で DB に登録しなかった、など）。そのままユーザーに伝える。
- `label` は `<stem>.md` を作り直す。`<stem>.voices.png` が既にあれば、それも作り直す。
  標準出力の JSON は `transcribe` と同じ形式なので、残りの未確定の話者を確認できる。
- 名前が声紋 DB に無い人は、このときに新しく登録される。

### 4. ユーザーに報告する

- `markdown_path` を伝える（可視化したときは `viz_path` も）。
- 誰が参加していたか（確定した名前と、まだ `話者N` のままの人）と、会議の要点を短く伝える。要点は `.md` を読んで書く。
- `warnings` があれば添える。

## 声紋 DB の管理

```bash
speaker-transcribe speakers list [--json]          # 登録済みの人、サンプル数、最終更新日時
speaker-transcribe speakers rename "旧名" "新名"          # 新名は label の名前と同じ規則（空・末尾の ？ / ? は不可）
speaker-transcribe speakers delete "名前"           # その人の声紋をすべて削除
```

`rename` と `delete` はユーザーに頼まれたときだけ行う。DB は `~/.local/share/speaker-transcribe/db.json`
（`SPEAKER_TRANSCRIBE_HOME` で変更可）。
