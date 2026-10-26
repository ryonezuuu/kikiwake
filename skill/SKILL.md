---
name: kikiwake
description: 会議・打ち合わせの音声ファイルを、話者名付きで文字起こしする（Avalon API + ローカル声紋 DB）。音声/動画ファイル（m4a, mp3, wav, mp4 など）を渡されて「文字起こし」「議事録」「誰が何を話したか」を求められたときや、過去の会議の振り返りに使う。
---

# kikiwake

会議音声を Avalon API で文字起こし・話者分離し、声紋 DB と照合して話者に名前を付ける CLI `kikiwake` を使う。
名前を一度教えれば声紋 DB に溜まり、次回以降の会議では同じ人に自動で名前が付く。

結果は**記録フォルダ**に会議ごとに溜める（文字起こし・作業データ・要点 `summary.md`、全体の一覧 `index.md`）。
あとから「前回の会議で誰が何を言ったか」と聞かれたら、この記録を読んで答える。

## 0. 環境を判定する（最初に必ず）

| モード | 判定 | 記録フォルダ（以下 `<記録フォルダ>`） |
|---|---|---|
| **Cowork** | Mac 側シェルで `[ -d "$HOME/mnt/kikiwake" ]` が真 | `$HOME/mnt/kikiwake` |
| **Mac** | Cowork ではなく、`command -v kikiwake` が通る | `$HOME/Documents/kikiwake` |

- Cowork には 2 種類のシェルがある。**Mac 側のシェル（`device_bash`）だけ**が、接続したフォルダ
  （`$HOME/mnt/<フォルダ名>`）を見られる。クラウド側のシェルでは**kikiwake を実行しない**
  （Mac のフォルダが見えず、ここで `$HOME/mnt/kikiwake` が無くても「未接続」とは限らない。Mac 側で確かめ直す）。
- Mac 側シェルで `$HOME/mnt/kikiwake` が無いときは、**止まって**「会話の + → Add folder で kikiwake フォルダを追加してください」と
  案内する（フォルダ名は `kikiwake` であること）。
- どちらでもない（`kikiwake` が無い）とき、Mac の Claude Code なら、リポジトリのルートで `uv tool install .` を実行する。
  リポジトリのルートは、このスキルのシンボリックリンク先の親ディレクトリ（`dirname "$(readlink ~/.claude/skills/kikiwake)"`）。
  コードを変更したあとは `uv tool install --reinstall .` で入れ直す。`ffmpeg` / `ffprobe` が必要（無ければ `brew install ffmpeg`）。
  このスキルにあるコマンドが無いと言われる（例: `kikiwake index` で `invalid choice`）ときも、入っている版が古いので同じく入れ直す。
- Mac モードでは、`echo "$KIKIWAKE_HOME"` が `$HOME/Documents/kikiwake/.kikiwake` でなければ、
  声紋 DB が Cowork と別になってしまう。`~/.zshrc` に
  `export KIKIWAKE_HOME="$HOME/Documents/kikiwake/.kikiwake"` を追加するようユーザーに案内する（勝手に書き換えない）。
  - 設定のあと、`~/.local/share/kikiwake/db.json` があるのに `$HOME/Documents/kikiwake/.kikiwake/db.json` が無ければ、
    新しい声紋 DB は空から始まる（これまでの人に名前が付かない）。README の「インストール」の 2 にあるコピーの手順を案内する
    （頼まれたら代わりにコピーしてよい。上書きしない・元は消さない）。`kikiwake speakers list` に登録済みの人が並べば移行できている。
  - 設定前から開いている Claude Code のシェルには `KIKIWAKE_HOME` や `AQUAVOICE_API` が渡っていないことがある。
    `zsh -i -c '…'` で実行するか、Claude Code を開き直してもらう。
- シェルで記録フォルダを使うときは `"$HOME/mnt/kikiwake/records/…"` のように **`$HOME` を使う**（引用符の中の `~` は展開されない）。

## 1. 準備

### Cowork モード（会話ごとに 1 回。すでに `command -v kikiwake` が通れば省略）

Cowork の VM は会話ごとに作り直されるので、会話の最初に kikiwake を GitHub から入れる。
**数分かかるので、始める前にユーザーに伝える**（その間は待つ。中断しない）。

```bash
export KIKIWAKE_HOME="$HOME/mnt/kikiwake/.kikiwake" KIKIWAKE_INPLACE_WRITES=1
if ! command -v kikiwake >/dev/null; then
  git clone --depth 1 https://github.com/ryonezuuu/kikiwake "$HOME/kikiwake-src" \
    || { mkdir -p "$HOME/kikiwake-src" && curl -sL https://codeload.github.com/ryonezuuu/kikiwake/tar.gz/refs/heads/master | tar xz --strip-components=1 -C "$HOME/kikiwake-src"; }
  (cd "$HOME/kikiwake-src" && uv sync --frozen --no-dev --python 3.12)
  export PATH="$HOME/kikiwake-src/.venv/bin:$PATH"
fi
```

- 過去の会議を振り返るだけ（`index.md` / `summary.md` / 文字起こしを読むだけ）なら、インストールは**不要**（3 参照）。
- 同じ会話の 2 回目以降で、すでに `$HOME/kikiwake-src/.venv/bin/kikiwake` があれば、インストールは済んでいる。
  下の `export` 行だけ付ければよい。
- インストールの途中でコマンドがタイムアウトなどで切れたら、**同じスニペットをもう一度実行してよい**（やり直しても安全）。
- **シェルの状態はコマンド間で引き継がれないことがある。** 以降の `kikiwake` コマンドは、毎回この 1 行を先頭に付けるか、
  `&&` でつないで 1 回の呼び出しにする。

  ```bash
  export KIKIWAKE_HOME="$HOME/mnt/kikiwake/.kikiwake" KIKIWAKE_INPLACE_WRITES=1 PATH="$HOME/kikiwake-src/.venv/bin:$PATH"
  ```

- `KIKIWAKE_INPLACE_WRITES=1` は、ファイルの削除が禁止されたフォルダでも保存できるようにする設定。必ず付ける。
- インストールに失敗したら、エラー内容をそのまま伝えて止まる（API はまだ呼ばれていないので料金はかからない）。
  ブロックされたホストが原因なら「うまくいかないとき」を見る。

### API キー

API キーは次の順に読まれる: 環境変数 `AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY` → **`<KIKIWAKE_HOME>/aquavoice_api_key`**
（ファイルの 1 行目）。Cowork の VM では環境変数が無いので、キーファイルを使う。

- **API キーをチャットに貼らせない。出さない。** キーファイルの中身を `cat` などで表示しない（有無は `[ -s "$KIKIWAKE_HOME/aquavoice_api_key" ]` で確かめる）。
- キーが見つからない（終了コード 2）ときは、ユーザーに**自分の Mac のターミナル**で次を実行してもらう
  （キーは画面にもシェルの履歴にも残らない）。

  ```bash
  mkdir -p ~/Documents/kikiwake/.kikiwake
  read -rs "key?API キーを貼り付けて Enter: "; echo
  (umask 077; printf '%s\n' "$key" > ~/Documents/kikiwake/.kikiwake/aquavoice_api_key); unset key
  chmod 600 ~/Documents/kikiwake/.kikiwake/aquavoice_api_key
  ```

### そのほか

- 初回実行では声紋モデル（約 90 MB）をダウンロードするので時間がかかる（`<KIKIWAKE_HOME>/models/` に保存され、次回からは不要）。ネットワークが必要。
- 長い録音は数分以上かかることがある。コマンドのタイムアウトを長めにするか、バックグラウンドで実行する。
- ユーザーの音声ファイル、`.md`、`.session.json`、`.voices.png` は個人情報を含む。git にコミットしない。

## 2. 記録の流れ

### 2-1. 文字起こしする

- **Cowork モード**: 録音は記録フォルダ（`$HOME/mnt/kikiwake`）の中にあるものを使う（中のどこでもよい）。
  見つからないときは、ユーザーに録音を kikiwake フォルダに置いてもらう
  （チャットに添付されただけのファイルは Mac 側のシェルから見えないことがある）。
- **Mac モード**: 録音はどこにあってもよい。記録フォルダにするのは `--output-dir`（`<記録フォルダ>/records/<stem>`）だけ。
- どちらのモードでも、録音を**移動も削除もしない**。

```bash
kikiwake transcribe "<音声ファイルのパス>" --output-dir "<記録フォルダ>/records/<stem>" --json
```

- `<stem>` は音声のファイル名から拡張子を除いたもの（`20261026 102027-FA0D0F9E.m4a` なら `20261026 102027-FA0D0F9E`）。
  1 会議 1 フォルダ。**`--output-dir` は必ず付ける。**
- パスは引用符で囲む（空白を含むことがある）。標準出力には JSON だけが出る。進捗は標準エラーに出る。
- 言語が分かっているなら `--language ja`、固有名詞のヒントは `--prompt "..."`、人数が分かっているなら `--max-speakers N`（1〜8）。
- 同じ音声をもう一度処理すると保存済みの結果を再利用する（API は呼ばれず、料金もかからない）。
  API から取り直すときだけ `--force-api`（話者 ID が変わり得るので、それまでの名前付けは破棄される）。
- 保存済みの結果を再利用するときは `--language` / `--prompt` / `--max-speakers` は**反映されない**
  （標準エラーに「保存済みの文字起こしを再利用するため … は反映されません（反映するには --force-api）」と出る）。
  反映させたいときだけ `--force-api` を付ける（API の料金がもう一度かかるので、ユーザーに確認してから）。
- 声の可視化画像（`<stem>.voices.png`）は、ユーザーが見たいと言ったときだけ作る。**最初の `transcribe` に `--viz` を付けるのがいちばん確実。**
  あとから作るなら `kikiwake viz "<session_path>"`。
- **`元の音声ファイルが見つかりません` と出たとき**（`viz` のエラー、または画像がすでにある会議の `label` の `警告: 可視化をスキップしました: …`）:
  `session.json` に残っている録音のパスが、今の会話・環境のものと違う（Cowork の VM のパスは会話ごとに変わり、Mac とも違う）。
  あきらめずに、**同じ録音で同じ `transcribe` をもう一度実行する**
  （`kikiwake transcribe "<録音>" --output-dir "<記録フォルダ>/records/<stem>" --json`）。保存済みの結果を再利用するので
  API は呼ばれず、料金もかからない。録音のパスが今のものに書き換わり、確定済みの名前も引き継がれる。
  画像がすでにある会議は、この再実行で画像も作り直される。まだ無いときは、そのあと `kikiwake viz "<session_path>"` をもう一度実行する
  （再実行に `--viz` を付ければ、その場で作られる）。なお、`label` の警告のときも名前の保存は成功している（古い画像が残るだけ）。

**終了コードが 2 のとき**（API キー未設定、ffmpeg 未導入、引数不正、声紋 DB のモデル不一致）は、標準エラーの
`エラー: ...` をそのままユーザーに伝え、直し方を案内して止まる。勝手にリトライしない。
**終了コードが 1 のとき**（API 失敗、4 時間超の音声、モデルのダウンロード失敗など）も、エラー文をそのまま伝える
（ただし `元の音声ファイルが見つかりません` は、上のとおり `transcribe` の再実行で直せる）。
Cowork で 2 時間を超える録音が途中で失敗したら、VM のメモリ（約 3.8 GiB）が足りなかった可能性が高い。
**Mac の Claude Code で処理するよう案内する。**

### 2-2. 結果を読む

標準出力の JSON:

```json
{
  "markdown_path": "…/records/meeting/meeting.md",
  "session_path": "…/records/meeting/meeting.session.json",
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
その話者について 2-3 でユーザーに尋ねたあとは繰り返さなくてよい。

標準エラーに次の行が出ることがある。

- `警告: ...`（`transcribe` / `label` / `index` とも）: 処理は成功している。内容をユーザーに伝える。
  `警告: 可視化をスキップしました: ...` のときは画像だけ作れなかった（`viz_path` は `null`）。
- `別の kikiwake が DB を使用中のため待機しています…`: 別の実行が声紋 DB を使い終わるのを待っているだけ。
  そのまま待つ（中断しない）。

### 2-3. 名前を尋ねて反映する

`auto` / `confirmed` 以外の話者がいるときだけ、ユーザーに尋ねる。

- `examples` の**時刻と発言を引用**して、誰の声かを尋ねる（例:「00:00:45 の『はい、その件は…』と話している人は誰ですか？」）。
- `candidate` は「00:12:30 の『予算は来月までに…』は田中さんで合っていますか？」のように確認する。
- 答えを `label` で反映する。**KEY には必ず話者 ID（`spk_N`）を使う**（表示ラベルは `label` のたびに変わるため）。
  複数人分をまとめて 1 回で渡せる。

```bash
kikiwake label "<session_path>" spk_2=田中 spk_3=佐藤 --json
kikiwake label "<session_path>" "spk_4=田中 太郎" --json   # 空白を含む名前は KEY=NAME ごと引用符で囲む
```

- `candidate` が合っていると言われたら、`？` を除いた名前で同じように `label` する（`spk_2=田中`）。違うと言われたら別の名前を尋ねる。
- ユーザーが**わからないと言った話者には `label` しない**。そのまま `話者N` として残す。名前を推測で付けない。
- 同じ人が 2 つの話者 ID に分かれていると分かったら、両方に同じ名前を付ける（自動的に統合される）。
- 名前は「さん」などを付けず、ユーザーが言った表記で統一する。迷ったら先に `kikiwake speakers list --json` で登録済みの名前を確認する。
  名前は `？` / `?` で終われない。
- `label` は標準エラーに `警告: ...` を出すことがある（声紋不足で DB に登録しなかった、など）。そのままユーザーに伝える。
- `label` は `<stem>.md` を作り直す。`<stem>.voices.png` が既にあれば、それも作り直す。
  標準出力の JSON は `transcribe` と同じ形式なので、残りの未確定の話者を確認できる。
- 名前が声紋 DB に無い人は、このときに新しく登録される。

### 2-4. 要点 `summary.md` を書く

名前付けが終わったら（ユーザーが「わからない」と答えた話者は `話者N` のままでよい）、`<stem>.md` を読んで
`<記録フォルダ>/records/<stem>/summary.md` を書く。長いときは区切って読む。形式は固定（あとから探しやすくするため）。
`summary.md` が**既にあれば先に読み**、ユーザーが手で直した内容を勝手に消さない（直すのは、名前の付け直しなど必要な箇所だけ）:

```markdown
# <会議の話題>（YYYY-MM-DD HH:MM）

要点: <1 行。index.md に載る>

## 決まったこと
- …

## 宿題（誰が・何を・いつまでに）
- 佐藤: 見積もりを見直して共有する（来週の定例まで）

## 主な論点（発言者付き）
- 田中: …
```

- 見出しの日時は、`session_path` の JSON にある `recorded_at`（録音日時）から取る。
  `grep -m1 '"recorded_at"' "<session_path>"` で 1 行だけ読める（`2026-10-26T10:20:27+09:00` なら `2026-10-26 10:20`）。
  値が `null` や無いときは `created_at` を使う。
- `要点:` の行は 1 行だけ。`index.md` には**この行の先頭 60 文字**が載るので、いちばん大事なことを先頭に書く。
- **推測で事実を足さない。** 文字起こしに無いこと、聞き取れなかったことは書かない。
  決まったこと・宿題が無ければ、その見出しの下に `- なし` と書く。
- 名前は文字起こしの表示名のまま書く（`話者N` のままの人は `話者N`）。
- **あとから `label` で名前を付け直したら、`summary.md` の名前も合わせて直し**、2-5 の `index` をもう一度実行する。
- **Cowork モードでは、シェルのリダイレクトで書く。** ファイルを置き換える操作（`sed -i`、`mv`、一時ファイル経由の保存）は、
  削除が禁止されたフォルダで失敗することがある。直すときも内容を丸ごと書き直す。

  ```bash
  cat > "<記録フォルダ>/records/<stem>/summary.md" <<'EOF'
  # …
  EOF
  ```

### 2-5. 一覧 `index.md` を更新する

```bash
kikiwake index "<記録フォルダ>"
```

- `<記録フォルダ>/records/*/` の記録から `<記録フォルダ>/index.md`（日時・長さ・参加者・要点・文字起こしへのリンクの表）を作り直す。
  新しい会議が上に並ぶ。`summary.md` を書いた・直したとき、`label` で名前を変えたときは、そのたびに実行する。
- `--json` を付けると `{"index_path": …, "meetings": N, "skipped": […]}` が出る。`skipped` に記録があれば
  （読めない `session.json`）、その会議が一覧から抜けているとユーザーに伝える。

### 2-6. ユーザーに報告する

- 文字起こし（`markdown_path`）と `summary.md` の場所を伝える（可視化したときは `viz_path` も）。
- 誰が参加していたか（確定した名前と、まだ `話者N` のままの人）と、会議の要点を短く伝える。
- `warnings` があれば添える。`index.md` を更新したことも伝える。

## 3. 過去の会議を振り返る

「前回の会議で誰が何を言った？」「〇〇の件はどうなった？」のように聞かれたら、次の順で読んで答える。

1. `<記録フォルダ>/index.md` を読み、日時・参加者・要点から該当しそうな会議を選ぶ。
   `index.md` が無い、または一覧にない会議フォルダがあるときだけ、`kikiwake index "<記録フォルダ>"` で作り直す（この場合は準備のインストールが要る。読むだけなら不要）。
2. その会議の `<記録フォルダ>/records/<stem>/summary.md` を読む（一覧のリンク先と同じフォルダにある）。
3. 発言の細部が必要なときだけ、同じフォルダの `<stem>.md`（文字起こし）を読む。`summary.md` が無い会議は最初から文字起こしを読む。

- **答えには会議の日時を添える**（例:「2026-10-26 10:20 の会議では、佐藤さんが…」）。
- 該当しそうな会議が複数あるときは、日時つきで挙げて、どれのことか確かめるか、それぞれについて答える。
- 記録に書かれていないことは「記録にはありません」と答える。推測で補わない。
- 発言者の名前は文字起こしの表示名のまま（`話者N` の人は `話者N`）。

## 4. やってはいけないこと

- **記録フォルダ内のファイルを削除しない。** 削除の許可をユーザーに求めない・すすめない
  （Cowork では削除が既定で禁止されている。許可するとゴミ箱を経由せずに消える）。
  ユーザーが記録を消したいと言ったときは、Finder で消すよう案内する。録音も移動・削除しない。
- **API キーをチャットに出さない・貼らせない。** キーファイルの中身を表示しない。キーは `<KIKIWAKE_HOME>/aquavoice_api_key`。
- **Cowork と Mac で同時に kikiwake を動かさない。** 同じ記録フォルダの声紋 DB を同時に書くと、排他制御が効かないおそれがある。
  ユーザーが別の側で処理中と言っていたら、終わるまで待ってもらう。
- クラウド側のシェルで kikiwake を実行しない（0 参照）。
- ユーザーが頼んでいないのに `speakers rename` / `speakers delete` を行わない。

## 5. うまくいかないとき

| 状況 | 対応 |
|---|---|
| `$HOME/mnt/kikiwake` が無い | Mac 側シェルで確かめ直す。それでも無ければ「+ → Add folder で kikiwake フォルダを追加してください」と案内して止まる |
| インストールに失敗した | エラーをそのまま伝えて止まる（API は呼ばれていない）。接続できないホストがあれば、下の「許可するドメイン」を案内する |
| API キーが見つからない（終了コード 2） | 1 の「API キー」の手順で、ユーザーに Mac のターミナルでキーファイルを作ってもらう |
| 2 時間超の録音が失敗した | メモリ不足の可能性。Mac の Claude Code で処理するよう案内する |
| `元の音声ファイルが見つかりません`（`viz` など） | 2-1 の「`元の音声ファイルが見つかりません` と出たとき」のとおり、同じ `transcribe` をもう一度実行してから `viz` を実行する |
| 処理の途中で会話が切れた | 新しい会話で、同じ録音に同じ `transcribe` をもう一度行う。Avalon の結果は保存済みなので再課金されない |
| ファイルを開くのが遅い | 記録フォルダは iCloud で同期されていることがある。「Mac のストレージを最適化」でクラウド側だけにあるファイルは、読み込みに時間がかかる。待つ |
| `別の kikiwake が DB を使用中のため待機しています…` | 待つ。Cowork と Mac で同時に動かしていないか確認する |

**許可するドメイン**: Cowork の VM はネットワークが制限されている。インストールの途中で接続できないホストがあるというエラーが出たら、
Cowork の設定でネットワークの許可（許可するドメインの一覧）を確認し、そのドメインを追加してもらう
（設定画面の名前や場所は、バージョンによって違うことがある）。kikiwake が使うのは次のとおり
（エラーに出たホスト名も追加する）:
`api.aquavoice.com`（文字起こし）、`pypi.org` / `files.pythonhosted.org`（Python パッケージ）、`huggingface.co`（声紋モデル）、
`github.com`（kikiwake 本体）、`download.pytorch.org` / `download-r2.pytorch.org`（CPU 版 PyTorch。ファイルの実体は後者から配信される）。
Python 3.12 の取得で `githubusercontent.com` 系のホストが必要になることもある。

## 6. 声紋 DB の管理

```bash
kikiwake speakers list [--json]          # 登録済みの人、サンプル数、最終更新日時
kikiwake speakers rename "旧名" "新名"          # 新名は label の名前と同じ規則（空・末尾の ？ / ? は不可）
kikiwake speakers delete "名前"           # その人の声紋をすべて削除
```

`rename` と `delete` はユーザーに頼まれたときだけ行う。DB は `<KIKIWAKE_HOME>/db.json`
（Cowork・Mac とも記録フォルダの `.kikiwake/db.json`。`KIKIWAKE_HOME` を設定していなければ `~/.local/share/kikiwake/db.json`）。
`rename` / `delete` が変えるのは声紋 DB だけで、保存済みの文字起こし・`summary.md`・`index.md` の名前は変わらない
（過去の記録の名前も直したいときは、ユーザーに確認してから `summary.md` を書き直す）。
