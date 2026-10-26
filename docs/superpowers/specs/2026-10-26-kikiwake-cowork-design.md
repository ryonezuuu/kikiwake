# kikiwake Cowork 対応と会議記録の永続化 設計書

- 作成日: 2026-10-26
- 状態: レビュー待ち
- 前提となる設計: `2026-10-26-speaker-transcribe-design.md`（kikiwake 本体。旧名 speaker-transcribe）

## 1. 目的と前提

### 目的

ユーザーは普段すべての作業を Claude の **Cowork** で行っている。Cowork に会議の録音を渡すだけで、
名前付きの文字起こしが**記録フォルダに会議ごとに溜まり**、声紋 DB が育って次の会議でも同じ人に同じ名前が付き、
あとから「前回の会議で誰が何を言ったか」を Cowork で振り返れるようにする。

### 成功基準

1. Cowork で録音を渡すと、`records/<録音名>/` に名前付きの文字起こし・作業データ・要点（`summary.md`）ができ、
   `index.md` の一覧に 1 行増える。
2. Cowork で名前を教えた人は、次の会議（Cowork でも Mac の Claude Code でも）で自動で名前が付く。
3. 過去の会議について質問すると、AI が `index.md` から該当する会議を見つけ、`summary.md` と文字起こしを根拠に答えられる。
4. 既存の Mac（Claude Code）での使い方は壊れない。

### 確認済みの Cowork 環境（2026-10-26、ユーザーの Max プランで実測）

- Cowork には 2 種類のシェルがある。
  - **クラウド側**（x86_64、Python 3.13）: Mac のフォルダは見えない。本設計では使わない。
  - **Mac 側**（`device_bash`。Mac 上の Linux 仮想マシン、aarch64、Ubuntu 22.04、Python 3.10、CPU 4 コア、メモリ 3.8 GiB、
    VM 内ディスク空き 9.2 GB）: 会話に接続した Mac のフォルダが FUSE で `$HOME/mnt/<フォルダ名>` にマウントされる。
    `HOME` は会話ごとに `/sessions/<id>` で、**VM は会話ごとに使い捨て**。
- Mac 側シェルには `ffmpeg` / `uv` / `pip3` があり、`api.aquavoice.com` / PyPI / Hugging Face / GitHub /
  `download.pytorch.org/whl/cpu` に接続できる。
- 接続フォルダ上では、読み書き・新規作成・`flock` ができる。**削除は既定で禁止**（許可すると、ゴミ箱を経由せずに消える）。
  `chmod` は無視され、常に `600` に見える。削除を許可していない会話で `os.replace` による上書きが通るかは**未確認**
  （ツールの説明では「既存ファイルを消して置き換える」操作は失敗するとされている）。
- 記録フォルダはユーザーの選択で `~/Documents/kikiwake`。**この Mac の書類フォルダは iCloud Drive で同期されている**ため、
  録音・文字起こし・声紋 DB・API キーのファイルも iCloud に上がる（ユーザー了承済み）。
  iCloud の容量の最適化でファイルがクラウド側だけに置かれると、読み込みに時間がかかることがある。

## 2. 全体構成

### 記録フォルダ（`~/Documents/kikiwake`。Cowork では `$HOME/mnt/kikiwake`）

```
kikiwake/
├─ （会議の録音。フォルダ内のどこに置いてもよい。kikiwake は録音を移動も削除もしない）
├─ records/
│   └─ <録音の stem>/
│        ├─ <stem>.md             名前付きの文字起こし（既存の出力）
│        ├─ <stem>.session.json   作業データ（既存。Avalon の結果も保存されている）
│        ├─ .<stem>.session.json.bak  作業データの控え（§3.2）
│        ├─ <stem>.voices.png     可視化（頼まれたときだけ。既存）
│        └─ summary.md            AI が書く要点（§4）
├─ index.md                       会議の一覧（`kikiwake index` が生成。§3.5）
└─ .kikiwake/                     KIKIWAKE_HOME: db.json, .db.json.bak（控え。§3.2）, db.json.lock, models/, config.toml, aquavoice_api_key
```

- 1 会議 1 フォルダ。`kikiwake transcribe "<録音>" --output-dir "<記録フォルダ>/records/<stem>"` で出力する。
- **kikiwake はこのフォルダ内のファイルを削除しない**（自分の一時ファイルの後片付けを除く。§3.2）。

### 声紋 DB は 1 つにまとめる

- Cowork（Mac 側 VM）も Mac の Claude Code も、`KIKIWAKE_HOME` を記録フォルダの `.kikiwake/` に向ける。
  - Cowork: `KIKIWAKE_HOME=$HOME/mnt/kikiwake/.kikiwake`（スキルが毎回設定）
  - Mac: `~/.zshrc` に `export KIKIWAKE_HOME="$HOME/Documents/kikiwake/.kikiwake"`（ユーザーが 1 回追加）
- Mac と Cowork の VM が同時に DB を書くと、`flock` が FUSE をまたいで効くかは未確認。**同時に実行しない**ことを
  スキルに書く（単一ユーザーなので実害は小さい）。

### Cowork での 1 回の流れ（スキルが行う。§5）

1. Mac 側シェルで `$HOME/mnt/kikiwake` の存在を確認する。無ければ「+ → Add folder で kikiwake フォルダを追加」と案内して止まる。
2. `kikiwake` が PATH に無ければ GitHub からインストールする（§3.1）。会話ごとに 1 回、数分かかる。
3. `KIKIWAKE_HOME` と `KIKIWAKE_INPLACE_WRITES=1` を設定する。API キーは §3.3 のファイルから kikiwake が読む。
4. `transcribe` → 名前を尋ねる → `label` → `summary.md` を書く → `kikiwake index`。

## 3. kikiwake 本体の変更

### 3.1 Linux では CPU 版 PyTorch を使う

- Linux（x86_64 / aarch64）では、`torch` と `torchaudio` を `https://download.pytorch.org/whl/cpu` から取得し、
  CUDA 版や `nvidia-*` パッケージを入れない。macOS は現状どおり PyPI の torch を使う。
- `pyproject.toml` に uv の index と sources（`sys_platform == 'linux'` の marker 付き）で指定し、`uv.lock` を更新する。
- Cowork でのインストールコマンドは、この指定が確実に効く形にする（`uv tool install "git+https://github.com/ryonezuuu/kikiwake"`
  で効かなければ、明示的な index 指定を付けたコマンドにする。どちらにするかは実装計画で uv の挙動を確かめて決める）。
- 受け入れ条件: Linux aarch64 / x86_64 向けに解決した依存関係に、`+cpu` の torch が選ばれ、`nvidia-` で始まるパッケージが含まれない。
  Python は uv が 3.12 を自動で用意する（VM の Python 3.10 は使わない）。

### 3.2 削除禁止のフォルダでも保存できるようにする

`speaker_db.atomic_write_text` を次のようにする。

1. 環境変数 `KIKIWAKE_INPLACE_WRITES=1` のときは、一時ファイルを使わず、対象ファイルを直接開いて書き込む
   （`open(path, "w")` → 書き込み → `flush` → `os.fsync`）。一時ファイルが残らない。Cowork ではスキルがこれを設定する。
2. それ以外は現状どおり、同じディレクトリの一時ファイル → `os.replace`。`os.replace` が `PermissionError` になったら、
   1. と同じ直接書き込みに切り替える。そのあと一時ファイルの削除を試み、失敗しても無視する
   （残った一時ファイルは `.` で始まる名前なので目立たない）。
- 直接書き込みは、書き込み中に強制終了されると中身が壊れ得る。作り直せないファイル、つまり声紋 DB（`db.json`。
  何か月も溜めていく）と作業データ（`<stem>.session.json`。支払い済みの Avalon の結果と確定した名前）は、
  `atomic_write_text(path, text, keep_backup=True)` で保存する。
  - 保存: まず同じフォルダの**控え** `.<ファイル名>.bak` に全文を直接書き込み（上の 1.・2. どちらの方式でも）、
    そのあとで本体を今までどおり書く。控えを書いている途中で止まっても本体は前の内容のまま、本体を書いている途中で
    止まっても控えに全文がある。
  - 読み込み: 本体があるのに読めない（UTF-8・JSON として不正、または形が不正）ときだけ控えから読み、
    標準エラーに `警告: <ファイル名> が壊れていたため控えから読み込みました` と出す。本体が無いとき、
    バージョン・埋め込みモデル・次元が違うときは控えを使わない。控えも読めなければ、本体のエラーを出す。
  - 次の保存で本体は全文が書き直されて直る。控えの名前は `*.session.json` に当たらないので `index` は拾わない。
    何も削除しない。
  - Markdown（`<stem>.md`）と `index.md` はいつでも作り直せるので、控えを作らない。
- 直接書き込みで新しく作るファイルは、一時ファイルと同じく本人だけが読める権限（`600`）で作る
  （Mac では控えも直接書き込みで作るため。接続フォルダでは `chmod` が効かず常に `600` に見える）。

### 3.3 API キーをファイルからも読む

- `avalon.get_api_key` の探索順を、環境変数 `AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY` →
  **`$KIKIWAKE_HOME/aquavoice_api_key` の 1 行目**（前後の空白を除く）にする。
- キーが見つからないときのエラー文に、ファイルの場所も含める。キーの値はどこにも出力しない（既存の制約）。
- ファイルはユーザーが Mac のターミナルで作る（§6）。チャットにキーを貼らせない。

### 3.4 録音日時を session に保存する

- `Session` に `recorded_at: str | None`（ISO 8601、UTC オフセット付き、秒まで）を追加する。古い session.json に無くても読める。
- `transcribe` のときに決める: 録音の stem が `YYYYMMDD HHMMSS` で始まれば（ボイスメモ形式。例 `20261016 125520-357FE498`）その日時、
  そうでなければ録音ファイルの更新日時。キャッシュを再利用する再実行でも、毎回この規則で決め直す。
  stem の日時がその時刻帯で表せない（暦の端など）ときも更新日時を使う（落ちない）。
- **時刻帯**: Cowork の VM の時計は UTC のことが多く、そのままでは日付がずれる。そこで `config.toml` に
  `timezone`（IANA 名。例 `timezone = "Asia/Tokyo"`。既定 `""` はシステムの時刻帯）を設ける。
  `load_settings` が `zoneinfo.ZoneInfo` で確かめ、不正なら設定名を含む `UsageError`。
  - ボイスメモの stem の日時はこの時刻帯の時刻として読み（`naive.replace(tzinfo=tz)`）、更新日時はこの時刻帯で表す
    （`datetime.fromtimestamp(mtime, tz)`）。session の `created_at` と声紋 DB の `created_at` も同じ時刻帯で書く。
  - `config.local_zone(settings)` がこの時刻帯を返す。空なら `None`（Python の「システムの時刻帯」。夏時間の規則も保たれる）。
  - Linux で時刻帯のデータが無くても動くよう、依存関係に `tzdata` を入れる。
  - 設定ファイルは `<KIKIWAKE_HOME>/config.toml`（記録フォルダの `.kikiwake/config.toml`）なので、Mac と Cowork で共通になる。

### 3.5 `kikiwake index ROOT [--json]`

- `ROOT/records/*/*.session.json` を読み、`ROOT/index.md` を書く（§3.2 の書き込み方式）。
- 並び順は `recorded_at`（無いか、日時として読めなければ `created_at`）の新しい順。
- 形式:

  ```markdown
  # 会議の記録

  | 日時 | 長さ | 参加者 | 要点 | 文字起こし |
  |---|---|---|---|---|
  | 2026-10-26 10:20 | 41 分 | 田中・佐藤・ほか1名 | 時刻同期の前提を見直す… | [開く](<records/20261026 102027-FA0D0F9E/20261026 102027-FA0D0F9E.md>) |
  ```

  - 日時: `YYYY-MM-DD HH:MM`。長さ: `round(duration / 60)` 分。
  - 参加者: `auto` / `confirmed` の表示名を登場順に重複なく「・」でつなぎ、それ以外の話者は「ほかN名」とまとめる。
    名前のある人がいなければ「N名（名前未確定）」。
  - 要点: 同じフォルダの `summary.md` の `要点:` で始まる最初の行の本文。60 文字を超えたら 60 文字 + `…`。無ければ `（要点なし）`。
    表の区切りを壊さないよう `|` は `／` に置き換える。
  - リンク: ROOT からの相対パスを `<...>` で囲む（空白を含むパスのため）。パスの中の `|` `<` `>` は、表やリンクを
    壊さないよう `%7C` `%3C` `%3E` にする。
- 読めない session.json があっても止まらず、その会議を飛ばして標準エラーに警告を出す。理由は問わない
  （形式の不正、日時として読めない `created_at`、長さが無限大など。1 件の不具合で `index` 全体を止めない）。
- `--json` のときは `{"index_path": ..., "meetings": N, "skipped": [...]}` を出す。

### 3.6 同じ名前の別の録音の記録を上書きしない

- `run_transcribe` は、前回の session を読んだ直後（API を呼ぶ前）に確かめる。出力先の `<stem>.session.json` があって読め、
  その `audio_sha256` がこの録音と違えば（同じ名前の別の会議）、`UsageError`
  「`<出力先> には別の録音（同じ名前）の記録があります。--output-dir で別のフォルダを指定してください（例: "<出力先> (2)"）`」で止まる。
  `--force-api` でも上書きしない。読めない session.json は従来どおり警告して無視する。
- 同じ場所の録音の中身が変わった場合も同じ扱いになる（以前は API を呼び直して上書きしていた）。
- スキルはこのエラーのとき `records/<stem> (2)/`（使われていれば `(3)`…）でやり直し、ユーザーに伝える。

### 3.7 可視化のときに録音を探す

- session の `audio_path` は、`transcribe` したときの録音の絶対パス（Cowork ではその会話の VM のパス）。次の会話や Mac では
  そこに無いので、`viz`（と、画像がある会議の `label` での描き直し）は録音を探す。
- `audio_path` に録音が無く、session が `<root>/records/<x>/` にあれば、`<root>` の下（`<root>/records` と `<root>/.kikiwake` を除く）を
  名前順にたどり、同じファイル名で SHA-256 が `audio_sha256` と同じ最初のファイルを使う。無ければ従来どおり
  `元の音声ファイルが見つかりません: <audio_path>`（終了コード 1）。session の形式・中身は変えない。
- スキルは、見つからないときは録音を記録フォルダに置いてもらう。最後の手段として、キャッシュを使う `transcribe` の再実行
  （`audio_path` が書き換わる）。ただし未確定の話者は照合し直されるので、名前の付き方が変わることがある。

## 4. summary.md（AI が書く）

決まった形にして、あとから AI が探しやすくする。

```markdown
# <会議の話題>（YYYY-MM-DD HH:MM）

要点: <1 行。index.md に載る>

## 決まったこと
- …

## 宿題（誰が・何を・いつまでに）
- 佐藤: 時刻同期を前提にしない遅延測定を検討する（次回の進捗報告まで）

## 主な論点（発言者付き）
- 田中: …
```

- 文字起こし（`<stem>.md`）を読んで書く。推測で事実を足さない。名前は文字起こしの表示名のまま書く。
- 名前を付け直したら（`label`）、`summary.md` の名前も合わせて直す。

## 5. スキル（`skill/SKILL.md`）の変更

- **環境の判定**: Mac 側シェル（`device_bash`）で `$HOME/mnt/kikiwake` があれば **Cowork モード**、
  それ以外で `kikiwake` が PATH にあれば **Mac モード**。クラウド側シェルでは実行しない。
- **Cowork モードの準備**（会話ごとに 1 回）: §3.1 のインストール、`KIKIWAKE_HOME` と `KIKIWAKE_INPLACE_WRITES=1` の設定。
  インストールの進行中は、数分かかることをユーザーに伝える。
- **記録の流れ**: `transcribe --output-dir "<記録フォルダ>/records/<stem>" --json` → 未確定の話者を尋ねる（既存の手順）→
  `label` → `summary.md` を書く → `kikiwake index "<記録フォルダ>"`。
- **振り返り**: 過去の会議について聞かれたら、まず `index.md` を読み、該当する会議の `summary.md`、必要なら文字起こしを読んで答える。
  答えには会議の日時を添える。
- **削除しない**: 記録フォルダ内のファイルを削除しない。削除の許可を求めない。
- **長い録音**: Cowork の VM はメモリが 3.8 GiB なので、2 時間を超える録音で失敗したら Mac の Claude Code で処理するよう案内する。
- **同時実行しない**: 同じ記録フォルダに対して、Cowork と Mac で同時に kikiwake を動かさない。
- frontmatter の `description` は 200 文字以内に保つ（Cowork のスキルの制限）。
- **配布**: `skill/` フォルダを ZIP にし、Cowork の Customize > Skills からアップロードする手順を README に書く。
  Mac の Claude Code では今のシンボリックリンクのまま。

## 6. 移行（1 回だけ。実装の最後に行う）

1. `~/.local/share/kikiwake/` の中身（`db.json`（登録済み 4 人）、`models/`）を `~/Documents/kikiwake/.kikiwake/` に移す。
2. ユーザーが `~/.zshrc` に `export KIKIWAKE_HOME="$HOME/Documents/kikiwake/.kikiwake"` を追加し、
   API キーを `~/Documents/kikiwake/.kikiwake/aquavoice_api_key` に書く（`chmod 600`）。
   `~/Documents/kikiwake/.kikiwake/config.toml` に `timezone = "Asia/Tokyo"` を書く（§3.4）。
3. 既存のサンプル会議（リポジトリの `sample/audio/`）を記録フォルダに移す: 録音を `~/Documents/kikiwake/` にコピーし、
   その `session.json` を `records/<stem>/` にコピーしてから `transcribe --output-dir records/<stem>` を実行する
   （SHA-256 が同じなので保存済みの結果を再利用し、API は呼ばない。確定済みの名前も付け直される）。
4. `kikiwake index ~/Documents/kikiwake` で `index.md` を作る。
5. ユーザーが確認用に作った `probe.txt` / `probe.lock` は、ユーザーの了承を得てから消す。

## 7. エラー処理

| 状況 | 動作 |
|---|---|
| 記録フォルダが Cowork に接続されていない | 「+ → Add folder で kikiwake フォルダを追加してください」と案内して止まる |
| インストールに失敗（ネットワークなど） | エラー内容を伝えて止まる（API は呼ばれていないので料金なし） |
| `os.replace` が削除禁止で拒否された | §3.2 のとおり直接書き込みに切り替える |
| API キーが環境変数にもファイルにも無い | 終了コード 2。ファイルの場所を含むエラー文（既存の挙動の拡張） |
| 処理の途中で会話が切れた | Avalon の結果は保存済みなので、やり直しても再課金されない（既存） |
| VM のメモリ不足（長い録音） | Mac の Claude Code で処理するよう案内する |
| `index` で読めない session.json | その会議を飛ばして警告。index.md は作る |
| 出力先に同じ名前の別の録音の記録がある | 終了コード 2（API は呼ばない）。スキルは `records/<stem> (2)/` でやり直す（§3.6） |

## 8. テスト方針

- 単体テスト（ネット接続なし）:
  - `atomic_write_text`: `KIKIWAKE_INPLACE_WRITES=1` で一時ファイルが作られないこと、`os.replace` が `PermissionError` のときに
    直接書き込みへ切り替わり中身が正しいこと。
  - 控え（§3.2）: 壊れた本体 + 正しい控え（声紋 DB・session とも）で控えから読めて次の保存で直ること、
    壊れた控え + 正しい本体では本体を使うこと、両方壊れていれば本体のエラー、本体が無いときは控えを読まないこと、
    直接書き込みのときに一時ファイルを作らず何も削除しないこと。
  - `get_api_key`: 環境変数が優先されること、ファイルから読めること、空白を除くこと、無いときのエラー文にファイルの場所が入りキーが入らないこと。
  - `recorded_at`: ボイスメモ形式の stem、それ以外（更新日時）、古い session.json（フィールド無し）。
  - `index`: 期待どおりの Markdown（並び順、参加者の「ほかN名」、要点の取り出しと 60 文字切り詰め、`|` の置き換え、`<...>` リンク）、
    読めない session.json を飛ばすこと、`--json` の出力。
- 依存関係: `uv pip compile`（または `uv lock` の結果）を Linux aarch64 / x86_64 向けに解決し、`+cpu` の torch が選ばれ
  `nvidia-*` が含まれないことを確かめる。
- Cowork での実地確認（手動）: ユーザーが Cowork で記録フォルダの `20261026 102027-FA0D0F9E.m4a` を処理し、
  インストール → 文字起こし → 名前付け → `summary.md` → `index.md` まで通ること、登録済みの 4 人に自動で名前が付くことを確かめる。

## 9. 対象外

クラウド側シェルでの処理、複数ユーザーでの記録フォルダ共有、録音の自動取り込み（フォルダ監視・定期実行）、
全文検索の専用機能（AI が index.md と文字起こしを読んで探す）、記録フォルダの暗号化。
