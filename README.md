# kikiwake（聞き分け）

会議・打ち合わせの録音を、**話者名付き**で文字起こしする CLI です。

- 文字起こしと録音内の話者分離は [Aqua Voice の Avalon API](https://aquavoice.com/avalon-api/docs) に任せます。
- 話者ごとの声を手元で**声紋**（話者埋め込みベクトル）にし、ローカルの声紋 DB と照合します。
  一度名前を教えた人は、別の会議の録音でも自動で同じ名前が付きます。
- AI エージェント用のスキル（[`skill/SKILL.md`](skill/SKILL.md)）付き。AI チャットに音声ファイルを渡すと、
  文字起こし → 未確定の話者を質問 → 名前を登録 → 要点の記録、までを AI が進めます。
  Claude の **Cowork** でも使えます（[Cowork で使う](#cowork-で使う)）。会議ごとの記録が溜まるので、
  あとから「前回の会議で誰が何を言ったか」を AI に聞けます。

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

- macOS（Apple Silicon で動作確認。GPU 不要）。Linux（Cowork の仮想マシン）では CPU 版の PyTorch を使います
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

`AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY` の順に読みます。どれも無ければ、
`$KIKIWAKE_HOME/aquavoice_api_key`（1 行目にキーだけを書いたファイル）を読みます（Cowork ではこちらを使います）。
キーはログや出力に書き出しません。

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
出力先に**同じ名前の別の録音**の記録（`<stem>.session.json`）があるときは、上書きせずに終了コード 2 で止まります
（API は呼びません）。`--output-dir` で別のフォルダ（例: `"records/会議 (2)"`）を指定してください。

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
元の録音を使います。`session.json` に残る録音の場所に無いときは、記録フォルダ（`records/<名前>/` の 2 つ上）の中を
`records/` と `.kikiwake/` を除いて探し、同じ名前で中身（SHA-256）も同じ録音を使います（Cowork では会話ごとに仮想マシンの
パスが変わるため）。

### 5. 会議の一覧

```bash
kikiwake index ~/Documents/kikiwake
```

記録フォルダ（`records/<名前>/` に会議ごとの出力がある）を読み、会議の一覧 `index.md` を作ります
（詳しくは [Cowork で使う](#cowork-で使う) の「記録フォルダの中身」）。`--json` で
`{"index_path": …, "meetings": N, "skipped": […]}` を標準出力に出します。読めない `session.json` があっても
止まらず、その会議を飛ばして標準エラーに警告を出します。

## AI スキルとして使う

Claude Code などの AI エージェントにスキルとして読み込ませます。

```bash
ln -s "$PWD/skill" ~/.claude/skills/kikiwake
```

AI チャットに音声ファイルを渡して「文字起こしして」と頼むと、AI が `kikiwake` を実行し、
名前の分からない話者について「00:12:30 の『予算は来月までに…』と話している人は誰ですか？」のように尋ねます。
名前が決まると、AI が要点（`summary.md`）を書き、会議の一覧（`index.md`）を更新します。

## Cowork で使う

Claude の **Cowork** に会議の録音を渡すと、名前付きの文字起こしと要点が**記録フォルダ**に会議ごとに溜まります。
声紋 DB も同じフォルダに置くので、Cowork で名前を教えた人は、次の会議（Cowork でも Mac の Claude Code でも）で
自動で名前が付きます。

### しくみ

- Cowork には 2 種類のシェルがあります。Mac 上の Linux 仮想マシンで動く「Mac 側」だけが、会話に接続した
  Mac のフォルダを `$HOME/mnt/<フォルダ名>` に見られます。スキルは Mac 側のシェルで `kikiwake` を動かします
  （クラウド側のシェルからは Mac のフォルダが見えないので使いません）。
- 仮想マシンは**会話ごとに使い捨て**です。スキルが会話の最初に GitHub から `kikiwake` を入れます（数分かかります）。
  声紋 DB・声紋モデル・API キー・記録は仮想マシンではなく記録フォルダにあるので、会話が変わっても残ります。
- 接続したフォルダでは、ファイルの削除が既定で禁止されています。`kikiwake` とスキルは**記録フォルダのファイルを削除しません**
  （保存は、置き換えではなく直接の書き込みで行います）。削除の許可を求められても許可しないでください
  （許可すると、ゴミ箱を経由せずに消えます）。記録を消したいときは Finder で消します。

### 準備（1 回だけ）

以下は記録フォルダを `~/Documents/kikiwake` にする例です。**フォルダの名前は `kikiwake` にしてください**
（スキルは `$HOME/mnt/kikiwake` を探します）。

1. **記録フォルダを作ります。**

   ```bash
   mkdir -p ~/Documents/kikiwake/.kikiwake
   ```

2. **API キーのファイルを、Mac のターミナルで作ります。** チャットにキーを貼らないでください。
   次のコマンドは、キーを画面にもシェルの履歴にも残しません。

   ```bash
   read -rs "key?API キーを貼り付けて Enter: "; echo
   (umask 077; printf '%s\n' "$key" > ~/Documents/kikiwake/.kikiwake/aquavoice_api_key); unset key
   chmod 600 ~/Documents/kikiwake/.kikiwake/aquavoice_api_key
   ```

3. **Mac 側の `~/.zshrc` に `KIKIWAKE_HOME` を書きます。** Mac の `kikiwake` が Cowork と同じ声紋 DB を使うようになります。

   ```bash
   echo 'export KIKIWAKE_HOME="$HOME/Documents/kikiwake/.kikiwake"' >> ~/.zshrc
   source ~/.zshrc
   ```

   すでに `~/.local/share/kikiwake/` に声紋 DB（`db.json`）と声紋モデル（`models/`）があれば、
   中身を `~/Documents/kikiwake/.kikiwake/` に移してください。

4. **タイムゾーンを設定します。** Cowork の仮想マシンの時計は UTC のことがあるので、録音日時などを
   日本時間で記録するよう、記録フォルダの `config.toml` に書きます（Mac と Cowork の両方に効きます）。

   ```bash
   printf '\ntimezone = "Asia/Tokyo"\n' >> ~/Documents/kikiwake/.kikiwake/config.toml
   ```

5. **スキルの ZIP を作ります。** スクリプトは自分の場所からリポジトリのルートを探すので、どのディレクトリから実行してもかまいません
   （`dist/` はリポジトリのルートに作られます）。

   ```bash
   bash /path/to/kikiwake/scripts/make-skill-zip.sh   # リポジトリのルートなら: bash scripts/make-skill-zip.sh
   ```

   `dist/kikiwake-skill.zip` ができます（ZIP のルートは `kikiwake/` で、中に `SKILL.md` があります）。

6. **ZIP を Cowork にアップロードします。** Cowork の **Customize > Skills** から `dist/kikiwake-skill.zip` を選びます。
   `skill/SKILL.md` を直したら、ZIP を作り直してもう一度アップロードします。
   `kikiwake` 本体は会話ごとに GitHub の `master` から入るので、コードを直したときは GitHub に push してください。

### 使い方

1. 会議の録音を `~/Documents/kikiwake/`（中のどこでもよい）に置きます。`kikiwake` は録音を移動も削除もしません。
2. Cowork の会話で、**+ → Add folder** から `kikiwake` フォルダを接続します。
3. 「`20261026 102027-FA0D0F9E.m4a` を文字起こしして」のように頼みます。
   AI が `kikiwake` を入れて（初回の会話では数分）、文字起こしを進め、名前の分からない人を尋ね、
   名前を声紋 DB に登録し、要点を `summary.md` に書き、`index.md` を更新します。

録音の長さの目安は 2 時間までです。Cowork の仮想マシンのメモリは約 3.8 GiB なので、それを超える録音は途中で失敗する
ことがあります。その場合は Mac の Claude Code で処理してください。

### 記録フォルダの中身

```
kikiwake/
├─ 20261026 102027-FA0D0F9E.m4a        会議の録音（置いた場所のまま）
├─ records/
│   └─ 20261026 102027-FA0D0F9E/        会議ごとのフォルダ
│        ├─ 20261026 102027-FA0D0F9E.md            名前付きの文字起こし
│        ├─ 20261026 102027-FA0D0F9E.session.json  作業データ（Avalon の結果を含む）
│        ├─ .20261026 102027-FA0D0F9E.session.json.bak  作業データの控え（隠しファイル）
│        └─ summary.md                              AI が書く要点
├─ index.md                              会議の一覧
└─ .kikiwake/                            声紋 DB（db.json と控えの .db.json.bak）、声紋モデル、設定、API キー（aquavoice_api_key）
```

声紋 DB と作業データは、保存のたびに先に控え（`.<ファイル名>.bak`）へ全文を書いてから本体を書きます。
保存の途中で処理が止まって本体が壊れても、次に読むときに控えから読み込み（標準エラーに
`警告: … が壊れていたため控えから読み込みました` と出ます）、次の保存で本体も直ります。控えは消さないでください。

**`index.md`** は会議の一覧です（`kikiwake index` が作ります。新しい会議が上）。

```markdown
| 日時 | 長さ | 参加者 | 要点 | 文字起こし |
|---|---|---|---|---|
| 2026-10-26 10:20 | 41 分 | 田中・佐藤・ほか1名 | 見積もりの前提を見直す… | [開く](<records/20261026 102027-FA0D0F9E/20261026 102027-FA0D0F9E.md>) |
```

- 日時は録音日時です（ボイスメモの `YYYYMMDD HHMMSS-…` という名前ならその日時、そうでなければファイルの更新日時）。
  `config.toml` の `timezone` のタイムゾーンで記録します。
- 参加者は、名前が確定した人を並べ、それ以外の話者は「ほかN名」とまとめます。
- 要点は、同じ会議の `summary.md` の `要点:` の行（先頭 60 文字）です。

**`summary.md`** は AI が文字起こしを読んで書く、決まった形の要点です。推測で事実を足さないようにしてあります。

```markdown
# 見積もりの前提の見直し（2026-10-26 10:20）

要点: 見積もりの前提を見直し、来週の定例で再提示することにした

## 決まったこと
- …

## 宿題（誰が・何を・いつまでに）
- 佐藤: 見積もりを見直して共有する（来週の定例まで）

## 主な論点（発言者付き）
- 田中: …
```

名前を付け直したときは、AI が `summary.md` の名前も直して `index.md` を更新します。

### 過去の会議を聞く

「前回の会議で、誰が何を宿題にした？」のように聞くと、AI が `index.md` から該当する会議を探し、
その会議の `summary.md`（必要なら文字起こし）を読んで、会議の日時を添えて答えます。

### iCloud 同期の注意

この Mac の「書類」フォルダが iCloud Drive で同期されていると、記録フォルダ（`~/Documents/kikiwake`）の中身は
**すべて iCloud に上がります**。録音・文字起こし・要点・声紋 DB（`db.json`）・声紋モデル・API キーのファイルも含みます。
iCloud に上げたくないときは、同期の対象外の場所に記録フォルダを作ってください（その場合は `KIKIWAKE_HOME` などのパスも合わせます）。

また、「Mac のストレージを最適化」（Optimize Mac Storage）がオンだと、容量が足りないときにファイルが iCloud 側だけに置かれ、
開くのに時間がかかることがあります。

### 注意

- **Cowork と Mac で同時に `kikiwake` を動かさないでください。** 同じ声紋 DB を同時に書くと、排他制御が効かないおそれがあります。
- 処理の途中で会話が切れたとき: Avalon の結果は受け取った直後に保存されるので、それより**あと**に切れたのなら
  （`records/<名前>/` に `session.json` があれば）、同じ録音にもう一度頼むと保存済みの結果を使い、再課金なしでやり直せます。
  結果を**待っている間**に切れたときは、やり直すと API がもう一度呼ばれ、料金がもう一度かかることがあります。

### うまくいかないとき

- **「+ → Add folder で kikiwake フォルダを追加してください」と言われる**: 会話にフォルダが接続されていません。
  フォルダの名前が `kikiwake` であることも確かめてください。
- **API キーが見つからない（終了コード 2）**: 準備の 2 を実行してください。
- **`警告: 記録フォルダの声紋 DB（…）ではなく … を使っています` と出る**: `KIKIWAKE_HOME` が記録フォルダの `.kikiwake` を
  指していません（その回の名前は別の声紋 DB に登録されています）。Mac なら準備の 3 を確認してください。
- **インストールが失敗する**: Cowork の仮想マシンはネットワークが制限されています。エラーにブロックされたホストが出ていたら、
  Cowork の設定でネットワークの許可（許可するドメインの一覧）を確認し、そのドメインを追加してください
  （設定画面の名前や場所は、バージョンによって違うことがあります）。`kikiwake` が使うのは次のとおりです
  （API はまだ呼ばれていないので、失敗しても料金はかかりません）。

  | ドメイン | 用途 |
  |---|---|
  | `api.aquavoice.com` | 文字起こし（Avalon API） |
  | `pypi.org` / `files.pythonhosted.org` | Python パッケージ |
  | `huggingface.co` | 声紋モデル |
  | `github.com` | `kikiwake` 本体（`codeload.github.com` も使うことがあります） |
  | `download.pytorch.org` / `download-r2.pytorch.org` | CPU 版 PyTorch（ファイルの実体は後者から配信されます） |

  Python 3.12 の取得で `githubusercontent.com` 系のドメインが必要になることもあります。

## データの置き場所

| もの | 場所 |
|---|---|
| 声紋 DB | `$KIKIWAKE_HOME/db.json`（既定は `~/.local/share/kikiwake/db.json`） |
| 設定（任意） | `$KIKIWAKE_HOME/config.toml` |
| 声紋モデル | `$KIKIWAKE_HOME/models/` |
| API キーのファイル（任意） | `$KIKIWAKE_HOME/aquavoice_api_key` |
| 会議の記録 | 記録フォルダの `records/<録音の名前>/` と `index.md`（Cowork 向けの構成。[Cowork で使う](#cowork-で使う)） |

`KIKIWAKE_HOME` で場所を変えられます（Cowork と共有するときは `~/Documents/kikiwake/.kikiwake`）。
閾値などは `config.toml` で上書きできます（例: `auto_threshold = 0.65`）。
`timezone = "Asia/Tokyo"` のように書くと、録音日時・作成日時をそのタイムゾーン（IANA 名）で記録します
（既定はシステムの設定。Cowork の仮想マシンでは UTC のことがあります）。

文字起こし・作業データ・声紋 DB は個人情報を含むので、本人だけが読める権限で保存します。git にコミットしないでください。
（Cowork の接続フォルダでは `chmod` が効かず、権限は常に `600` に見えます。）

## 制限

- 1 回の録音で区別できる話者は 8 人まで（Avalon の上限）。録音は 4 時間まで（Cowork では 2 時間程度まで）。
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
