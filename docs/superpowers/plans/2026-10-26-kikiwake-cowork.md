# kikiwake Cowork Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** kikiwake を Cowork の Mac 側シェル（Linux VM）で動かし、会議ごとの記録を記録フォルダに溜めて `index.md` から振り返れるようにする。

**Architecture:** 本体の変更は小さく 5 つ（Linux で CPU 版 torch、削除禁止フォルダ向けの直接書き込み、API キーのファイル読み込み、`recorded_at`、`kikiwake index`）。要点の文章（`summary.md`）と Cowork の手順はスキル（SKILL.md）が担う。Cowork ではリポジトリを clone して `uv sync --frozen` で入れる（`uv.lock` の CPU 版指定がそのまま効くため）。

**Tech Stack:** 既存（Python 3.12 / uv / pytest）。追加の依存は無し。

**Spec:** `docs/superpowers/specs/2026-10-26-kikiwake-cowork-design.md`（本体の設計は `docs/superpowers/specs/2026-10-26-speaker-transcribe-design.md`。旧名 speaker-transcribe = 現 kikiwake）

## Global Constraints

- パッケージ `kikiwake`（`src/kikiwake/`）、コマンド `kikiwake`、データディレクトリは `KIKIWAKE_HOME`（既定 `~/.local/share/kikiwake`）。
- API キーの値をログ・出力・例外メッセージ・テストの出力に出さない。
- 終了コード: `0` 成功 / `1` `AppError` / `2` `UsageError`。`--json` 時の標準出力は JSON のみ。
- 記録フォルダ内のファイルを kikiwake が削除しない（自分の一時ファイルの後片付けを除く）。
- テスト: `uv run pytest`（既定で `-m "not model and not e2e"`）。e2e は実行しない。出力は警告なし。
- 既存の 285 テストを壊さない。既存のインターフェース（`run_transcribe`, `atomic_write_text(path, text)`, CLI のサブコマンドとオプション）は互換を保つ。
- repo の `.venv` は macOS の hidden フラグのため `uv run kikiwake` が動かない（既知）。CLI の確認は `uv run pytest` 経由か、`uv tool install --reinstall .` 後の `kikiwake` で行う。
- コミットの作成者メールは repo-local 設定の `ryonezuuu@users.noreply.github.com` のまま（変えない）。

## Review Focus

1. 古い session.json（`recorded_at` 無し）と、F2 の途中保存（`speakers` が空で `turns` がある）が `records/` に混ざっている → `index` は落ちずに、前者は `created_at` で並べ、後者は参加者欄を「（処理未完了）」にする。テストは Task 5。
2. `summary.md` の要点行が全角コロン（`要点：`）で書かれている → 半角と同じに扱う。テストは Task 5。
3. `KIKIWAKE_INPLACE_WRITES` が `1` 以外（`true` / `yes` / `0` / 空）→ `1`・`true`・`yes`（大文字小文字無視）だけを有効とみなす。テストは Task 2。
4. API キーのファイルが改行付き・CRLF・空・存在しない → 前後の空白を除いて使い、空や無しなら次の候補へ（最終的に UsageError）。テストは Task 3。
5. 録音名に空白・日本語がある → `index.md` のリンクが `<...>` 形式で正しく、`recorded_at` の解析も壊れない。テストは Task 4・5。

---

## File Structure

```
pyproject.toml / uv.lock          Task 1: Linux 向け CPU 版 torch
src/kikiwake/speaker_db.py        Task 2: atomic_write_text の直接書き込み
src/kikiwake/avalon.py            Task 3: get_api_key のファイル読み込み
src/kikiwake/cli.py               Task 3（キーファイルの場所を渡す）, Task 5（index サブコマンド）
src/kikiwake/session.py           Task 4: recorded_at
src/kikiwake/pipeline.py          Task 4: recorded_at を決める
src/kikiwake/archive.py           Task 5: index.md の生成（新規）
skill/SKILL.md, README.md         Task 6
scripts/make-skill-zip.sh         Task 6（新規）
tests/test_lock.py                Task 1（新規）
tests/test_archive.py             Task 5（新規）
```

---

### Task 1: Linux では CPU 版 PyTorch を使う

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Test: `tests/test_lock.py`

**Interfaces:**
- Produces: `uv.lock` の Linux（`sys_platform == 'linux'`）向け `torch` / `torchaudio` が `https://download.pytorch.org/whl/cpu` 由来の `+cpu` 版になり、lock 全体に `nvidia-` で始まるパッケージが無い。macOS 向けは PyPI の torch のまま。

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_lock.py`（`tomllib` で `uv.lock` を読む。ネット接続なし）

```python
def test_linux_torch_comes_from_the_cpu_index():
    pkgs = [p for p in lock()["package"] if p["name"] in ("torch", "torchaudio")]
    linux = [p for p in pkgs if "download.pytorch.org/whl/cpu" in p["source"].get("registry", "")]
    assert {p["name"] for p in linux} == {"torch", "torchaudio"}
    assert all("+cpu" in p["version"] for p in linux)

def test_no_cuda_packages_in_the_lock():
    assert not [p["name"] for p in lock()["package"] if p["name"].startswith("nvidia-")]

def test_macos_torch_still_from_pypi():
    assert any(p["name"] == "torch" and "pypi.org" in p["source"].get("registry", "") for p in lock()["package"])
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_lock.py -v` → FAIL
- [ ] **Step 3: `pyproject.toml` に追加して `uv lock`**

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchaudio = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```
`+cpu` の torchaudio が torch と同じ系列で揃わない場合は、両方が解決できる版に下限を調整し、理由を報告に書く。

- [ ] **Step 4: 確認** — `uv run pytest tests/test_lock.py -v` → 3 passed。さらに `uv export --frozen --no-dev --no-hashes | grep -E '^(torch|torchaudio)'` で Linux 行が `+cpu` であることを報告に貼る。`uv sync` 後に全テスト `uv run pytest` → 全件 pass（macOS の torch が変わっていないこと）。
- [ ] **Step 5: コミット** — `git add -A && git commit -m "build: use CPU-only PyTorch on Linux"`

---

### Task 2: 削除禁止のフォルダでも保存できる `atomic_write_text`

**Files:**
- Modify: `src/kikiwake/speaker_db.py`
- Test: `tests/test_speaker_db.py`

**Interfaces:**
- Produces: `atomic_write_text(path: Path, text: str) -> None`（シグネチャ不変）。
  - `inplace_writes_enabled(env: Mapping[str, str] = os.environ) -> bool` — `KIKIWAKE_INPLACE_WRITES` が `1` / `true` / `yes`（大文字小文字・前後空白無視）なら True。
  - 有効時: 一時ファイルを作らず `open(path, "w", encoding="utf-8")` → write → flush → `os.fsync`。
  - 無効時: 現状どおり一時ファイル → `os.replace`。`os.replace` が `PermissionError` なら直接書き込みに切り替え、そのあと一時ファイルの `unlink` を試みて失敗は無視。それ以外の例外は従来どおり一時ファイルを消して再送出。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_inplace_mode_creates_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("KIKIWAKE_INPLACE_WRITES", "1")
    created = []
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *a, **k: created.append(1) or pytest.fail("temp"))
    p = tmp_path / "db.json"; p.write_text("old"); atomic_write_text(p, "new")
    assert p.read_text() == "new" and created == []

@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), (" yes ", True),
                                            ("0", False), ("", False), ("no", False)])
def test_inplace_flag_values(value, expected):
    assert inplace_writes_enabled({"KIKIWAKE_INPLACE_WRITES": value}) is expected
    assert inplace_writes_enabled({}) is False

def test_replace_denied_falls_back_to_writing_in_place(tmp_path, monkeypatch):
    monkeypatch.delenv("KIKIWAKE_INPLACE_WRITES", raising=False)
    p = tmp_path / "db.json"; p.write_text("old")
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError("denied")))
    atomic_write_text(p, "new")
    assert p.read_text() == "new"

def test_replace_denied_and_unlink_denied_still_succeeds(tmp_path, monkeypatch):
    # os.replace と Path.unlink の両方が PermissionError でも、内容は書けて例外にならない
    ...
    assert p.read_text() == "new"

def test_other_errors_still_clean_up_and_raise(tmp_path, monkeypatch):
    # os.replace が OSError(EIO) → 一時ファイルは消え、例外が伝わる（既存の挙動）
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_speaker_db.py -v` → 新規が FAIL
- [ ] **Step 3: 実装**
- [ ] **Step 4: 確認** — `uv run pytest tests/test_speaker_db.py -v` → pass、`uv run pytest` → 全件 pass
- [ ] **Step 5: コミット** — `git commit -am "feat: write in place where replacing files is not allowed"`

---

### Task 3: API キーをファイルからも読む

**Files:**
- Modify: `src/kikiwake/avalon.py`, `src/kikiwake/cli.py`
- Test: `tests/test_avalon.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `API_KEY_FILENAME = "aquavoice_api_key"`
  - `get_api_key(env: Mapping[str, str] = os.environ, key_file: Path | None = None) -> str` — 環境変数 3 つ（既存の順）→ `key_file` の内容（`strip()`。空なら無視。読めない・無いなら無視）。見つからなければ `UsageError`。エラー文は既存の文に `" または次のファイルに API キーを 1 行で書いてください: {key_file}"` を足す（`key_file` が None なら足さない）。
  - `cli.make_transcriber(settings, key_file: Path | None = None)` — 最初の呼び出しで `get_api_key(key_file=key_file)`。`_cmd_transcribe` は `data_home() / API_KEY_FILENAME` を渡す。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_env_wins_over_file(tmp_path):
    f = tmp_path / "aquavoice_api_key"; f.write_text("from-file\n")
    assert get_api_key({"AQUAVOICE_API": "from-env"}, key_file=f) == "from-env"

@pytest.mark.parametrize("content", ["k-123\n", "  k-123  ", "k-123\r\n"])
def test_key_from_file_is_stripped(tmp_path, content):
    f = tmp_path / "aquavoice_api_key"; f.write_text(content)
    assert get_api_key({}, key_file=f) == "k-123"

@pytest.mark.parametrize("make", [lambda f: None, lambda f: f.write_text(""), lambda f: f.write_text(" \n")])
def test_missing_or_empty_file_is_a_usage_error_naming_the_file(tmp_path, make):
    f = tmp_path / "aquavoice_api_key"; make(f)
    with pytest.raises(UsageError) as e:
        get_api_key({}, key_file=f)
    assert str(f) in str(e.value) and "AQUAVOICE_API" in str(e.value)

def test_file_key_never_in_messages(...)  # 4xx 応答の本文にファイル由来のキーが入っても伏せられる（既存の _redact が効く）

# tests/test_cli.py
def test_transcribe_reads_the_key_file_in_kikiwake_home(cli_env, monkeypatch):
    # 環境変数を全部消し、KIKIWAKE_HOME/aquavoice_api_key にキーを置く。real make_transcriber を使い、
    # AvalonClient をスタブにして、渡された api_key がファイルの値であることを確認
```

- [ ] **Step 2: 失敗を確認** — `uv run pytest tests/test_avalon.py tests/test_cli.py -v`
- [ ] **Step 3: 実装**
- [ ] **Step 4: 確認** — 対象ファイル pass、`uv run pytest` 全件 pass
- [ ] **Step 5: コミット** — `git commit -am "feat: read the Avalon API key from KIKIWAKE_HOME/aquavoice_api_key"`

---

### Task 4: 録音日時（`recorded_at`）

**Files:**
- Modify: `src/kikiwake/session.py`, `src/kikiwake/pipeline.py`
- Test: `tests/test_session.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `Session.recorded_at: str | None = None`（`version` の前に置く。既定値 None なので既存の `Session(...)` 呼び出しは壊れない）。`save_session` は常に書き、`load_session` は無ければ None。
  - `pipeline.recorded_at_for(audio: Path) -> str` — stem が正規表現 `^(\d{8}) (\d{6})` に一致し、その日時が実在すれば（`datetime.strptime(..., "%Y%m%d %H%M%S")`）ローカルタイムゾーン付き ISO 8601（秒まで）。それ以外は `audio.stat().st_mtime` のローカル時刻 ISO 8601。
  - `run_transcribe` は、途中保存（F2）と最終保存の両方の `Session` に `recorded_at=recorded_at_for(audio)` を入れる。キャッシュ再利用時も毎回決め直す。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_voice_memo_stem_gives_recorded_at(tmp_path):
    p = tmp_path / "20261016 125520-357FE498.m4a"; p.write_bytes(b"x")
    assert recorded_at_for(p).startswith("2026-10-16T12:55:20")

@pytest.mark.parametrize("name", ["meeting.m4a", "20261399 999999-x.m4a", "会議 10.26.m4a"])
def test_other_stems_use_mtime(tmp_path, name):
    p = tmp_path / name; p.write_bytes(b"x"); os.utime(p, (1_790_000_000, 1_790_000_000))
    assert recorded_at_for(p) == datetime.fromtimestamp(1_790_000_000).astimezone().isoformat(timespec="seconds")

def test_session_without_recorded_at_still_loads(tmp_path):   # 既存形式の JSON から recorded_at キーを消して load → None
def test_recorded_at_roundtrips(tmp_path):
def test_pipeline_stores_recorded_at(env):                    # 最終 session と F2 の途中保存の両方
```

- [ ] **Step 2: 失敗を確認** → **Step 3: 実装** → **Step 4: 確認**（対象 + 全件 pass）
- [ ] **Step 5: コミット** — `git commit -am "feat: store when each meeting was recorded"`

---

### Task 5: `kikiwake index`

**Files:**
- Create: `src/kikiwake/archive.py`, `tests/test_archive.py`
- Modify: `src/kikiwake/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_session`, `Session.recorded_at` / `created_at` / `speakers` / `transcription`, `SpeakerState.display_label` / `status`, `atomic_write_text`
- Produces:
  - `@dataclass(frozen=True) MeetingRow(when: str, minutes: int, participants: str, summary: str, link: str)` — `when` は `YYYY-MM-DD HH:MM`。
  - `summary_line(summary_md: Path) -> str` — `要点:` または `要点：` で始まる最初の行（前の空白可）の本文を strip。60 文字を超えたら 60 文字 + `…`。`|` → `／`。無い・ファイルが無いなら `（要点なし）`。
  - `participants_text(session: Session) -> str` — `auto` / `confirmed` の表示名を `session.speakers` の順に重複なく「・」で連結し、それ以外の話者が k 人いれば `・ほか{k}名`（名前のある人が 0 人なら `{k}名（名前未確定）`）。`speakers` が空で `transcription.turns` が空でなければ `（処理未完了）`、どちらも空なら `0名`。
  - `build_index(root: Path) -> tuple[str, int, list[str]]` — `root/records/*/*.session.json` を読み（読めないものは skipped に `root` からの相対パスを入れて飛ばす）、`recorded_at or created_at` の新しい順に並べた Markdown 全文、会議数、skipped を返す。リンクは md ファイル（`session_paths(Path(s.audio_path), <session のディレクトリ>)[0]`）の `root` からの相対パスを `[開く](<...>)`。
  - `write_index(root: Path) -> tuple[Path, int, list[str]]` — `build_index` → `atomic_write_text(root / "index.md", text)`。skipped ごとに `警告: 読めない記録を飛ばしました: {rel}` を標準エラーへ。
  - CLI: `kikiwake index ROOT [--json]` — 成功時、人間向けは index.md のパスを 1 行、`--json` は `{"index_path": str, "meetings": int, "skipped": [str]}`。`ROOT` が存在しないディレクトリなら `UsageError`。

Markdown（`build_index` の出力。末尾は改行 1 つ。会議 0 件なら表のヘッダーの後に `| （まだ記録がありません） | | | | |`）:

```markdown
# 会議の記録

| 日時 | 長さ | 参加者 | 要点 | 文字起こし |
|---|---|---|---|---|
| 2026-10-26 10:20 | 41 分 | 田中・佐藤・ほか1名 | 時刻同期の前提を見直す | [開く](<records/20261026 102027-FA0D0F9E/20261026 102027-FA0D0F9E.md>) |
```

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_archive.py`（`session_from_words` などの conftest ヘルパで session を作り、`save_session` で `tmp_path/records/<stem>/` に置く）

```python
def test_index_golden(tmp_path):         # 2 会議。新しい順、日時・分・参加者・要点・<> リンクが上の形式どおり（文字列完全一致）
def test_summary_line_variants(tmp_path):  # 半角/全角コロン、前の空白、60 文字超の切り詰め、| の置き換え、無い場合
def test_participants_variants():          # ほかN名、全員未確定、処理未完了（speakers 空・turns あり）、0名
def test_old_session_without_recorded_at_sorts_by_created_at(tmp_path):
def test_unreadable_session_is_skipped_with_warning(tmp_path, capsys):
def test_empty_root_says_no_records(tmp_path):
def test_index_uses_inplace_writes_when_enabled(tmp_path, monkeypatch):   # KIKIWAKE_INPLACE_WRITES=1 で一時ファイルを作らない

# tests/test_cli.py
def test_index_command_json(cli_env, capsys):   # transcribe --output-dir root/records/meeting → index root --json
def test_index_missing_root_exit_2(capsys):
```

- [ ] **Step 2: 失敗を確認** → **Step 3: 実装** → **Step 4: 確認**（対象 + 全件 pass）
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: kikiwake index builds the meeting archive index"`

---

### Task 6: スキル・README・スキル ZIP

**Files:**
- Modify: `skill/SKILL.md`, `README.md`
- Create: `scripts/make-skill-zip.sh`

**Interfaces:**
- Consumes: Task 1–5 の CLI（`transcribe --output-dir`, `label`, `index`, `KIKIWAKE_HOME`, `KIKIWAKE_INPLACE_WRITES`, キーファイル）

- [ ] **Step 1: `skill/SKILL.md` を書き直す** — spec §5 のとおり。必ず含める:
  - **環境の判定**: Cowork の Mac 側シェル（`device_bash`）で `[ -d "$HOME/mnt/kikiwake" ]` → Cowork モード。クラウド側シェルでは実行しない。それ以外で `kikiwake` が PATH にあれば Mac モード（記録フォルダは `~/Documents/kikiwake`）。
  - **Cowork モードの準備**（会話ごとに 1 回。`command -v kikiwake` があれば省略）:
    ```bash
    export KIKIWAKE_HOME="$HOME/mnt/kikiwake/.kikiwake" KIKIWAKE_INPLACE_WRITES=1
    if ! command -v kikiwake >/dev/null; then
      git clone --depth 1 https://github.com/ryonezuuu/kikiwake "$HOME/kikiwake-src" \
        || { mkdir -p "$HOME/kikiwake-src" && curl -sL https://codeload.github.com/ryonezuuu/kikiwake/tar.gz/refs/heads/master | tar xz --strip-components=1 -C "$HOME/kikiwake-src"; }
      (cd "$HOME/kikiwake-src" && uv sync --frozen --no-dev --python 3.12)
      export PATH="$HOME/kikiwake-src/.venv/bin:$PATH"
    fi
    ```
    （シェルの状態がコマンド間で引き継がれない場合は、毎回この `export` 行を先頭に付ける、と書く。）
  - **記録の流れ**: `kikiwake transcribe "<録音>" --output-dir "<記録フォルダ>/records/<stem>" --json` → 既存の質問・`label` 手順 → spec §4 の形式で `records/<stem>/summary.md` を書く → `kikiwake index "<記録フォルダ>"`。
  - **振り返り**: `index.md` → 該当会議の `summary.md` → 必要なら文字起こし。答えに会議の日時を添える。
  - **禁止事項**: 記録フォルダ内のファイルを削除しない／削除の許可を求めない／API キーをチャットに出さない・貼らせない。キーは `<KIKIWAKE_HOME>/aquavoice_api_key`。
  - **注意**: Cowork と Mac で同時に実行しない。2 時間超で失敗したら Mac の Claude Code へ。
  - frontmatter の `description` は 200 文字以内（現行の文を保つ。`python3 -c` で長さを確認して報告に書く）。
- [ ] **Step 2: `scripts/make-skill-zip.sh`** — 一時ディレクトリに `kikiwake/` を作って `skill/` の中身をコピーし、`dist/kikiwake-skill.zip`（ZIP のルートに `kikiwake/SKILL.md`）を作る。`dist/` を `.gitignore` に追加。実行して `unzip -l dist/kikiwake-skill.zip` の結果を報告に貼る。
- [ ] **Step 3: README に「Cowork で使う」節を追加** — 記録フォルダの作り方と接続（+ → Add folder）、キーファイル（`.kikiwake/aquavoice_api_key`、Mac のターミナルで作る）、Mac 側の `~/.zshrc` の `KIKIWAKE_HOME`、ZIP の作り方と Customize > Skills からのアップロード、`index.md` / `summary.md` の説明、iCloud 同期の注意（spec §1）。
- [ ] **Step 4: 確認** — `uv run pytest` 全件 pass（コードは変えていないこと）、`bash scripts/make-skill-zip.sh` 成功。
- [ ] **Step 5: コミット** — `git add -A && git commit -m "docs: Cowork mode in the skill and README; skill ZIP script"`

---

## 実装後に controller（このセッション）が行うこと（サブエージェントには渡さない）

1. `uv tool install --reinstall .`、`cd ~ && kikiwake --help` に `index` があることを確認。
2. spec §6 の移行（DB とモデルの移動、サンプル会議の取り込み、`index` 作成）。`~/.zshrc` とキーファイルはユーザーと相談して行う。`probe.*` の削除はユーザーの了承後。
3. GitHub への push はユーザーに確認してから（Cowork のインストールは GitHub の内容を使うため、実地確認の前に必要）。
4. ユーザーに Cowork での実地確認（spec §8）を依頼し、結果を見て直す。
