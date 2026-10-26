# speaker-transcribe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 会議録音を Avalon API で話者分離付き文字起こしし、ローカルの声紋 DB で録音をまたいで話者に名前を付ける CLI と、それを呼ぶ AI スキルを作る。

**Architecture:** `src/speaker_transcribe/` に責務ごとの小さなモジュールを置き、`pipeline` がそれらを順に呼ぶ。外部依存（Avalon・ffmpeg・埋め込みモデル）は注入可能にし、単体テストは偽実装でネット接続なしに回す。CLI は `argparse` の薄い層で、`AppError` を終了コードに変換するだけ。

**Tech Stack:** Python 3.12（uv 管理）, httpx, numpy, scipy, speechbrain（ECAPA-TDNN）+ torch/torchaudio, librosa, matplotlib, pytest, ffmpeg。

**Spec:** `docs/superpowers/specs/2026-10-26-speaker-transcribe-design.md`（各タスクの §番号はこの設計書の節）

## Global Constraints

- Python: `.python-version` は `3.12`、`requires-python = ">=3.11,<3.14"`（このマシンの既定 3.15 には torch の wheel が無い前提）。
- パッケージ `speaker_transcribe`、`src/` レイアウト、ビルドは hatchling、コマンド名 `speaker-transcribe`（`speaker_transcribe.cli:main`）。
- API キー環境変数の優先順: `AQUAVOICE_API` → `AQUAVOICE_API_KEY` → `AQUA_API_KEY`。キーをログ・出力・例外メッセージに出さない。
- データディレクトリ: `~/.local/share/speaker-transcribe/`、環境変数 `SPEAKER_TRANSCRIBE_HOME` で上書き。中に `db.json` / `config.toml` / `models/`。
- 埋め込みモデル: `speechbrain/spkrec-ecapa-voxceleb`、192 次元、L2 正規化済み。
- Avalon: base URL `https://api.aquavoice.com`、モデル `avalon-v1.5`、`diarize=true`、`response_format=diarized_json`、非同期ジョブのみ。
- 終了コード: `0` 成功 / `1` 実行時エラー（`AppError`）/ `2` 使い方・環境エラー（`UsageError`）。
- `--json` 指定時、標準出力には JSON だけ（`ensure_ascii=False`）。進捗ログは常に標準エラー。
- ファイルの書き込み（`db.json`・`session.json`・`.md`）は一時ファイル → `os.replace` で原子的に行う。
- テスト: `uv run pytest`（既定で `-m "not model and not e2e"`）。`model` は実モデルをダウンロードするテスト、`e2e` は実 API を叩くテスト。

## Review Focus

1. 音声ファイル名に空白・日本語・複数のドットがある（`会議 10.26.m4a`）→ 出力は `会議 10.26.md` などになり、ffmpeg に正しく渡る。テストは Task 9（`session_paths`）と Task 4（`audio`）。
2. 無音・発話ゼロの録音（`turns` が空、`speakers` が空）→ 落ちずに、本文の無い Markdown を出力する。テストは Task 10（render）と Task 11（pipeline）。
3. ある話者の単語がすべて `overlap=true`、または `words` が空のターンがある → その話者は `insufficient` になり、例外にならない。テストは Task 5。
4. `label` に空の名前（`話者1=`）、`？`/`?` で終わる名前、半角 `?` 付きの KEY（`田中?`）が来る → 前の 2 つは `UsageError`、半角 `?` は全角 `？` と同じに扱う。テストは Task 9。
5. 音声ファイルを別フォルダへ移してから再度 `transcribe` する（前回の session が見つからないが、DB には同じ音声の声紋が残っている）→ 同じ SHA-256 の声紋はすべて照合前に取り消され、自分自身と照合しない。テストは Task 7（`remove_by_session`）と Task 11。

---

## File Structure

```
pyproject.toml / .python-version / .gitignore
src/speaker_transcribe/
  __init__.py
  errors.py       AppError, UsageError
  config.py       Settings, data_home, load_settings
  models.py       Avalon 応答の型と parse_diarized
  avalon.py       get_api_key, AvalonClient
  audio.py        ffmpeg ラッパ、wav 読み込み、切り出し、sha256
  embedder.py     Embedder Protocol, EcapaEmbedder
  voiceprint.py   select_segments, split_chunks, Voiceprint, build_voiceprint
  speaker_db.py   Sample, SpeakerDB
  matcher.py      Match, person_score, score_table, assign
  session.py      SpeakerState, Session, load/save, session_paths, atomic_write_text
  labeling.py     resolve_key, apply_label, apply_labels
  render.py       format_ts, warning_messages, render_markdown, build_summary
  pipeline.py     TranscribeOptions, run_transcribe
  viz.py          render_viz
  cli.py          main
tests/
  conftest.py     raw_diarized ビルダー, FakeEmbedder, say_wav ヘルパ
  test_*.py       モジュールごと
  e2e/test_e2e.py
skill/SKILL.md
```

---

### Task 1: プロジェクトの土台・エラー型・設定

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/speaker_transcribe/__init__.py`, `src/speaker_transcribe/errors.py`, `src/speaker_transcribe/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `class AppError(Exception)`（`exit_code = 1`）、`class UsageError(AppError)`（`exit_code = 2`）
  - `@dataclass(frozen=True) class Settings` — フィールドと既定値（§4・§5・§7）:
    `min_word_confidence=0.7, max_word_gap=0.3, edge_trim=0.1, min_segment=1.5, max_total_per_speaker=60.0, min_total_per_speaker=4.0, chunk_seconds=3.0, min_consistency=0.5, auto_threshold=0.60, candidate_threshold=0.45, max_samples_per_person=20, top_k=3, job_timeout=3600.0, poll_initial=5.0, poll_max=15.0, max_retries=5, max_duration=14400.0, model="avalon-v1.5", base_url="https://api.aquavoice.com"`
  - `data_home(env: Mapping[str, str] = os.environ) -> Path`
  - `load_settings(home: Path) -> Settings` — `home / "config.toml"` があれば `tomllib` で読み、同名フィールドを上書き。

- [ ] **Step 1: 土台を作る**

`uv init --lib --name speaker-transcribe` は使わず手で書く。`pyproject.toml` に `[project]`（name `speaker-transcribe`、`requires-python = ">=3.11,<3.14"`）、`[project.scripts] speaker-transcribe = "speaker_transcribe.cli:main"`、hatchling、`[tool.pytest.ini_options]` に `addopts = '-m "not model and not e2e"'` と `markers = ["model: downloads the real embedding model", "e2e: calls the real Avalon API"]`、`pythonpath = ["src"]`。`.python-version` は `3.12`。`.gitignore` に `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`。
依存追加: `uv add httpx numpy scipy speechbrain torch torchaudio librosa matplotlib` と `uv add --dev pytest`。

- [ ] **Step 2: 失敗するテストを書く** — `tests/test_config.py`

```python
def test_defaults_match_spec():
    s = Settings()
    assert (s.auto_threshold, s.candidate_threshold) == (0.60, 0.45)
    assert (s.min_segment, s.max_total_per_speaker, s.min_total_per_speaker) == (1.5, 60.0, 4.0)
    assert s.max_samples_per_person == 20 and s.top_k == 3

def test_data_home_env_override(tmp_path):
    assert data_home({"SPEAKER_TRANSCRIBE_HOME": str(tmp_path)}) == tmp_path

def test_data_home_default():
    assert data_home({"HOME": "/Users/x"}) == Path("/Users/x/.local/share/speaker-transcribe")

def test_load_settings_overrides(tmp_path):
    (tmp_path / "config.toml").write_text("auto_threshold = 0.7\n")
    assert load_settings(tmp_path).auto_threshold == 0.7

def test_load_settings_missing_file(tmp_path):
    assert load_settings(tmp_path) == Settings()

def test_load_settings_unknown_key(tmp_path):
    (tmp_path / "config.toml").write_text("nope = 1\n")
    with pytest.raises(UsageError, match="nope"):
        load_settings(tmp_path)

def test_load_settings_wrong_type(tmp_path):
    (tmp_path / "config.toml").write_text('auto_threshold = "high"\n')
    with pytest.raises(UsageError, match="auto_threshold"):
        load_settings(tmp_path)

def test_exit_codes():
    assert AppError("x").exit_code == 1 and UsageError("x").exit_code == 2
```

- [ ] **Step 3: 実行して失敗を確認** — `uv run pytest tests/test_config.py -v` → ImportError で FAIL
- [ ] **Step 4: `errors.py` と `config.py` を実装** — 型チェックは既定値の型と比較（`int` は `float` フィールドに許可）。`data_home` の既定は `Path(env["HOME"]) / ".local/share/speaker-transcribe"`。
- [ ] **Step 5: 実行して成功を確認** — `uv run pytest tests/test_config.py -v` → 8 passed
- [ ] **Step 6: コミット** — `git add -A && git commit -m "feat: project scaffold, errors and settings"`

---

### Task 2: Avalon 応答の型とパース

**Files:**
- Create: `src/speaker_transcribe/models.py`, `tests/conftest.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Word(text: str, start: float, end: float, speaker_confidence: float, overlap: bool)`
  - `@dataclass(frozen=True) Turn(speaker: str, start: float, end: float, text: str, words: tuple[Word, ...])`
  - `@dataclass(frozen=True) DiarizedTranscription(text: str, language: str, duration: float, speaker_ids: tuple[str, ...], turns: tuple[Turn, ...], warnings: tuple[str, ...])` — `speaker_ids` は `speakers[].first_seen` 昇順。
  - `parse_diarized(raw: dict) -> DiarizedTranscription` — 必須フィールド欠落・型違いは `AppError("Avalon の応答形式が想定外です: <フィールド名>")`。
  - テスト用 `tests/conftest.py`:
    `raw_diarized(words: list[tuple[str, float, float, float, bool]], *, duration: float | None = None, warnings: list[str] = (), language: str = "ja") -> dict`
    — 各タプルは `(speaker, start, end, confidence, overlap)`。連続する同一 speaker の単語を 1 ターンにまとめ、単語の text は `"w"`、ターン text は単語 text の連結、`speakers` は初出順で `first_seen` 付き、`duration` 省略時は最後の単語の end。

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_models.py`

```python
def test_parse_orders_speakers_by_first_seen():
    raw = raw_diarized([("spk_1", 0.0, 0.5, 0.9, False), ("spk_0", 1.0, 1.5, 0.9, False)])
    raw["speakers"].reverse()
    assert parse_diarized(raw).speaker_ids == ("spk_1", "spk_0")

def test_parse_turns_and_words():
    tr = parse_diarized(raw_diarized([("spk_0", 0.0, 0.5, 0.8, False), ("spk_0", 0.6, 1.0, 0.9, True)]))
    assert len(tr.turns) == 1 and tr.turns[0].words[1] == Word("w", 0.6, 1.0, 0.9, True)

def test_parse_empty_transcript():
    tr = parse_diarized(raw_diarized([], duration=30.0))
    assert tr.turns == () and tr.speaker_ids == () and tr.duration == 30.0

def test_parse_warnings():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)], warnings=["speaker_limit_reached"])
    assert parse_diarized(raw).warnings == ("speaker_limit_reached",)

def test_parse_missing_field():
    raw = raw_diarized([("spk_0", 0, 1, 0.9, False)]); del raw["turns"]
    with pytest.raises(AppError, match="turns"):
        parse_diarized(raw)
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_models.py -v` → FAIL
- [ ] **Step 3: `models.py` と `conftest.py` の `raw_diarized` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_models.py -v` → 5 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: parse Avalon diarized_json"`

---

### Task 3: Avalon クライアント

**Files:**
- Create: `src/speaker_transcribe/avalon.py`
- Test: `tests/test_avalon.py`

**Interfaces:**
- Consumes: `Settings`, `AppError`, `UsageError`
- Produces:
  - `API_KEY_ENVS = ("AQUAVOICE_API", "AQUAVOICE_API_KEY", "AQUA_API_KEY")`
  - `get_api_key(env: Mapping[str, str] = os.environ) -> str` — 見つからなければ `UsageError`（3 つの変数名を列挙）。
  - `class AvalonClient(api_key: str, settings: Settings, http: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic)`
  - `AvalonClient.transcribe(audio_path: Path, *, language: str | None = None, prompt: str | None = None, max_speakers: int | None = None) -> dict` — 生の `diarized_json` を返す。

動作（§1・§7）:
- `POST {base_url}/v1/audio/transcription-jobs`、multipart: `file`, `model`, `diarize="true"`, `response_format="diarized_json"`, 指定時のみ `language` / `prompt` / `max_speakers`。ヘッダ `Authorization: Bearer <key>` と `Idempotency-Key: <uuid4>`（1 回の `transcribe` 中は同じ値）。
- リトライ対象: 429、5xx、`httpx.TransportError`。待ち時間 `2, 4, 8, 16, 32` 秒で最大 `max_retries` 回。それ以外の 4xx は即 `AppError(f"Avalon API エラー HTTP {status}: {本文の先頭 300 文字}")`。
- ポーリング: `GET .../transcription-jobs/{id}`。間隔は `poll_initial` から毎回 1.5 倍、上限 `poll_max`。`succeeded` で `GET .../{id}/result`（409 なら次のポーリングへ）。`failed` / `cancelled` / `abandoned` は `AppError` に `status` と `error_code` を含める。投入から `job_timeout` 秒を超えたら `AppError` にジョブ ID を含める。
- 進捗（ジョブ ID・状態）は標準エラーへ。キーは出さない。

- [ ] **Step 1: 失敗するテストを書く** — `httpx.MockTransport` で応答列を返すハンドラを作り、`AvalonClient(..., http=httpx.Client(transport=...), sleep=sleeps.append, monotonic=fake_clock)` で注入する。

```python
def test_get_api_key_priority():
    assert get_api_key({"AQUA_API_KEY": "c", "AQUAVOICE_API_KEY": "b"}) == "b"

def test_get_api_key_missing():
    with pytest.raises(UsageError, match="AQUAVOICE_API"):
        get_api_key({})

def test_job_flow_returns_result(tmp_path):
    # POST→{"id":"j1","status":"queued"}, GET j1→queued, GET j1→succeeded, GET result→DIARIZED
    assert client.transcribe(audio) == DIARIZED
    post = requests[0]
    assert post.headers["Authorization"] == "Bearer secret-key-123" and "Idempotency-Key" in post.headers
    assert b'name="diarize"' in post.content and b"diarized_json" in post.content
    assert sleeps == [5.0, 7.5]

def test_poll_interval_capped():
    # queued を 5 回返してから succeeded
    assert sleeps == [5.0, 7.5, 11.25, 15.0, 15.0, 15.0]

def test_result_409_keeps_polling():
    # succeeded → result 409 → status succeeded → result 200
    assert client.transcribe(audio) == DIARIZED

def test_retry_on_429_reuses_idempotency_key():
    # POST 429, POST 503, POST 200 ...
    assert sleeps[:2] == [2, 4]
    assert len({r.headers["Idempotency-Key"] for r in posts}) == 1

def test_retry_exhausted():
    # POST が常に 500
    with pytest.raises(AppError, match="500"):
        client.transcribe(audio)
    assert len(posts) == 6

def test_4xx_not_retried():
    # POST 401。client は api_key="secret-key-123" で作る
    with pytest.raises(AppError, match="401") as excinfo:
        client.transcribe(audio)
    assert len(posts) == 1 and "secret-key-123" not in str(excinfo.value)

def test_failed_job_reports_error_code():
    # status failed, error_code "audio_decode_failed"
    with pytest.raises(AppError, match="audio_decode_failed"):
        client.transcribe(audio)

def test_job_timeout_mentions_job_id():
    # fake_clock が 3601 秒進む
    with pytest.raises(AppError, match="j1"):
        client.transcribe(audio)

def test_optional_fields_only_when_given():
    client.transcribe(audio)
    assert b'name="language"' not in posts[0].content
    client.transcribe(audio, language="ja", max_speakers=4)
    assert b'name="max_speakers"' in posts[1].content
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_avalon.py -v` → FAIL
- [ ] **Step 3: `avalon.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_avalon.py -v` → 11 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: Avalon async job client with retries"`

---

### Task 4: 音声ユーティリティ

**Files:**
- Create: `src/speaker_transcribe/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Produces:
  - `SAMPLE_RATE = 16000`
  - `require_ffmpeg() -> None` — `ffmpeg` / `ffprobe` が `shutil.which` で見つからなければ `UsageError`（`brew install ffmpeg` を含める）。
  - `probe_duration(path: Path) -> float` — `ffprobe -v error -show_entries format=duration -of csv=p=0`。
  - `to_upload_mp3(src: Path, dst: Path) -> None` — `-ac 1 -ar 16000 -c:a libmp3lame -b:a 64k`
  - `to_wav16k(src: Path, dst: Path) -> None` — `-ac 1 -ar 16000 -c:a pcm_s16le`
  - `load_wav(path: Path) -> np.ndarray` — 標準ライブラリ `wave` で読み、`float32`、範囲 [-1, 1]。
  - `cut(wave: np.ndarray, start: float, end: float) -> np.ndarray` — サンプル位置は `round(t * SAMPLE_RATE)`、範囲外は切り詰め。
  - `sha256_file(path: Path) -> str`
- ffmpeg は `subprocess.run([...], check=False, capture_output=True)` でリスト引数（シェルを通さない）。失敗時は `AppError` に stderr の末尾 500 文字を含める。

- [ ] **Step 1: 失敗するテストを書く** — テスト内ヘルパ `make_sine(path, seconds)` は `ffmpeg -f lavfi -i sine=frequency=440:duration=<seconds>` で作り、`ffprobe_stream(path) -> dict` は `ffprobe -show_streams -of json` の最初のストリームの `channels` と `sample_rate`（int）を返す。

```python
def test_roundtrip_wav_and_duration(tmp_path):
    src = tmp_path / "会議 10.26.m4a"; make_sine(src, seconds=2)
    wav = tmp_path / "a.wav"; to_wav16k(src, wav)
    w = load_wav(wav)
    assert w.dtype == np.float32 and abs(len(w) - 32000) < 400 and np.abs(w).max() <= 1.0
    assert probe_duration(src) == pytest.approx(2.0, abs=0.1)

def test_upload_mp3_is_mono_16k(tmp_path):
    to_upload_mp3(src, mp3)
    assert ffprobe_stream(mp3) == {"channels": 1, "sample_rate": 16000}

def test_cut_clamps():
    w = np.zeros(16000, dtype=np.float32)
    assert len(cut(w, 0.5, 2.0)) == 8000 and len(cut(w, -1.0, 0.25)) == 4000

def test_ffmpeg_failure_raises(tmp_path):
    bad = tmp_path / "x.m4a"; bad.write_bytes(b"not audio")
    with pytest.raises(AppError):
        to_wav16k(bad, tmp_path / "y.wav")

def test_require_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(UsageError, match="brew install ffmpeg"):
        require_ffmpeg()

def test_sha256(tmp_path):
    p = tmp_path / "f"; p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_audio.py -v` → FAIL
- [ ] **Step 3: `audio.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_audio.py -v` → 6 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: ffmpeg audio helpers"`

---

### Task 5: 声紋の抽出（区間選択・小片分割・重心）

**Files:**
- Create: `src/speaker_transcribe/embedder.py`（Protocol のみ。実装は Task 6）, `src/speaker_transcribe/voiceprint.py`
- Modify: `tests/conftest.py`（`FakeEmbedder` を追加）
- Test: `tests/test_voiceprint.py`

**Interfaces:**
- Consumes: `DiarizedTranscription`, `Settings`, `audio.cut`
- Produces:
  - `class Embedder(Protocol)`: `model_id: str`, `dim: int`, `embed(wave: np.ndarray) -> np.ndarray`（L2 正規化済み）
  - `Segment = tuple[float, float]`
  - `select_segments(tr: DiarizedTranscription, speaker_id: str, s: Settings) -> list[Segment]` — 時刻順で返す。
  - `split_chunks(segments: list[Segment], s: Settings) -> list[Segment]`
  - `@dataclass Voiceprint(segments: list[Segment], total_seconds: float, centroid: np.ndarray | None, consistency: float | None)` — `centroid is None` が声紋不足。
  - `build_voiceprint(wave: np.ndarray, tr: DiarizedTranscription, speaker_id: str, embedder: Embedder, s: Settings) -> Voiceprint`
  - テスト用 `FakeEmbedder`（`model_id="fake"`, `dim=4`）: `embed(w)` は `[w.mean(), 1.0, 0.0, 0.0]` を L2 正規化して返す。

`select_segments` のアルゴリズム（§4.1。テストだけでは決まらない部分）:
1. 全ターンの全単語を `start` 順に並べて走査する。単語が「対象話者 かつ `not overlap` かつ `speaker_confidence >= min_word_confidence`」なら採用。採用単語は、開いている区間があり `word.start - 区間.end < max_word_gap` なら延長、そうでなければ区間を閉じて新しく開く。不採用の単語（他の話者・重なり・低信頼の自分の単語）が来たら区間を閉じる。
2. 各区間を `(start + edge_trim, end - edge_trim)` にし、長さ `< min_segment` を捨てる。
3. 長い順（同じ長さなら開始が早い順）に、合計が `max_total_per_speaker` を超えない範囲で採用する。次の区間で超える場合は、残り時間が `min_segment` 以上ならその区間を `(start, start + 残り時間)` に切り詰めて採用し、そこで打ち切る。
4. 時刻順に並べ直して返す。

`build_voiceprint`: 合計 `< min_total_per_speaker` なら `centroid=None, consistency=None`。それ以外は `split_chunks` の各小片を `embed` → 平均 → L2 正規化。`consistency` は小片と重心のコサイン類似度の平均。

- [ ] **Step 1: 失敗するテストを書く**

```python
S = Settings()

def seg(tr_words, sid="spk_0", **kw):
    out = select_segments(parse_diarized(raw_diarized(tr_words)), sid, replace(S, **kw))
    return [(round(a, 6), round(b, 6)) for a, b in out]      # 浮動小数の誤差を吸収

def test_joins_close_words_and_trims_edges():
    words = [("spk_0", 0.0, 1.0, 0.9, False), ("spk_0", 1.2, 2.0, 0.9, False)]
    assert seg(words) == [(0.1, 1.9)]

def test_gap_splits_segments():
    words = [("spk_0", 0.0, 2.0, 0.9, False), ("spk_0", 2.4, 4.5, 0.9, False)]
    assert seg(words) == [(0.1, 1.9), (2.5, 4.4)]

def test_overlap_and_low_confidence_break_runs():
    words = [("spk_0", 0, 2, 0.9, False), ("spk_0", 2.0, 2.2, 0.9, True), ("spk_0", 2.2, 4.4, 0.9, False),
             ("spk_0", 4.4, 4.6, 0.5, False)]
    assert seg(words) == [(0.1, 1.9), (2.3, 4.3)]

def test_other_speaker_breaks_run():
    words = [("spk_0", 0, 2, 0.9, False), ("spk_1", 2.0, 2.1, 0.9, False), ("spk_0", 2.1, 4.1, 0.9, False)]
    assert seg(words) == [(0.1, 1.9), (2.2, 4.0)]

def test_short_segments_dropped():
    assert seg([("spk_0", 0.0, 1.6, 0.9, False)]) == []          # 1.4 s after trim

def test_total_capped_longest_first_and_truncated():
    words = [("spk_0", 0, 30.2, 0.9, False), ("spk_1", 30.2, 31, 0.9, False),
             ("spk_0", 31, 51.2, 0.9, False), ("spk_1", 51.2, 52, 0.9, False),
             ("spk_0", 52, 72.2, 0.9, False)]
    assert seg(words) == [(0.1, 30.1), (31.1, 51.1), (52.1, 62.1)]   # 最後の 20 秒は先頭 10 秒に切り詰め

def test_all_overlap_speaker_is_insufficient():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 10, 0.9, True)]))
    vp = build_voiceprint(np.zeros(160000, np.float32), tr, "spk_0", FakeEmbedder(), S)
    assert vp.centroid is None and vp.segments == [] and vp.total_seconds == 0.0

def test_under_four_seconds_is_insufficient():
    tr = parse_diarized(raw_diarized([("spk_0", 0, 3.9, 0.9, False)]))
    assert build_voiceprint(np.zeros(80000, np.float32), tr, "spk_0", FakeEmbedder(), S).centroid is None

def test_split_chunks():
    assert split_chunks([(0.0, 7.0)], S) == [(0.0, 3.0), (3.0, 7.0)]          # remainder 1.0 merged
    assert split_chunks([(0.0, 7.5)], S) == [(0.0, 3.0), (3.0, 6.0), (6.0, 7.5)]
    assert split_chunks([(0.0, 2.0)], S) == [(0.0, 2.0)]

def test_centroid_normalized_and_consistency():
    wave = np.full(16000 * 10, 0.5, np.float32)
    tr = parse_diarized(raw_diarized([("spk_0", 0, 10, 0.9, False)]))
    vp = build_voiceprint(wave, tr, "spk_0", FakeEmbedder(), S)
    assert np.linalg.norm(vp.centroid) == pytest.approx(1.0) and vp.consistency == pytest.approx(1.0)
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_voiceprint.py -v` → FAIL
- [ ] **Step 3: `embedder.py`（Protocol）、`voiceprint.py`、`FakeEmbedder` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_voiceprint.py -v` → 10 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: clean-segment selection and voiceprint centroid"`

---

### Task 6: ECAPA 埋め込みモデル

**Files:**
- Modify: `src/speaker_transcribe/embedder.py`, `tests/conftest.py`（`say_wav` を追加）
- Test: `tests/test_embedder.py`（`@pytest.mark.model`）

**Interfaces:**
- Consumes: `Embedder` Protocol, `audio.to_wav16k`, `audio.load_wav`
- Produces:
  - 定数 `MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"`, `DIM = 192`（モデルを読まずに DB を開くため、`cli` はこの定数を使う）
  - `class EcapaEmbedder(savedir: Path)` — クラス属性 `model_id = MODEL_ID`, `dim = DIM`。コンストラクタで `speechbrain.inference.speaker.EncoderClassifier.from_hparams(source=model_id, savedir=str(savedir), run_opts={"device": "cpu"})`。ダウンロード失敗は `AppError("埋め込みモデルのダウンロードに失敗しました。ネットワーク接続を確認してください")`。
  - `embed`: `encode_batch(torch.from_numpy(wave)[None])` を `torch.no_grad()` で実行 → `(192,)` の numpy → L2 正規化。
  - テスト用 `say_wav(voice: str, text: str, path: Path) -> Path` — `say -v <voice> -o x.aiff` → `to_wav16k`。`say` が無ければ `pytest.skip`。

- [ ] **Step 1: 失敗するテストを書く**

```python
pytestmark = pytest.mark.model

def test_shape_and_norm(tmp_path, embedder):
    v = embedder.embed(load_wav(say_wav("Kyoko", "本日の議題は予算についてです。", tmp_path / "a.wav")))
    assert v.shape == (192,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)

def test_same_voice_closer_than_different(tmp_path, embedder):
    k1 = emb("Kyoko", "本日の議題は予算についてです。")
    k2 = emb("Kyoko", "来週までに資料をまとめておきます。")
    g = emb("Grandpa", "来週までに資料をまとめておきます。")
    assert k1 @ k2 > k1 @ g
```
`embedder` は `tmp_path_factory` の savedir を使う session スコープ fixture。

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest -m model tests/test_embedder.py -v` → FAIL
- [ ] **Step 3: `EcapaEmbedder` を実装**。import 時に torchaudio の API 削除で speechbrain が落ちる場合は `torch<2.9` と `torchaudio<2.9` に固定する（`uv add "torch<2.9" "torchaudio<2.9"`）。
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest -m model tests/test_embedder.py -v` → 2 passed（初回はモデルを数百 MB ダウンロード）
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: SpeechBrain ECAPA embedder"`

---

### Task 7: 声紋 DB

**Files:**
- Create: `src/speaker_transcribe/speaker_db.py`
- Test: `tests/test_speaker_db.py`

**Interfaces:**
- Consumes: `UsageError`, `embedder.MODEL_ID` / `embedder.DIM`（Task 6。テストでは任意の文字列・整数を渡す）
- Produces:
  - `atomic_write_text(path: Path, text: str) -> None` — 同じディレクトリの一時ファイルに書いて `os.replace`。親ディレクトリが無ければ作る。`session`（Task 9）と `cli`（Task 13）もこれを import して使う。
  - `@dataclass Sample(id: str, session_sha256: str, source: Literal["auto", "label"], created_at: str, embedding: np.ndarray)`
  - `class SpeakerDB`:
    - `SpeakerDB.load(path: Path, model_id: str, dim: int, max_samples: int, now: Callable[[], str] = <ISO8601 ローカル時刻>) -> SpeakerDB` — ファイルが無ければ空。`model_id` か `dim` が違えば `UsageError`（両方のモデル名を含める）。
    - `save() -> None` — §5.1 の形式（`version: 1`, `model_id`, `dim`, `people{name: {created_at, samples[...]}}`）。
    - `people() -> dict[str, list[Sample]]`
    - `embeddings() -> dict[str, list[np.ndarray]]`
    - `add_sample(name: str, embedding: np.ndarray, session_sha256: str, source: str) -> str` — UUID4 を返す。人が無ければ作る。`max_samples` を超えたら `created_at` の最も古いものから削除。
    - `remove_samples(ids: Iterable[str]) -> None` — 存在しない ID は無視。サンプルが 0 になった人は削除。
    - `remove_by_session(session_sha256: str) -> int` — その SHA のサンプルをすべて削除し、件数を返す。空の人は削除。
    - `rename(old: str, new: str) -> None` / `delete(name: str) -> None` — 無い名前・既存の新名は `UsageError`。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_add_save_load_roundtrip(tmp_path):
    db = new_db(tmp_path); sid = db.add_sample("田中", unit(0), "sha1", "label"); db.save()
    db2 = new_db(tmp_path)
    s = db2.people()["田中"][0]
    assert s.id == sid and s.source == "label" and np.allclose(s.embedding, unit(0))
    assert json.loads((tmp_path / "db.json").read_text())["model_id"] == "m"

def test_max_samples_drops_oldest(tmp_path):
    db = new_db(tmp_path, max_samples=2, clock=["t1", "t2", "t3"])
    ids = [db.add_sample("田中", unit(0), f"s{i}", "auto") for i in range(3)]
    assert [s.id for s in db.people()["田中"]] == ids[1:]

def test_remove_samples_ignores_missing_and_drops_empty_person(tmp_path):
    db = new_db(tmp_path); sid = db.add_sample("田中", unit(0), "s", "auto")
    db.remove_samples([sid, "nope"])
    assert "田中" not in db.people()

def test_remove_by_session(tmp_path):
    db = new_db(tmp_path)
    db.add_sample("田中", unit(0), "A", "auto"); db.add_sample("田中", unit(0), "B", "label")
    db.add_sample("佐藤", unit(1), "A", "label")
    assert db.remove_by_session("A") == 2
    assert list(db.people()) == ["田中"] and db.people()["田中"][0].session_sha256 == "B"

def test_model_mismatch(tmp_path):
    new_db(tmp_path).save()
    with pytest.raises(UsageError, match="other"):
        SpeakerDB.load(tmp_path / "db.json", "other", 192, 20)

def test_rename_and_delete(tmp_path):
    db = new_db(tmp_path); db.add_sample("田中", unit(0), "s", "auto"); db.add_sample("佐藤", unit(1), "s", "auto")
    with pytest.raises(UsageError): db.rename("田中", "佐藤")
    db.rename("田中", "鈴木"); db.delete("佐藤")
    assert list(db.people()) == ["鈴木"]
    with pytest.raises(UsageError): db.delete("佐藤")

def test_atomic_write_leaves_no_temp(tmp_path):
    atomic_write_text(tmp_path / "sub" / "f.json", "x")
    assert [p.name for p in (tmp_path / "sub").iterdir()] == ["f.json"]
```
（`new_db(tmp_path, max_samples=20, clock=None)` は `model_id="m", dim=192` で `load` するテスト内ヘルパ、`unit(i)` は i 番目だけ 1 の 192 次元ベクトル。）

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_speaker_db.py -v` → FAIL
- [ ] **Step 3: `speaker_db.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_speaker_db.py -v` → 7 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: persistent voiceprint DB"`

---

### Task 8: 照合

**Files:**
- Create: `src/speaker_transcribe/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `Settings`
- Produces:
  - `@dataclass(frozen=True) Match(name: str | None, score: float | None, status: Literal["auto", "candidate", "unknown"])`
  - `person_score(centroid: np.ndarray, embeddings: list[np.ndarray], top_k: int) -> float` — コサイン類似度の上位 `top_k` 件の平均。
  - `score_table(centroids: dict[str, np.ndarray], people: dict[str, list[np.ndarray]], top_k: int) -> dict[str, dict[str, float]]` — `{speaker_id: {name: score}}`
  - `assign(scores: dict[str, dict[str, float]], s: Settings) -> dict[str, Match]` — `scipy.optimize.linear_sum_assignment(matrix, maximize=True)`。`score >= auto_threshold` → `auto`、`>= candidate_threshold` → `candidate`、それ未満・未割り当て → `Match(None, None, "unknown")`。`scores` の全キーが結果に含まれる。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_person_score_top3_mean():
    c = unit(0)
    embs = [unit(0), unit(0), mix(0.5), unit(1)]       # sims 1, 1, 0.5, 0
    assert person_score(c, embs, 3) == pytest.approx(2.5 / 3)
    assert person_score(c, [mix(0.5)], 3) == pytest.approx(0.5)

def test_thresholds():
    m = assign({"a": {"田中": 0.60}, "b": {"佐藤": 0.59}, "c": {"鈴木": 0.44}}, Settings())
    assert m["a"] == Match("田中", 0.60, "auto")
    assert m["b"] == Match("佐藤", 0.59, "candidate")
    assert m["c"] == Match(None, None, "unknown")

def test_one_to_one_maximizes_total():
    m = assign({"a": {"田中": 0.9, "佐藤": 0.8}, "b": {"田中": 0.85, "佐藤": 0.1}}, Settings())
    assert (m["a"].name, m["b"].name) == ("佐藤", "田中")

def test_more_speakers_than_people():
    m = assign({"a": {"田中": 0.9}, "b": {"田中": 0.8}}, Settings())
    assert m["a"].name == "田中" and m["b"] == Match(None, None, "unknown")

def test_empty_db():
    assert assign({"a": {}}, Settings()) == {"a": Match(None, None, "unknown")}
    assert assign({}, Settings()) == {}
```
（`mix(x)` は `unit(0)` とのコサインが `x` になる単位ベクトル。）

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_matcher.py -v` → FAIL
- [ ] **Step 3: `matcher.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_matcher.py -v` → 5 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: one-to-one voiceprint matching"`

---

### Task 9: セッションと名前付け（`label` のロジック）

**Files:**
- Create: `src/speaker_transcribe/session.py`, `src/speaker_transcribe/labeling.py`
- Test: `tests/test_session.py`, `tests/test_labeling.py`

**Interfaces:**
- Consumes: `parse_diarized`, `SpeakerDB`, `atomic_write_text`, `UsageError`
- Produces:
  - `Status = Literal["auto", "candidate", "unknown", "insufficient", "confirmed"]`
  - `@dataclass SpeakerState(id: str, number: int, segments: list[tuple[float, float]], total_seconds: float, centroid: list[float] | None, consistency: float | None, scores: dict[str, float], status: Status, name: str | None, score: float | None, db_sample_ids: list[str])`
    - `display_label` プロパティ: `auto` / `confirmed` → `name`、`candidate` → `f"{name}？"`、それ以外 → `f"話者{number}"`
  - `@dataclass Session(audio_path: str, audio_sha256: str, created_at: str, model_id: str, avalon_response: dict, speakers: dict[str, SpeakerState], warnings: list[dict], version: int = 1)`
    - `transcription` プロパティ → `parse_diarized(avalon_response)`
    - `warnings` の要素: `{"code": "speaker_limit_reached"}` / `{"code": "low_consistency", "speaker": "spk_1"}` / `{"code": "labels_discarded"}`
  - `load_session(path: Path) -> Session` / `save_session(session: Session, path: Path) -> None`（`atomic_write_text`、JSON は `ensure_ascii=False, indent=2`）
  - `session_paths(audio: Path, output_dir: Path | None) -> tuple[Path, Path, Path]` — `(<stem>.md, <stem>.session.json, <stem>.voices.png)`。`stem` は `audio.stem`（最後の拡張子だけ除く）。
  - `resolve_key(session: Session, key: str) -> str` — 話者 ID か `display_label` に一致。KEY の半角 `?` は全角 `？` に置換してから比較。0 件は `UsageError`（有効な KEY をすべて列挙）、2 件以上は `UsageError`（「話者 ID で指定してください」と該当 ID）。
  - `apply_label(session: Session, db: SpeakerDB, speaker_id: str, name: str) -> list[str]` — 警告文のリストを返す。
  - `apply_labels(session: Session, db: SpeakerDB, pairs: list[tuple[str, str]]) -> list[str]` — 先に全 KEY を解決・全名前を検証し、1 つでも不正なら何も変更せず `UsageError`。名前は前後の空白を除き、空・`？` または `?` で終わる名前は `UsageError`。

`apply_label` の手順（§3 の表）:
1. 声紋不足（`centroid is None`）: `name` を設定、`status="confirmed"`、DB は触らず、警告 `f"{name} は声紋が不足しているため、DB には登録しません"` を返す。
2. それ以外: `db.remove_samples(st.db_sample_ids)` → 他の話者で `status == "candidate" and name == 新名` のものを `unknown` に戻す（`name=None, score=None`）→ `db.add_sample(name, np.array(st.centroid), session.audio_sha256, "label")` → `st.db_sample_ids=[新ID]`, `st.name=name`, `st.status="confirmed"`。
   （統合は特別扱い不要。別の話者の `auto` / `confirmed` はそのまま残り、両方が同じ表示名になる。）

- [ ] **Step 1: 失敗するテストを書く** — `tests/test_session.py`

```python
def test_session_paths_keep_spaces_and_dots(tmp_path):
    md, ss, png = session_paths(tmp_path / "会議 10.26.m4a", None)
    assert (md.name, ss.name, png.name) == ("会議 10.26.md", "会議 10.26.session.json", "会議 10.26.voices.png")
    assert md.parent == tmp_path
    assert session_paths(tmp_path / "a.m4a", tmp_path / "out")[0] == tmp_path / "out" / "a.md"

def test_display_labels():
    assert state(status="auto", name="田中").display_label == "田中"
    assert state(status="candidate", name="田中").display_label == "田中？"
    assert state(status="unknown", number=2).display_label == "話者2"
    assert state(status="insufficient", number=3).display_label == "話者3"

def test_roundtrip(tmp_path):
    s = make_session(); save_session(s, tmp_path / "x.session.json")
    assert load_session(tmp_path / "x.session.json") == s
```

`tests/test_labeling.py`（`make_session` は spk_0〜spk_2 を持つ Session を作るヘルパ。DB は Task 7 の `SpeakerDB` を tmp に作る）:

```python
def test_new_name_creates_person_and_confirms(db):
    s = make_session(spk_1=dict(status="unknown", number=2))
    apply_labels(s, db, [("話者2", "田中")])
    st = s.speakers["spk_1"]
    assert (st.status, st.name, st.display_label) == ("confirmed", "田中", "田中")
    assert [x.id for x in db.people()["田中"]] == st.db_sample_ids

def test_correction_removes_wrong_auto_sample(db):
    wrong = db.add_sample("田中", unit(0), "sha", "auto")
    s = make_session(spk_0=dict(status="auto", name="田中", db_sample_ids=[wrong]))
    apply_labels(s, db, [("田中", "佐藤")])
    assert "田中" not in db.people() and s.speakers["spk_0"].name == "佐藤"

def test_merge_keeps_both(db):
    s = make_session(spk_0=dict(status="auto", name="田中"), spk_2=dict(status="unknown", number=3))
    apply_labels(s, db, [("spk_2", "田中")])
    assert s.speakers["spk_0"].display_label == s.speakers["spk_2"].display_label == "田中"

def test_label_clears_other_candidate(db):
    s = make_session(spk_0=dict(status="candidate", name="田中", number=1), spk_1=dict(status="unknown", number=2))
    apply_labels(s, db, [("話者2", "田中")])
    assert s.speakers["spk_0"].display_label == "話者1"

def test_insufficient_renames_without_db(db):
    s = make_session(spk_2=dict(status="insufficient", centroid=None, number=3))
    warnings = apply_labels(s, db, [("話者3", "鈴木")])
    assert s.speakers["spk_2"].display_label == "鈴木" and db.people() == {} and "鈴木" in warnings[0]

def test_halfwidth_question_mark_key(db):
    s = make_session(spk_0=dict(status="candidate", name="田中"))
    apply_labels(s, db, [("田中?", "田中")])
    assert s.speakers["spk_0"].status == "confirmed"

def test_ambiguous_key(db):
    s = make_session(spk_0=dict(status="auto", name="田中"), spk_1=dict(status="confirmed", name="田中"))
    with pytest.raises(UsageError, match="spk_0"):
        apply_labels(s, db, [("田中", "佐藤")])

def test_invalid_input_changes_nothing(db):
    s = make_session(); before = copy.deepcopy(s)
    for pairs in ([("話者1", "田中"), ("nope", "佐藤")], [("話者1", "")], [("話者1", "田中？")], [("話者1", "田中?")]):
        with pytest.raises(UsageError):
            apply_labels(s, db, pairs)
    assert s == before and db.people() == {}
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_session.py tests/test_labeling.py -v` → FAIL
- [ ] **Step 3: `session.py` と `labeling.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_session.py tests/test_labeling.py -v` → 11 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: session model and label logic"`

---

### Task 10: Markdown と AI 向けサマリ

**Files:**
- Create: `src/speaker_transcribe/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Session`, `SpeakerState.display_label`, `Session.transcription`
- Produces:
  - `format_ts(seconds: float) -> str` — `HH:MM:SS`（切り捨て）
  - `warning_messages(session: Session) -> list[str]` — 以下の固定文言を、この順で:
    1. 未確定（`candidate` / `unknown` / `insufficient`）が n > 0 人: `f"名前が未確定の話者が {n} 人います。"`
    2. `speaker_limit_reached`: `"9 人以上いた可能性があり、一部の発言は別の話者に割り当てられています。"`
    3. `low_consistency`: `f"{display_label} の声に複数人が混ざっている可能性があります。"`
    4. `labels_discarded`: `"API を呼び直したため、前回の名前付けは破棄されました。"`
  - `render_markdown(session: Session) -> str`
  - `build_summary(session: Session, markdown_path: Path, session_path: Path, viz_path: Path | None) -> dict` — §6.3 の形。`warnings` は `warning_messages` の結果。

`render_markdown` の形（§6.1）:
```
# {Path(audio_path).name} の文字起こし

- 録音: {round(duration / 60)} 分 / 話者: {display_label を speakers の順に「・」で連結。同じラベルは 1 回}

> ⚠ {warning}            ← warning_messages ごとに 1 行。無ければこのブロックごと省略

**[{format_ts(段落の最初のターンの start)}] {display_label}**
{本文}{" ※同時発話あり" if 段落の単語の過半数が overlap}
```
- 連続する同じ `display_label` のターンを 1 段落にまとめる。本文はターン text の連結で、`language` が `ja` で始まるなら区切りなし、それ以外は半角スペース区切り。段落の間は空行 1 つ。ファイル末尾は改行 1 つ。
- `examples`: `auto` / `confirmed` 以外の話者に付ける。その話者のターンを text の長い順に最大 3 件、`{"time": format_ts(start), "text": text[:60] + ("…" if len(text) > 60 else "")}`。
- `speakers` の各要素: `{"id", "label", "status", "score"}`（`score` は無ければ `null`）＋ 必要なら `"examples"`。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_format_ts():
    assert format_ts(0) == "00:00:00" and format_ts(3725.9) == "01:02:05"

def test_markdown_golden():
    s = session_from_words(
        [("spk_0", 12.0, 13.0, 0.9, False), ("spk_0", 13.0, 14.0, 0.9, False),
         ("spk_1", 45.0, 46.0, 0.9, True), ("spk_1", 46.0, 47.0, 0.9, True), ("spk_1", 47.0, 48.0, 0.9, False),
         ("spk_0", 50.0, 51.0, 0.9, False)],
        duration=3120.0,
        states={"spk_0": dict(status="auto", name="田中"), "spk_1": dict(status="unknown", number=2)})
    assert render_markdown(s) == (
        "# meeting.m4a の文字起こし\n\n"
        "- 録音: 52 分 / 話者: 田中・話者2\n\n"
        "> ⚠ 名前が未確定の話者が 1 人います。\n\n"
        "**[00:00:12] 田中**\nww\n\n"
        "**[00:00:45] 話者2**\nwww ※同時発話あり\n\n"
        "**[00:00:50] 田中**\nw\n")

def test_merged_speakers_form_one_paragraph():
    # spk_0 と spk_2 が両方 confirmed「田中」で、連続して話す
    assert render_markdown(s).count("] 田中**") == 1

def test_empty_transcript():
    s = session_from_words([], duration=30.0, states={})
    assert render_markdown(s) == "# meeting.m4a の文字起こし\n\n- 録音: 0 分 / 話者: なし\n"

def test_warning_order():
    s = ...  # unknown 1 人、warnings = [low_consistency spk_0, speaker_limit_reached, labels_discarded]
    assert warning_messages(s) == ["名前が未確定の話者が 1 人います。",
        "9 人以上いた可能性があり、一部の発言は別の話者に割り当てられています。",
        "田中 の声に複数人が混ざっている可能性があります。",
        "API を呼び直したため、前回の名前付けは破棄されました。"]

def test_summary_examples():
    sm = build_summary(s, Path("m.md"), Path("m.session.json"), None)
    unknown = next(x for x in sm["speakers"] if x["id"] == "spk_1")
    assert unknown["label"] == "話者2" and unknown["examples"][0]["time"] == "00:00:45"
    assert "examples" not in next(x for x in sm["speakers"] if x["id"] == "spk_0")
    assert sm["viz_path"] is None and sm["markdown_path"] == "m.md"

def test_example_truncated_to_60():
    # text が 80 文字のターン
    assert unknown["examples"][0]["text"] == "あ" * 60 + "…"
```
（`session_from_words` は `raw_diarized` から `Session` を作るヘルパ。`audio_path="meeting.m4a"`。発話ゼロの話者一覧は `話者: なし`。）

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_render.py -v` → FAIL
- [ ] **Step 3: `render.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_render.py -v` → 7 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: markdown and AI summary rendering"`

---

### Task 11: transcribe パイプライン

**Files:**
- Create: `src/speaker_transcribe/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 2〜10 のすべて
- Produces:
  - `@dataclass TranscribeOptions(output_dir: Path | None = None, language: str | None = None, prompt: str | None = None, max_speakers: int | None = None, force_api: bool = False)`
  - `Transcriber = Callable[[Path, TranscribeOptions], dict]` — アップロード用 MP3 のパスを受け取り、生の応答を返す（CLI では `AvalonClient.transcribe` を包む）。
  - `run_transcribe(audio: Path, opts: TranscribeOptions, *, settings: Settings, db: SpeakerDB, embedder: Embedder, transcriber: Transcriber, now: Callable[[], str]) -> tuple[Session, Path, Path]` — `(session, markdown_path, session_path)` を返す。

手順（§3 transcribe）:
1. `require_ffmpeg()`。`sha = sha256_file(audio)`。`md, ss, _ = session_paths(audio, opts.output_dir)`。
2. `prev = load_session(ss)`（存在し、`audio_sha256 == sha` のときだけ。違えば無視）。
3. `db.remove_by_session(sha)`（Review Focus 5）。
4. `probe_duration(audio) > settings.max_duration` → `AppError("録音が 4 時間を超えています（上限 4 時間）")`。
5. 一時ディレクトリで `to_wav16k`。応答は `prev.avalon_response`（`prev` があり `not force_api`）か、`to_upload_mp3` した MP3 で `transcriber(mp3, opts)`。
6. `tr = parse_diarized(raw)` → `tr.speaker_ids` の各話者に `build_voiceprint` → 声紋不足以外で `score_table(..., db.embeddings(), settings.top_k)` → `assign`。
7. `SpeakerState` を `tr.speaker_ids` の順に作る（`number` は 1 から）。`auto` は `db.add_sample(name, centroid, sha, "auto")` の ID を `db_sample_ids` に入れる。`consistency < min_consistency` なら `low_consistency` 警告。`tr.warnings` に `speaker_limit_reached` があれば警告。
8. `prev` がある場合: `not force_api` なら、`prev` で `confirmed` だった話者に `apply_label(session, db, sid, name)`（ID が今回も存在するものだけ）。`force_api` で `confirmed` が 1 人でもいれば `labels_discarded` 警告。
9. `db.save()` → `save_session` → `atomic_write_text(md, render_markdown(session))`。

- [ ] **Step 1: 失敗するテストを書く** — テスト fixture `env` が numpy で `meeting.wav`（16 kHz、30 秒）を書く。spk_0 の発話区間は 150 Hz、spk_1 は 250 Hz の正弦波。`env.response` はその区間に合わせた `raw_diarized`。`env.other_meeting_same_voices` は振幅だけ変えた別ファイル（SHA が違い、声は同じ）。埋め込みはテスト内の `FreqEmbedder`（FFT の最大ピークが 200 Hz 未満なら `[1,0,0,0]`、以上なら `[0,1,0,0]`）。`transcriber` は呼び出し回数 `env.calls` を数えて `env.response` を返す偽物。`env.run(audio=None, **opts)` は `run_transcribe` を呼ぶ。

```python
def test_first_run_writes_outputs_and_unknown_labels(env):
    session, md, ss = env.run()
    assert md.read_text().startswith("# meeting.wav の文字起こし")
    assert [s.display_label for s in session.speakers.values()] == ["話者1", "話者2"]
    assert env.calls == 1

def test_second_meeting_auto_names_after_label(env):
    s1, _, ss = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")]); env.db.save()
    s2, _, _ = env.run(audio=env.other_meeting_same_voices)
    assert s2.speakers["spk_0"].status == "auto" and s2.speakers["spk_0"].name == "田中"
    assert len(s2.speakers["spk_0"].db_sample_ids) == 1

def test_rerun_uses_cache_and_reapplies_labels(env):
    s1, _, ss = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")]); save_session(s1, ss); env.db.save()
    s2, _, _ = env.run()
    assert env.calls == 1                                   # API は呼ばれない
    assert s2.speakers["spk_0"].status == "confirmed"
    assert len(env.db.people()["田中"]) == 1                # 二重追加されない

def test_force_api_discards_labels(env):
    ...  # 上と同じ準備で force_api=True
    assert env.calls == 2 and {"code": "labels_discarded"} in s2.warnings

def test_moved_audio_does_not_self_match(env, tmp_path):
    s1, _, _ = env.run()
    apply_labels(s1, env.db, [("spk_0", "田中")]); env.db.save()
    moved = tmp_path / "moved" / "meeting.wav"; moved.parent.mkdir(); shutil.copy(env.audio, moved)
    s2, _, _ = env.run(audio=moved)
    assert s2.speakers["spk_0"].status == "unknown"         # 自分自身の声紋は取り消し済み

def test_too_long_audio_rejected_before_upload(env, monkeypatch):
    monkeypatch.setattr(pipeline, "probe_duration", lambda p: 14400.1)
    with pytest.raises(AppError, match="4 時間"):
        env.run()
    assert env.calls == 0

def test_silent_recording(env):
    env.response = raw_diarized([], duration=30.0)
    session, md, _ = env.run()
    assert session.speakers == {} and "話者: なし" in md.read_text()

def test_speaker_limit_warning(env):
    env.response["warnings"] = ["speaker_limit_reached"]
    assert {"code": "speaker_limit_reached"} in env.run()[0].warnings
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_pipeline.py -v` → FAIL
- [ ] **Step 3: `pipeline.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_pipeline.py -v` → 8 passed
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: transcribe pipeline with cache and re-labeling"`

---

### Task 12: 可視化

**Files:**
- Create: `src/speaker_transcribe/viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `Session`, `SpeakerState.segments` / `centroid` / `display_label`, `audio.cut`, `SAMPLE_RATE`
- Produces: `render_viz(session: Session, wave: np.ndarray, people: dict[str, list[np.ndarray]], out_path: Path) -> None`

内容（§6.4）:
- `matplotlib.use("Agg")`。日本語フォントは `rcParams["font.family"] = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "sans-serif"]`。1 行 3 列、`figsize=(18, 5)`、`dpi=120`。
- 対象は `centroid is not None` の話者。凡例は `display_label`（統合された同名の話者は別の線として両方描く）。
- ① ピッチ: 採用区間ごとに `librosa.pyin(seg, fmin=60, fmax=500, sr=16000)` の有声フレームの F0 を集め、`bins=np.arange(60, 505, 10)` のヒストグラム（`histtype="step"`）。x 軸「基本周波数 (Hz)」。
- ② 長時間平均スペクトル: 採用区間を連結し、`n_fft=1024, hop=512` の `librosa.stft` の |X|² を時間平均 → `10*log10(+1e-12)`、0〜8000 Hz。x 軸「周波数 (Hz)」、y 軸「パワー (dB)」。
- ③ 声紋マップ: 会議の重心と DB の全サンプルを縦に積み、平均を引いて `np.linalg.svd` の上位 2 成分に射影。会議の話者は `marker="*", s=300` で `display_label` を注記、DB のサンプルは人ごとの色の小さな点（凡例に人名）。点が 2 つ未満なら軸の中央に「データ不足」と書く。
- 対象話者が 0 人なら 3 枚とも「データ不足」。
- 保存は `fig.savefig(out_path)` → `plt.close(fig)`。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_writes_png(tmp_path):
    session, wave = two_speaker_session_with_sine_waves()   # 150 Hz と 250 Hz の 10 秒ずつ
    out = tmp_path / "m.voices.png"
    render_viz(session, wave, {"田中": [unit(0), unit(1)]}, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

def test_no_eligible_speakers(tmp_path):
    render_viz(session_with_only_insufficient(), np.zeros(16000, np.float32), {}, tmp_path / "x.png")
    assert (tmp_path / "x.png").exists()
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_viz.py -v` → FAIL
- [ ] **Step 3: `viz.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest tests/test_viz.py -v` → 2 passed。生成された PNG を開き、3 枚の図と日本語ラベルが文字化けしていないことを目で確認する。
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: voice visualization"`

---

### Task 13: CLI

**Files:**
- Create: `src/speaker_transcribe/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: すべて
- Produces: `main(argv: list[str] | None = None) -> int`

構成（§3）:
- 共通: `home = data_home()`、`settings = load_settings(home)`。DB は `SpeakerDB.load(home / "db.json", MODEL_ID, DIM, settings.max_samples_per_person)`。`EcapaEmbedder(home / "models")` は `transcribe` のときだけ作る（`label` / `speakers` / `viz` ではモデルを読まない）。テストから差し替えられるよう、`cli` モジュールに `make_embedder(home) -> Embedder` と `make_transcriber(settings) -> Transcriber` を置く。`make_transcriber` は最初の呼び出し時にだけ `get_api_key()` を呼ぶ（キャッシュ再利用時はキー不要）。
- `transcribe AUDIO [--output-dir] [--language] [--prompt] [--max-speakers N(1..8)] [--viz] [--force-api] [--json]`
  → `run_transcribe`。`--viz` なら `viz` と同じ処理。`--json` なら `build_summary` を出力、なければ標準出力に Markdown のパスを 1 行。
- `label SESSION KEY=NAME... [--json]` → `load_session` → `apply_labels`（`=` が無い引数は `UsageError`。`split("=", 1)`）→ `db.save()` → `save_session` → Markdown を書き直し → `.voices.png` が既にあれば再生成。警告は標準エラー。`--json` なら `build_summary`。
- `speakers list [--json]` → `[{"name", "samples", "updated_at"}]`（`updated_at` は最新サンプルの `created_at`）。人間向けは 1 人 1 行 `田中  サンプル 3 件  最終更新 2026-10-26T10:00:00+09:00`。
- `speakers rename OLD NEW` / `speakers delete NAME` → 変更後 `db.save()`。
- `viz SESSION [--json]` → `session.audio_path` から `to_wav16k` して `render_viz(session, wave, db.embeddings(), png)`。音声が無ければ `AppError("元の音声ファイルが見つかりません: <path>")`。
- 例外処理: `AppError` は `f"エラー: {e}"` を標準エラーに出し `e.exit_code` を返す。`argparse` の誤りは 2。

- [ ] **Step 1: 失敗するテストを書く** — fixture `cli_env` は Task 11 と同じ方法で `meeting.wav` と応答を用意し（最初の発話は 0.0 秒から）、`monkeypatch` で `SPEAKER_TRANSCRIBE_HOME=tmp`、`cli.make_embedder` と `cli.make_transcriber` を偽物に差し替える。`cli_env.audio` と `cli_env.session_path`（`meeting.session.json`）を持つ。

```python
def test_transcribe_json_then_label(cli_env, capsys):
    assert main(["transcribe", str(cli_env.audio), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert [s["label"] for s in summary["speakers"]] == ["話者1", "話者2"]
    assert main(["label", summary["session_path"], "spk_0=田中", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["speakers"][0]["label"] == "田中"
    assert "**[00:00:00] 田中**" in Path(summary["markdown_path"]).read_text()

def test_label_bad_key_exit_2(cli_env, capsys):
    main(["transcribe", str(cli_env.audio)])
    assert main(["label", str(cli_env.session_path), "nope=田中"]) == 2
    assert "話者1" in capsys.readouterr().err

def test_label_arg_without_equals(cli_env):
    main(["transcribe", str(cli_env.audio)])
    assert main(["label", str(cli_env.session_path), "田中"]) == 2

def test_missing_api_key_exit_2(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_transcriber", real_make_transcriber)
    for k in API_KEY_ENVS: monkeypatch.delenv(k, raising=False)
    assert main(["transcribe", str(cli_env.audio)]) == 2
    assert "AQUAVOICE_API" in capsys.readouterr().err

def test_speakers_list_rename_delete(cli_env, capsys):
    # label で田中を登録後
    assert main(["speakers", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["name"] == "田中"
    assert main(["speakers", "rename", "田中", "鈴木"]) == 0
    assert main(["speakers", "delete", "佐藤"]) == 2

def test_json_stdout_is_pure_json(cli_env, capsys):
    main(["transcribe", str(cli_env.audio), "--json"])
    json.loads(capsys.readouterr().out)          # 進捗ログが混ざっていないこと

def test_viz_flag_writes_png(cli_env):
    assert main(["transcribe", str(cli_env.audio), "--viz"]) == 0
    assert cli_env.audio.with_name("meeting.voices.png").exists()
```

- [ ] **Step 2: 実行して失敗を確認** — `uv run pytest tests/test_cli.py -v` → FAIL
- [ ] **Step 3: `cli.py` を実装**
- [ ] **Step 4: 実行して成功を確認** — `uv run pytest -v` → 全テスト passed（`model` / `e2e` を除く）
- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: speaker-transcribe CLI"`

---

### Task 14: 通しのテスト・AI スキル・インストール

**Files:**
- Create: `tests/e2e/test_e2e.py`, `skill/SKILL.md`
- Test: `tests/e2e/test_e2e.py`（`@pytest.mark.e2e`、実 API・実モデル）

**Interfaces:**
- Consumes: `say_wav`, `main`

- [ ] **Step 1: 通しのテストを書く**

疑似会議を `say_wav` で作る。1 発言 6〜10 秒の文章を使い、発言の間に 0.7 秒の無音を入れて `ffmpeg` の concat で連結する。
- 会議 A: Kyoko → Grandpa → Sandy → Kyoko → Grandpa → Sandy（各 2 回）
- 会議 B: Grandpa → Sandy → Kyoko → Grandpa → Sandy → Kyoko（別の文章）

```python
pytestmark = pytest.mark.e2e

def test_names_carry_over_between_meetings(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPEAKER_TRANSCRIBE_HOME", str(tmp_path / "home"))
    a, b = make_meeting_a(tmp_path), make_meeting_b(tmp_path)
    assert main(["transcribe", str(a), "--language", "ja", "--json"]) == 0
    sa = json.loads(capsys.readouterr().out)
    assert len(sa["speakers"]) == 3
    assert main(["label", sa["session_path"], "spk_0=京子", "spk_1=祖父", "spk_2=サンディ", "--json"]) == 0
    capsys.readouterr()
    assert main(["transcribe", str(b), "--language", "ja", "--viz", "--json"]) == 0
    sb = json.loads(capsys.readouterr().out)
    assert [(s["label"], s["status"]) for s in sb["speakers"]] == [("祖父", "auto"), ("サンディ", "auto"), ("京子", "auto")]
```

- [ ] **Step 2: 実行** — `AQUAVOICE_API` を設定して `uv run pytest -m e2e -v -s` → 1 passed。失敗したら `sb` の `score` を見て、閾値の問題か、Avalon の話者分離の問題（話者数が 3 にならない）かを切り分けて報告する（閾値を勝手に変えない）。`meeting_b.voices.png` を開いて目で確認する。

- [ ] **Step 3: `skill/SKILL.md` を書く** — frontmatter:
```yaml
---
name: speaker-transcribe
description: 会議・打ち合わせの音声ファイルを、話者名付きで文字起こしする（Avalon API + ローカル声紋 DB）。音声/動画ファイル（m4a, mp3, wav, mp4 など）を渡されて「文字起こし」「議事録」「誰が何を話したか」を求められたときに使う。
---
```
本文は §9 の手順 1〜4 を、実際のコマンド（`speaker-transcribe transcribe "<file>" --json`、`speaker-transcribe label "<session_path>" spk_N=NAME ... --json`）で書く。あわせて、終了コード 2 のときはエラー文をそのままユーザーに伝えること（API キー未設定・ffmpeg 未導入など）、名前を尋ねるときは `examples` の時刻と発言を引用すること、`candidate` は「〜さんで合っていますか？」と確認すること、ユーザーがわからないと言った話者には `label` しないことを書く。

- [ ] **Step 4: インストール**

```bash
uv tool install --editable .
ln -s "$PWD/skill" ~/.claude/skills/speaker-transcribe
speaker-transcribe speakers list
```
期待: 最後のコマンドが終了コード 0 で、何も表示しない（DB が空）。

- [ ] **Step 5: コミット** — `git add -A && git commit -m "feat: e2e test and AI skill"`
