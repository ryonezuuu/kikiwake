"""The ``kikiwake`` command: ``transcribe``, ``label``, ``speakers``, ``viz`` and ``index``."""

import argparse
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, tzinfo
from functools import partial
from pathlib import Path

from kikiwake.archive import write_index
from kikiwake.audio import load_wav, require_ffmpeg, sha256_file, to_wav16k
from kikiwake.avalon import API_KEY_FILENAME, AvalonClient, get_api_key
from kikiwake.config import Settings, data_home, load_settings, local_zone
from kikiwake.embedder import DIM, MODEL_ID, EcapaEmbedder, Embedder
from kikiwake.errors import AppError, UsageError
from kikiwake.labeling import apply_labels, clean_name
from kikiwake.pipeline import TranscribeOptions, Transcriber, check_audio, run_transcribe
from kikiwake.render import build_summary, render_markdown
from kikiwake.session import Session, load_session, save_session, session_paths
from kikiwake.speaker_db import SpeakerDB, atomic_write_text, locked
from kikiwake.viz import render_viz

_MAX_SPEAKERS = 8

Handler = Callable[[argparse.Namespace, Path, Settings], int]


def make_embedder(home: Path) -> Embedder:
    with warnings.catch_warnings():
        # speechbrain 1.1.2 passes a deprecated huggingface_hub argument while loading.
        warnings.filterwarnings(
            "ignore",
            message="The `local_dir_use_symlinks` argument is deprecated",
            category=UserWarning,
        )
        return EcapaEmbedder(home / "models")


def make_transcriber(settings: Settings, key_file: Path | None = None) -> Transcriber:
    client: AvalonClient | None = None

    def transcribe(mp3: Path, opts: TranscribeOptions) -> dict:
        nonlocal client
        if client is None:  # the key is needed only once an API call really happens
            client = AvalonClient(get_api_key(key_file=key_file), settings)
        return client.transcribe(
            mp3, language=opts.language, prompt=opts.prompt, max_speakers=opts.max_speakers
        )

    return transcribe


def _now(tz: tzinfo | None = None) -> str:
    """Now, as ISO 8601 to the second in zone ``tz`` (``None``: the system's)."""
    return datetime.now(tz).astimezone(tz).isoformat(timespec="seconds")


def _db_path(home: Path) -> Path:
    return home / "db.json"


def _load_db(home: Path, settings: Settings) -> SpeakerDB:
    """The DB as saved, for reading only (writes replace the file atomically)."""
    return SpeakerDB.load(
        _db_path(home),
        MODEL_ID,
        DIM,
        settings.max_samples_per_person,
        now=partial(_now, local_zone(settings)),  # when a sample is added
    )


@contextmanager
def _open_db(home: Path, settings: Settings) -> Iterator[SpeakerDB]:
    """Lock the DB and load it, for a change that is saved before the block ends."""
    with locked(_db_path(home)):
        yield _load_db(home, settings)


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


_NOT_SEARCHED = ("records", ".kikiwake")  # meeting folders and KIKIWAKE_HOME hold no recordings


def _records_root(folder: Path) -> Path | None:
    """``<root>`` if ``folder`` is a meeting folder ``<root>/records/<x>`` of a records folder."""
    return folder.parent.parent if folder.parent.name == "records" else None


def _find_audio(session: Session, session_path: Path) -> Path | None:
    """The recording of ``session``, or ``None`` if it cannot be found.

    The stored ``audio_path`` is where it was when transcribed (in Cowork, a path in that
    conversation's VM). If it is not there and the session is in ``<root>/records/<x>/``, the first
    file under ``<root>`` (not in ``records/`` or ``.kikiwake/``) with the same name and SHA-256.
    """
    audio = Path(session.audio_path)
    if audio.is_file():
        return audio
    root = _records_root(session_path.parent)
    if root is None:
        return None
    for folder, subfolders, files in os.walk(root):
        if folder == str(root):
            subfolders[:] = [d for d in subfolders if d not in _NOT_SEARCHED]
        subfolders.sort()
        candidate = Path(folder) / audio.name
        if audio.name in files and candidate.is_file() and sha256_file(candidate) == session.audio_sha256:
            return candidate
    return None


def _write_viz(session: Session, session_path: Path, db: SpeakerDB, png_path: Path) -> None:
    """Draw the voice picture of ``session`` (saved at ``session_path``) from its original audio."""
    audio = _find_audio(session, session_path)
    if audio is None:
        raise AppError(f"元の音声ファイルが見つかりません: {session.audio_path}")
    require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        to_wav16k(audio, wav_path)
        wave = load_wav(wav_path)
    render_viz(session, wave, db.embeddings(), png_path)


def _refresh_viz(
    session: Session, session_path: Path, db: SpeakerDB, png_path: Path
) -> Path | None:
    """Draw the voice picture after the main work is saved; failing there is only a warning.

    Returns the picture's path, or ``None`` if it could not be drawn.
    """
    try:
        _write_viz(session, session_path, db, png_path)
    except (AppError, OSError) as e:
        print(f"警告: 可視化をスキップしました: {e}", file=sys.stderr)
        return None
    return png_path


def _cmd_transcribe(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    # Before the (slow) model load and the API call, so these mistakes fail fast.
    check_audio(args.audio)
    _load_db(home, settings)  # a DB made with another model
    opts = TranscribeOptions(
        output_dir=args.output_dir,
        language=args.language,
        prompt=args.prompt,
        max_speakers=args.max_speakers,
        force_api=args.force_api,
    )
    session, markdown_path, session_path = run_transcribe(
        args.audio,
        opts,
        settings=settings,
        open_db=lambda: _open_db(home, settings),
        embedder=make_embedder(home),
        transcriber=make_transcriber(settings, key_file=home / API_KEY_FILENAME),
        now=partial(_now, local_zone(settings)),
    )
    _, _, png_path = session_paths(args.audio, args.output_dir)
    viz_path = None
    if args.viz or png_path.exists():  # an existing picture is redrawn so it never shows an old state
        viz_path = _refresh_viz(session, session_path, _load_db(home, settings), png_path)
    if args.json:
        _print_json(build_summary(session, markdown_path, session_path, viz_path))
    else:
        print(markdown_path)
    return 0


def _parse_pair(arg: str) -> tuple[str, str]:
    key, equals, name = arg.partition("=")
    if not equals:
        raise UsageError(f"KEY=NAME の形式で指定してください: {arg}")
    return key, name


def _cmd_label(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    pairs = [_parse_pair(arg) for arg in args.pairs]
    with _open_db(home, settings) as db:
        session = load_session(args.session)  # under the lock too: another `label` may be saving it
        for warning in apply_labels(session, db, pairs):
            print(f"警告: {warning}", file=sys.stderr)
        db.save()
        save_session(session, args.session)
        markdown_path, _, png_path = session_paths(Path(session.audio_path), args.session.parent)
        atomic_write_text(markdown_path, render_markdown(session))
    viz_path = _refresh_viz(session, args.session, db, png_path) if png_path.exists() else None
    if args.json:
        _print_json(build_summary(session, markdown_path, args.session, viz_path))
    else:
        print(markdown_path)
    return 0


def _cmd_speakers_list(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    people = _load_db(home, settings).people()
    rows = [
        {
            "name": name,
            "samples": len(samples),
            "updated_at": max(s.created_at for s in samples),
        }
        for name, samples in people.items()
    ]
    if args.json:
        _print_json(rows)
    else:
        for row in rows:
            print(f"{row['name']}  サンプル {row['samples']} 件  最終更新 {row['updated_at']}")
    return 0


def _cmd_speakers_rename(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    new = clean_name(args.new)  # the same rules as names given to `label`
    with _open_db(home, settings) as db:
        db.rename(args.old, new)
        db.save()
    if args.json:
        _print_json({"renamed": {"from": args.old, "to": new}})
    return 0


def _cmd_speakers_delete(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    with _open_db(home, settings) as db:
        db.delete(args.name)
        db.save()
    if args.json:
        _print_json({"deleted": args.name})
    return 0


def _cmd_viz(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    session = load_session(args.session)
    db = _load_db(home, settings)
    markdown_path, _, png_path = session_paths(Path(session.audio_path), args.session.parent)
    _write_viz(session, args.session, db, png_path)  # drawing is the whole job: a failure is an error
    if args.json:
        _print_json(build_summary(session, markdown_path, args.session, png_path))
    else:
        print(png_path)
    return 0


def _cmd_index(args: argparse.Namespace, home: Path, settings: Settings) -> int:
    if not args.root.is_dir():
        raise UsageError(f"記録フォルダが見つかりません: {args.root}")
    index_path, meetings, skipped = write_index(args.root)
    if args.json:
        _print_json({"index_path": str(index_path), "meetings": meetings, "skipped": skipped})
    else:
        print(index_path)
    return 0


def _path(text: str) -> Path:
    """An absolute path, so what is stored in a session works from any directory."""
    return Path(text).expanduser().resolve()


def _max_speakers(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定してください: {text!r}") from None
    if not 1 <= value <= _MAX_SPEAKERS:
        raise argparse.ArgumentTypeError(f"1〜{_MAX_SPEAKERS} で指定してください: {value}")
    return value


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="標準出力に機械可読の JSON だけを出す")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kikiwake",
        description="会議音声を文字起こしし、声紋 DB で話者に名前を付ける",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = commands.add_parser("transcribe", help="音声を文字起こしして話者を照合する")
    p.add_argument("audio", type=_path, metavar="AUDIO", help="音声ファイル")
    p.add_argument(
        "--output-dir", type=_path, metavar="DIR", help="出力先（既定: 音声と同じディレクトリ）"
    )
    p.add_argument("--language", metavar="CODE", help="言語コード（ISO 639-1）。既定は自動判定")
    p.add_argument("--prompt", metavar="TEXT", help="固有名詞・用語のヒント")
    p.add_argument(
        "--max-speakers",
        type=_max_speakers,
        metavar="N",
        help=f"話者数の上限（1〜{_MAX_SPEAKERS}）",
    )
    p.add_argument("--viz", action="store_true", help="処理後に声の可視化画像も作る")
    p.add_argument("--force-api", action="store_true", help="キャッシュがあっても API を呼び直す")
    _add_json(p)
    p.set_defaults(handler=_cmd_transcribe)

    p = commands.add_parser("label", help="話者に名前を付けて声紋 DB に登録する")
    p.add_argument("session", type=_path, metavar="SESSION", help="<stem>.session.json")
    p.add_argument(
        "pairs",
        nargs="+",
        metavar="KEY=NAME",
        help="KEY は話者 ID か表示ラベル（話者1、田中？ など）",
    )
    _add_json(p)
    p.set_defaults(handler=_cmd_label)

    p = commands.add_parser("speakers", help="声紋 DB の人を一覧・改名・削除する")
    people = p.add_subparsers(dest="speakers_command", metavar="ACTION", required=True)
    sp = people.add_parser("list", help="登録済みの人を一覧する")
    _add_json(sp)
    sp.set_defaults(handler=_cmd_speakers_list)
    sp = people.add_parser("rename", help="人の名前を変える")
    sp.add_argument("old", metavar="OLD")
    sp.add_argument("new", metavar="NEW")
    _add_json(sp)
    sp.set_defaults(handler=_cmd_speakers_rename)
    sp = people.add_parser("delete", help="人とその声紋を削除する")
    sp.add_argument("name", metavar="NAME")
    _add_json(sp)
    sp.set_defaults(handler=_cmd_speakers_delete)

    p = commands.add_parser("viz", help="声の可視化画像（<stem>.voices.png）を作る")
    p.add_argument("session", type=_path, metavar="SESSION", help="<stem>.session.json")
    _add_json(p)
    p.set_defaults(handler=_cmd_viz)

    p = commands.add_parser("index", help="会議の一覧（ROOT/index.md）を作る")
    p.add_argument(
        "root", type=_path, metavar="ROOT", help="記録フォルダ（records/<名前>/ を含む）"
    )
    _add_json(p)
    p.set_defaults(handler=_cmd_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse exits on --help (0) and on bad arguments (2)
        return int(e.code or 0)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    handler: Handler = args.handler
    try:
        home = data_home()
        return handler(args, home, load_settings(home))
    except AppError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return e.exit_code
    except OSError as e:  # e.g. an output directory that cannot be written
        print(f"エラー: {e}", file=sys.stderr)
        return 1
