"""Persistent voiceprint database (``db.json``): each known person and their embedding samples."""

import fcntl
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np

from speaker_transcribe.errors import UsageError

DB_VERSION = 1

_WAITING = "別の speaker-transcribe が DB を使用中のため待機しています…"

Source = Literal["auto", "label"]
_SOURCES: tuple[str, ...] = get_args(Source)


@dataclass
class Sample:
    id: str
    session_sha256: str
    source: Source
    created_at: str
    embedding: np.ndarray


@dataclass
class _Person:
    created_at: str
    samples: list[Sample] = field(default_factory=list)  # in insertion order


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file in the same directory and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def locked(db_path: Path) -> Iterator[None]:
    """Hold the exclusive inter-process lock of the DB at ``db_path`` (on ``<db_path>.lock``).

    Commands hold it from loading the DB to saving it, so that one cannot overwrite what another
    saved in between. If another process holds it, say so once on stderr and wait.
    """
    lock_path = db_path.with_name(f"{db_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as fh:  # the file only carries the lock; it is never written
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(_WAITING, file=sys.stderr, flush=True)
            fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _local_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise TypeError(f"'{key}' must be a string")
    return value


def _sample_from_json(raw: dict[str, Any]) -> Sample:
    source = _text(raw, "source")
    if source not in _SOURCES:
        raise ValueError(f"unknown source '{source}'")
    embedding = np.asarray(raw["embedding"], dtype=np.float32)
    if embedding.ndim != 1:
        raise ValueError("'embedding' must be a flat list of numbers")
    return Sample(
        id=_text(raw, "id"),
        session_sha256=_text(raw, "session_sha256"),
        source=cast(Source, source),
        created_at=_text(raw, "created_at"),
        embedding=embedding,
    )


def _people_from_json(raw: dict[str, Any]) -> dict[str, _Person]:
    return {
        name: _Person(
            created_at=_text(person, "created_at"),
            samples=[_sample_from_json(s) for s in person["samples"]],
        )
        for name, person in raw.items()
    }


def _sample_to_json(s: Sample) -> dict[str, Any]:
    return {
        "id": s.id,
        "session_sha256": s.session_sha256,
        "source": s.source,
        "created_at": s.created_at,
        "embedding": s.embedding.tolist(),
    }


class SpeakerDB:
    """People and their voiceprint samples. Changes stay in memory until ``save()``."""

    def __init__(
        self,
        path: Path,
        model_id: str,
        dim: int,
        max_samples: int,
        now: Callable[[], str],
        people: dict[str, _Person],
    ) -> None:
        self._path = path
        self._model_id = model_id
        self._dim = dim
        self._max_samples = max_samples
        self._now = now
        self._people = people

    @classmethod
    def load(
        cls,
        path: Path,
        model_id: str,
        dim: int,
        max_samples: int,
        now: Callable[[], str] = _local_now,
    ) -> "SpeakerDB":
        people: dict[str, _Person] = {}
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = None
        except (OSError, UnicodeDecodeError) as e:
            raise UsageError(f"{path}: 声紋 DB を読み込めません: {e}") from e

        if text is not None:
            try:
                data = json.loads(text)
                version, file_model, file_dim = data["version"], data["model_id"], data["dim"]
                if version != DB_VERSION:
                    raise UsageError(f"{path}: 未対応の声紋 DB バージョンです: {version!r}")
                if (file_model, file_dim) != (model_id, dim):
                    raise UsageError(
                        f"{path}: 声紋 DB は別の埋め込みモデルで作られています"
                        f"（DB: {file_model}・{file_dim} 次元 / 現在: {model_id}・{dim} 次元）。"
                        "古い声紋とは比較できないため、db.json を退避して作り直してください"
                    )
                people = _people_from_json(data["people"])
            except UsageError:
                raise
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                # ValueError covers invalid JSON; the rest cover a wrong shape.
                raise UsageError(f"{path}: 声紋 DB の形式が不正です: {type(e).__name__}: {e}") from e
        return cls(path, model_id, dim, max_samples, now, people)

    def save(self) -> None:
        data = {
            "version": DB_VERSION,
            "model_id": self._model_id,
            "dim": self._dim,
            "people": {
                name: {
                    "created_at": person.created_at,
                    "samples": [_sample_to_json(s) for s in person.samples],
                }
                for name, person in self._people.items()
            },
        }
        atomic_write_text(self._path, json.dumps(data, ensure_ascii=False, indent=2))

    def people(self) -> dict[str, list[Sample]]:
        return {name: list(person.samples) for name, person in self._people.items()}

    def embeddings(self) -> dict[str, list[np.ndarray]]:
        return {
            name: [s.embedding for s in person.samples] for name, person in self._people.items()
        }

    def add_sample(
        self, name: str, embedding: np.ndarray, session_sha256: str, source: str
    ) -> str:
        """Add a voiceprint sample for ``name`` (creating the person) and return its new id.

        The oldest samples (by ``created_at``, then insertion order) are dropped once the
        person has more than ``max_samples``.
        """
        if source not in _SOURCES:
            raise ValueError(f"unknown sample source: {source!r}")
        now = self._now()
        person = self._people.get(name)
        if person is None:
            person = self._people[name] = _Person(created_at=now)
        sample = Sample(
            id=str(uuid.uuid4()),
            session_sha256=session_sha256,
            source=cast(Source, source),
            created_at=now,
            embedding=np.array(embedding, dtype=np.float32),
        )
        person.samples.append(sample)
        while len(person.samples) > self._max_samples:
            # min() returns the first of equal keys, i.e. the earliest inserted.
            oldest = min(range(len(person.samples)), key=lambda i: person.samples[i].created_at)
            del person.samples[oldest]
        return sample.id

    def _remove_where(self, drop: Callable[[Sample], bool]) -> dict[str, list[Sample]]:
        """Remove the matching samples (and anyone left without samples); return them by person."""
        removed: dict[str, list[Sample]] = {}
        for name in list(self._people):
            kept: list[Sample] = []
            dropped: list[Sample] = []
            for sample in self._people[name].samples:
                (dropped if drop(sample) else kept).append(sample)
            if dropped:
                removed[name] = dropped
            if kept:
                self._people[name].samples = kept
            else:
                del self._people[name]
        return removed

    def remove_samples(self, ids: Iterable[str]) -> None:
        """Remove the samples with the given ids (unknown ids are ignored)."""
        wanted = set(ids)
        self._remove_where(lambda s: s.id in wanted)

    def remove_by_session(self, session_sha256: str) -> dict[str, list[Sample]]:
        """Remove every sample taken from that session; return them by the person they belonged to."""
        return self._remove_where(lambda s: s.session_sha256 == session_sha256)

    def rename(self, old: str, new: str) -> None:
        if old not in self._people:
            raise UsageError(f"声紋 DB に「{old}」という人はいません")
        if new == old:
            return
        if new in self._people:
            raise UsageError(f"声紋 DB に「{new}」という人が既にいます（「{old}」から改名できません）")
        # Rebuild the dict so the person keeps their position.
        self._people = {(new if n == old else n): p for n, p in self._people.items()}

    def delete(self, name: str) -> None:
        if name not in self._people:
            raise UsageError(f"声紋 DB に「{name}」という人はいません")
        del self._people[name]
