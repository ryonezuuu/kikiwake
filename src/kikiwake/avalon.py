import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from kikiwake.config import Settings
from kikiwake.errors import AppError, UsageError

API_KEY_ENVS = ("AQUAVOICE_API", "AQUAVOICE_API_KEY", "AQUA_API_KEY")

_JOBS_PATH = "/v1/audio/transcription-jobs"
_FAILED_STATUSES = frozenset({"failed", "cancelled", "abandoned"})
_POLL_GROWTH = 1.5
_ERROR_BODY_CHARS = 300


def get_api_key(env: Mapping[str, str] = os.environ) -> str:
    for name in API_KEY_ENVS:
        value = env.get(name, "").strip()
        if value:
            return value
    raise UsageError(
        "Avalon の API キーが見つかりません。次のいずれかの環境変数を設定してください: "
        + ", ".join(API_KEY_ENVS)
    )


def _log(message: str) -> None:
    print(f"[avalon] {message}", file=sys.stderr)


def _json_object(response: httpx.Response, what: str) -> dict:
    try:
        data = response.json()
    except ValueError as e:
        raise AppError(f"Avalon の応答が JSON ではありません: {what}") from e
    if not isinstance(data, dict):
        raise AppError(f"Avalon の応答形式が想定外です: {what}")
    return data


def _job_object(response: httpx.Response, what: str) -> dict:
    job = _json_object(response, what)
    for key in ("id", "status"):
        if not isinstance(job.get(key), str):
            raise AppError(f"Avalon の応答形式が想定外です: {what}の {key}")
    return job


class AvalonClient:
    """Client for Avalon's asynchronous transcription jobs (submit, poll, fetch)."""

    def __init__(
        self,
        api_key: str,
        settings: Settings,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._settings = settings
        # Uploads can be ~100 MB, so allow a long write phase.
        self._http = http or httpx.Client(timeout=httpx.Timeout(60.0, write=600.0))
        self._sleep = sleep
        self._monotonic = monotonic

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
        max_speakers: int | None = None,
    ) -> dict:
        """Transcribe ``audio_path`` with diarization and return the raw ``diarized_json``."""
        fields = {
            "model": self._settings.model,
            "diarize": "true",
            "response_format": "diarized_json",
        }
        if language is not None:
            fields["language"] = language
        if prompt is not None:
            fields["prompt"] = prompt
        if max_speakers is not None:
            fields["max_speakers"] = str(max_speakers)

        job_id, status = self._submit(audio_path, fields, str(uuid.uuid4()))
        return self._wait_for_result(job_id, status)

    def _redact(self, text: str) -> str:
        """``text`` from the server, with the API key masked in case the server echoed it."""
        return text.replace(self._api_key, "***") if self._api_key else text

    def _url(self, path: str) -> str:
        return self._settings.base_url.rstrip("/") + path

    def _submit(
        self, audio_path: Path, fields: dict[str, str], idempotency_key: str
    ) -> tuple[str, str]:
        _log(f"アップロード中: {audio_path.name}")
        headers = {**self._headers, "Idempotency-Key": idempotency_key}

        def send() -> httpx.Response:
            # Reopened on every attempt: a retry must re-send the whole file.
            with audio_path.open("rb") as fh:
                return self._http.post(
                    self._url(_JOBS_PATH),
                    headers=headers,
                    data=fields,
                    files={"file": (audio_path.name, fh, "audio/mpeg")},
                )

        job = _job_object(self._request(send), "ジョブ投入の応答")
        _log(f"ジョブ投入: {job['id']} ({job['status']})")
        return job["id"], job["status"]

    def _wait_for_result(self, job_id: str, status: str) -> dict:
        settings = self._settings
        status_url = self._url(f"{_JOBS_PATH}/{job_id}")
        result_url = self._url(f"{_JOBS_PATH}/{job_id}/result")
        started = self._monotonic()
        interval = settings.poll_initial
        while True:
            if self._monotonic() - started > settings.job_timeout:
                raise AppError(
                    f"ジョブ {job_id} が {settings.job_timeout:g} 秒以内に完了しませんでした"
                    f"（最後の状態: {status}）"
                )
            self._sleep(interval)
            interval = min(interval * _POLL_GROWTH, settings.poll_max)

            job = _job_object(
                self._request(lambda: self._http.get(status_url, headers=self._headers)),
                "ジョブ状態の応答",
            )
            if job["status"] != status:
                status = job["status"]
                _log(f"ジョブ {job_id}: {status}")

            if status in _FAILED_STATUSES:
                error_code = self._redact(str(job.get("error_code") or "(なし)"))
                raise AppError(
                    f"Avalon ジョブ {job_id} が完了しませんでした: "
                    f"status={status}, error_code={error_code}"
                )
            if status == "succeeded":
                response = self._request(
                    lambda: self._http.get(result_url, headers=self._headers), accept=(409,)
                )
                if response.status_code != 409:  # 409 job_not_complete: poll again
                    return _json_object(response, "文字起こし結果")

    def _request(
        self, send: Callable[[], httpx.Response], *, accept: tuple[int, ...] = ()
    ) -> httpx.Response:
        """Run ``send``, retrying 429 / 5xx / transport errors with exponential backoff.

        Returns a 2xx response, or one whose status is in ``accept``. Any other
        4xx fails immediately.
        """
        max_retries = self._settings.max_retries
        failure = ""
        for attempt in range(max_retries + 1):
            try:
                response = send()
            except httpx.TransportError as e:
                failure = type(e).__name__
            else:
                status = response.status_code
                if 200 <= status < 300 or status in accept:
                    return response
                if status != 429 and status < 500:
                    # Masked before the cut, so no part of the key can be left at the end.
                    body = self._redact(response.text)[:_ERROR_BODY_CHARS]
                    raise AppError(f"Avalon API エラー HTTP {status}: {body}")
                failure = f"HTTP {status}"
            if attempt < max_retries:
                wait = 2 ** (attempt + 1)
                _log(f"{failure}、{wait} 秒後にリトライします ({attempt + 1}/{max_retries})")
                self._sleep(wait)
        raise AppError(
            f"Avalon API へのリクエストが {max_retries} 回のリトライ後も失敗しました"
            f"（最後のエラー: {failure}）"
        )
