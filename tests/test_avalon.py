import httpx
import pytest
from conftest import raw_diarized

from kikiwake.avalon import AvalonClient, get_api_key
from kikiwake.config import Settings
from kikiwake.errors import AppError, UsageError

KEY = "secret-key-123"
JOBS = "/v1/audio/transcription-jobs"
AUDIO_BYTES = b"FAKE-MP3-BYTES"
DIARIZED = raw_diarized([("spk_0", 0.0, 1.0, 0.9, False), ("spk_1", 1.5, 2.5, 0.9, False)])


def reply(status: int, payload: object = None, *, text: str | None = None) -> httpx.Response:
    if text is not None:
        return httpx.Response(status, text=text)
    return httpx.Response(status, json=payload if payload is not None else {})


def job(status: str, *, error_code: str | None = None) -> httpx.Response:
    # result_url deliberately points elsewhere: the client must not follow it.
    body = {
        "id": "j1",
        "status": status,
        "error_code": error_code,
        "result_url": "https://elsewhere.example/ignored",
    }
    return reply(200, body)


class FakeApi:
    """Scripted Avalon server for job ``j1``.

    Each route has a queue of items served in request order: an ``httpx.Response``
    or an exception to raise. Asking for more than was scripted fails the test.
    """

    def __init__(self, *, post=(), status=(), result=()):
        self.queues = {
            ("POST", JOBS): list(post),
            ("GET", f"{JOBS}/j1"): list(status),
            ("GET", f"{JOBS}/j1/result"): list(result),
        }
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = (request.method, request.url.path)
        if route not in self.queues:
            raise AssertionError(f"unexpected request {route}")
        if not self.queues[route]:
            raise AssertionError(f"script exhausted for {route}")
        item = self.queues[route].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _on(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if (r.method, r.url.path) == (method, path)]

    @property
    def posts(self) -> list[httpx.Request]:
        return self._on("POST", JOBS)

    @property
    def status_gets(self) -> list[httpx.Request]:
        return self._on("GET", f"{JOBS}/j1")

    @property
    def result_gets(self) -> list[httpx.Request]:
        return self._on("GET", f"{JOBS}/j1/result")


class FakeClock:
    """``sleep`` records the wait and advances ``monotonic`` by the same amount."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(AUDIO_BYTES)
    return path


@pytest.fixture
def clock():
    return FakeClock()


def make_client(api: FakeApi, clock: FakeClock, settings: Settings | None = None) -> AvalonClient:
    return AvalonClient(
        KEY,
        settings or Settings(),
        http=httpx.Client(transport=httpx.MockTransport(api.handler)),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )


def test_get_api_key_priority():
    assert get_api_key({"AQUA_API_KEY": "c", "AQUAVOICE_API_KEY": "b"}) == "b"
    assert get_api_key({"AQUA_API_KEY": "c", "AQUAVOICE_API_KEY": "b", "AQUAVOICE_API": "a"}) == "a"


def test_get_api_key_skips_empty_values():
    assert get_api_key({"AQUAVOICE_API": "", "AQUAVOICE_API_KEY": "  ", "AQUA_API_KEY": "c"}) == "c"


def test_get_api_key_missing():
    with pytest.raises(UsageError, match="AQUAVOICE_API") as excinfo:
        get_api_key({})
    assert "AQUAVOICE_API_KEY" in str(excinfo.value) and "AQUA_API_KEY" in str(excinfo.value)


def test_job_flow_returns_result(audio, clock, capsys):
    api = FakeApi(
        post=[job("queued")],
        status=[job("queued"), job("succeeded")],
        result=[reply(200, DIARIZED)],
    )
    assert make_client(api, clock).transcribe(audio) == DIARIZED

    assert [(r.method, r.url.path) for r in api.requests] == [
        ("POST", JOBS),
        ("GET", f"{JOBS}/j1"),
        ("GET", f"{JOBS}/j1"),
        ("GET", f"{JOBS}/j1/result"),
    ]
    assert {r.url.host for r in api.requests} == {"api.aquavoice.com"}
    assert all(r.headers["Authorization"] == f"Bearer {KEY}" for r in api.requests)
    post = api.posts[0]
    assert "Idempotency-Key" in post.headers
    assert b'name="diarize"' in post.content and b"diarized_json" in post.content
    assert b"avalon-v1.5" in post.content
    assert b'filename="meeting.mp3"' in post.content and AUDIO_BYTES in post.content
    assert b"audio/mpeg" in post.content
    assert clock.sleeps == [5.0, 7.5]

    err = capsys.readouterr().err
    assert "j1" in err and "succeeded" in err and KEY not in err


def test_poll_interval_capped(audio, clock):
    api = FakeApi(
        post=[job("queued")],
        status=[job("queued")] * 5 + [job("succeeded")],
        result=[reply(200, DIARIZED)],
    )
    make_client(api, clock).transcribe(audio)
    assert clock.sleeps == [5.0, 7.5, 11.25, 15.0, 15.0, 15.0]


def test_result_409_keeps_polling(audio, clock):
    api = FakeApi(
        post=[job("queued")],
        status=[job("succeeded"), job("succeeded")],
        result=[reply(409, {"error": {"code": "job_not_complete"}}), reply(200, DIARIZED)],
    )
    assert make_client(api, clock).transcribe(audio) == DIARIZED
    assert len(api.status_gets) == 2 and len(api.result_gets) == 2
    assert clock.sleeps == [5.0, 7.5]


def test_retry_on_429_reuses_idempotency_key(audio, clock):
    api = FakeApi(
        post=[reply(429), reply(503), job("queued")],
        status=[job("succeeded")],
        result=[reply(200, DIARIZED)],
    )
    assert make_client(api, clock).transcribe(audio) == DIARIZED
    assert clock.sleeps[:2] == [2, 4]
    assert len(api.posts) == 3
    assert len({r.headers["Idempotency-Key"] for r in api.posts}) == 1
    # The upload is re-sent in full on every attempt.
    assert all(AUDIO_BYTES in r.content for r in api.posts)


def test_idempotency_key_differs_between_transcriptions(audio, clock):
    api = FakeApi(
        post=[job("queued"), job("queued")],
        status=[job("succeeded"), job("succeeded")],
        result=[reply(200, DIARIZED), reply(200, DIARIZED)],
    )
    client = make_client(api, clock)
    client.transcribe(audio)
    client.transcribe(audio)
    assert len({r.headers["Idempotency-Key"] for r in api.posts}) == 2


def test_retry_exhausted(audio, clock):
    api = FakeApi(post=[reply(500, text="boom") for _ in range(6)])
    with pytest.raises(AppError, match="500"):
        make_client(api, clock).transcribe(audio)
    assert len(api.posts) == 6
    assert clock.sleeps == [2, 4, 8, 16, 32]


def test_transport_error_is_retried_and_named_when_exhausted(audio, clock):
    api = FakeApi(post=[httpx.ConnectError("refused") for _ in range(6)])
    with pytest.raises(AppError, match="ConnectError"):
        make_client(api, clock).transcribe(audio)
    assert len(api.posts) == 6

    api = FakeApi(
        post=[httpx.ReadTimeout("slow"), job("queued")],
        status=[job("succeeded")],
        result=[reply(200, DIARIZED)],
    )
    assert make_client(api, FakeClock()).transcribe(audio) == DIARIZED
    assert len(api.posts) == 2


def test_status_and_result_requests_are_retried_too(audio, clock):
    api = FakeApi(
        post=[job("queued")],
        status=[reply(503), job("succeeded")],
        result=[httpx.ReadError("reset"), reply(502), reply(200, DIARIZED)],
    )
    assert make_client(api, clock).transcribe(audio) == DIARIZED
    # poll wait, status retry wait, then result retry waits (the backoff restarts per request).
    assert clock.sleeps == [5.0, 2, 2, 4]


def test_max_retries_setting_is_honoured(audio, clock):
    api = FakeApi(post=[reply(500) for _ in range(3)])
    with pytest.raises(AppError, match="500"):
        make_client(api, clock, Settings(max_retries=2)).transcribe(audio)
    assert len(api.posts) == 3
    assert clock.sleeps == [2, 4]


def test_4xx_not_retried(audio, clock):
    api = FakeApi(post=[reply(401, {"error": {"message": "invalid api key"}})])
    with pytest.raises(AppError, match="401") as excinfo:
        make_client(api, clock).transcribe(audio)
    assert len(api.posts) == 1 and clock.sleeps == []
    message = str(excinfo.value)
    assert message.startswith("Avalon API エラー HTTP 401: ") and "invalid api key" in message
    assert KEY not in message


def test_4xx_message_never_shows_the_api_key(audio, clock):
    body = f'{{"error": {{"message": "invalid api key: {KEY}"}}}}'
    assert body.index(KEY) == 40
    api = FakeApi(post=[reply(401, text="x" * 255 + body)])  # the key straddles the 300-char cut
    with pytest.raises(AppError) as excinfo:
        make_client(api, clock).transcribe(audio)
    message = str(excinfo.value)
    assert KEY[:5] not in message and "***" in message


def test_failed_job_error_code_never_shows_the_api_key(audio, clock):
    api = FakeApi(post=[job("queued")], status=[job("failed", error_code=f"bad_key_{KEY}")])
    with pytest.raises(AppError) as excinfo:
        make_client(api, clock).transcribe(audio)
    assert KEY not in str(excinfo.value) and "bad_key_***" in str(excinfo.value)


def test_4xx_message_truncates_body_to_300_chars(audio, clock):
    api = FakeApi(post=[reply(400, text="x" * 1000)])
    with pytest.raises(AppError) as excinfo:
        make_client(api, clock).transcribe(audio)
    assert str(excinfo.value) == "Avalon API エラー HTTP 400: " + "x" * 300


@pytest.mark.parametrize(
    ("status", "error_code"),
    [("failed", "audio_decode_failed"), ("cancelled", "user_cancelled"), ("abandoned", None)],
)
def test_failed_job_reports_status_and_error_code(audio, clock, status, error_code):
    api = FakeApi(post=[job("queued")], status=[job(status, error_code=error_code)])
    with pytest.raises(AppError) as excinfo:
        make_client(api, clock).transcribe(audio)
    message = str(excinfo.value)
    assert "j1" in message and status in message
    if error_code:
        assert error_code in message
    assert api.result_gets == []


def test_job_timeout_mentions_job_id(audio, clock):
    api = FakeApi(post=[job("queued")], status=[job("queued")] * 400)
    with pytest.raises(AppError, match="j1") as excinfo:
        make_client(api, clock).transcribe(audio)
    assert "3600" in str(excinfo.value)
    assert clock.now - 1000.0 > 3600.0
    assert len(api.status_gets) < 400


def test_job_timeout_is_measured_from_after_submission(audio, clock):
    # 2 + 4 seconds of upload retries must not count against a 10 second job_timeout.
    api = FakeApi(
        post=[reply(503), reply(503), job("queued")],
        status=[job("queued"), job("succeeded")],
        result=[reply(200, DIARIZED)],
    )
    client = make_client(api, clock, Settings(job_timeout=10.0))
    assert client.transcribe(audio) == DIARIZED


def test_optional_fields_only_when_given(audio, clock):
    api = FakeApi(
        post=[job("queued"), job("queued")],
        status=[job("succeeded"), job("succeeded")],
        result=[reply(200, DIARIZED), reply(200, DIARIZED)],
    )
    client = make_client(api, clock)
    client.transcribe(audio)
    first = api.posts[0].content
    for field in (b"language", b"prompt", b"max_speakers"):
        assert b'name="' + field + b'"' not in first

    client.transcribe(audio, language="ja", prompt="会議の議事録", max_speakers=4)
    second = api.posts[1].content
    assert b'name="language"' in second and b"\r\n\r\nja\r\n" in second
    assert b'name="prompt"' in second and "会議の議事録".encode() in second
    assert b'name="max_speakers"' in second and b"\r\n\r\n4\r\n" in second
