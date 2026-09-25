"""HTTP client for the Slumbot API: one request at a time, a pause between requests, retries with
exponential backoff on network errors and 5xx answers, never more than a few attempts.

Endpoints (POST, JSON body, ``Content-Type: application/json``; paths from the official sample
client https://www.slumbot.com/sample_api.py):

    {host}/slumbot/api/new_hand   {"token": ...}          (no token on the very first request)
    {host}/slumbot/api/act        {"token": ..., "incr": "c"}

Any response may carry a new token; the client always keeps the latest one.  Login
(``/slumbot/api/login``) is optional and not implemented.

Retrying ``act`` is not idempotent: if a request reached the server but its answer was lost, the
retry sends the same move again.  The adapter catches that (the new action string must be the old
one plus our move plus Slumbot's moves only) and aborts the hand instead of playing on blind.

The transport is injectable (``transport(url, payload, timeout) -> (status, body_bytes)``), so the
tests drive the same client against an in-process mock without sockets, or against a localhost
``http.server`` through the default urllib transport.
"""
from __future__ import annotations

import http.client
import json
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Optional, Tuple

DEFAULT_HOST = "https://slumbot.com"
API_PREFIX = "/slumbot/api"
USER_AGENT = "negpluribus-slumbot-client/0.1 (research bot; one hand at a time)"

Transport = Callable[[str, dict, float], Tuple[int, bytes]]

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class SlumbotError(Exception):
    """The server answered, but with an error (``error_msg`` or a 4xx status).  Not retried."""

    def __init__(self, message: str, status: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status = status
        self.response = response


class TransportError(Exception):
    """The server could not be reached (or kept failing) after all retries."""


def _opener_for(url: str) -> urllib.request.OpenerDirector:
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy the local mock
    return urllib.request.build_opener()


def urllib_transport(url: str, payload: dict, timeout: float) -> Tuple[int, bytes]:
    """POST ``payload`` as JSON; return (HTTP status, raw body).  Network errors propagate."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with _opener_for(url).open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:  # a real answer with a non-2xx status
        try:
            body = exc.read()
        except Exception:  # pragma: no cover - body already gone
            body = b""
        return exc.code, body


class KeepAliveTransport:
    """POST over ONE persistent HTTP(S) connection per (scheme, host, port) instead of a new TCP + TLS
    handshake per request (measured 24.09.2026 against slumbot.com: the handshake is 0.44 s of a
    ~0.77-0.97 s request).  Same contract as ``urllib_transport``: returns (status, body); network
    errors propagate unchanged to the client's retry logic, and the broken connection is dropped
    so the next attempt opens a fresh one.  No retry here: ``act`` is not idempotent, and the
    client/adapter already handle a request whose answer was lost."""

    def __init__(self) -> None:
        self._conns: Dict[tuple, http.client.HTTPConnection] = {}

    def _connection(self, scheme: str, host: str, port: Optional[int], timeout: float) -> http.client.HTTPConnection:
        key = (scheme, host, port)
        conn = self._conns.get(key)
        if conn is None:
            if scheme == "https":
                conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
            self._conns[key] = conn
        elif conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn

    def _drop(self, key: tuple) -> None:
        conn = self._conns.pop(key, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - closing a dead socket
                pass

    def close(self) -> None:
        for key in list(self._conns):
            self._drop(key)

    def __call__(self, url: str, payload: dict, timeout: float) -> Tuple[int, bytes]:
        parts = urllib.parse.urlsplit(url)
        key = (parts.scheme, parts.hostname, parts.port)
        path = parts.path + ("?" + parts.query if parts.query else "")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": USER_AGENT, "Connection": "keep-alive"}
        conn = self._connection(parts.scheme, parts.hostname or "", parts.port, timeout)
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except BaseException:
            self._drop(key)
            raise
        if resp.will_close:
            self._drop(key)
        return resp.status, data


def _is_retryable_exception(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(exc, ssl.SSLCertVerificationError) or isinstance(reason, ssl.SSLCertVerificationError):
        return False  # a certificate problem does not go away by retrying
    return isinstance(exc, (OSError, http.client.HTTPException))  # URLError, timeouts, resets


class SlumbotClient:
    """Stateful client: remembers the token, paces and retries requests, counts them."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        token: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        min_interval: float = 0.5,
        transport: Optional[Transport] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Optional[random.Random] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.host = host.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.min_interval = min_interval
        self.transport = transport or KeepAliveTransport()  # one persistent connection (urllib_transport: one per request)
        self.sleep = sleep
        self.clock = clock
        self.rng = rng or random.Random()
        self.log = log
        self.n_requests = 0  # HTTP attempts, retries included
        self.n_retries = 0
        self.n_token_changes = 0
        self.last_latency = 0.0  # seconds of the last successful request (transport call only)
        self.last_token_status = ""  # of the last response: "absent", "same" or "new"
        self._last_done: Optional[float] = None

    # ------------------------------------------------------------------ API
    def new_hand(self) -> dict:
        payload: Dict[str, str] = {"token": self.token} if self.token else {}
        return self._post("new_hand", payload)

    def act(self, incr: str) -> dict:
        if not self.token:
            raise SlumbotError("act() without a token: call new_hand() first")
        return self._post("act", {"token": self.token, "incr": incr})

    # ------------------------------------------------------------- plumbing
    def url(self, endpoint: str) -> str:
        return f"{self.host}{API_PREFIX}/{endpoint}"

    def backoff_delay(self, attempt: int) -> float:
        """Wait before retry number ``attempt + 1``: base * 2^attempt, capped, with jitter in [0.5, 1]."""
        return min(self.backoff_max, self.backoff_base * (2 ** attempt)) * (0.5 + 0.5 * self.rng.random())

    def _pace(self) -> None:
        if self._last_done is None or self.min_interval <= 0:
            return
        wait = self._last_done + self.min_interval - self.clock()
        if wait > 0:
            self.sleep(wait)

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = self.url(endpoint)
        last_problem = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.n_retries += 1
            self._pace()
            self.n_requests += 1
            t0 = self.clock()
            try:
                status, body = self.transport(url, payload, self.timeout)
            except Exception as exc:  # noqa: BLE001 - classify below
                self._last_done = self.clock()
                if not _is_retryable_exception(exc):
                    raise TransportError(f"{endpoint}: {exc!r}") from exc
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                self._last_done = self.clock()
                if 200 <= status < 300:
                    self.last_latency = self._last_done - t0
                    return self._accept(endpoint, status, body)
                data = _json_or_none(body)
                message = data.get("error_msg") if isinstance(data, dict) else None
                if status not in RETRYABLE_STATUS:
                    raise SlumbotError(f"{endpoint}: HTTP {status}: {message or _snippet(body)}", status, data)
                last_problem = f"HTTP {status}: {message or _snippet(body)}"
            if attempt == self.max_retries:
                break
            delay = self.backoff_delay(attempt)
            if self.log:
                self.log(f"slumbot {endpoint}: {last_problem}; retry {attempt + 1}/{self.max_retries} in {delay:.1f}s")
            self.sleep(delay)
        raise TransportError(f"{endpoint}: giving up after {self.max_retries + 1} attempts ({last_problem})")

    def _accept(self, endpoint: str, status: int, body: bytes) -> dict:
        data = _json_or_none(body)
        if not isinstance(data, dict):
            raise SlumbotError(f"{endpoint}: HTTP {status} but no JSON object: {_snippet(body)}", status)
        if "error_msg" in data:
            raise SlumbotError(f"{endpoint}: {data['error_msg']}", status, data)
        new_token = data.get("token")
        self.last_token_status = "absent" if not new_token else ("same" if new_token == self.token else "new")
        if new_token and new_token != self.token:
            if self.token is not None:
                self.n_token_changes += 1
            self.token = new_token
        return data


def _json_or_none(body: bytes):
    try:
        return json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _snippet(body: bytes, n: int = 200) -> str:
    try:
        text = body.decode("utf-8", "replace")
    except Exception:  # pragma: no cover
        text = repr(body)
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")
