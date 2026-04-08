from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

BRIDGE_HOST = os.environ.get("HDRTVNET_TAB_BRIDGE_HOST", "127.0.0.1").strip() or "127.0.0.1"
BRIDGE_PORT = max(1024, int(os.environ.get("HDRTVNET_TAB_BRIDGE_PORT", "39091")))
_SESSION_TIMEOUT_S = max(10.0, float(os.environ.get("HDRTVNET_TAB_SESSION_TIMEOUT_S", "45.0")))
_CLEANUP_INTERVAL_S = 1.0
_REVOKED_SESSION_TTL_S = max(5.0, float(os.environ.get("HDRTVNET_TAB_REVOKED_SESSION_TTL_S", "30.0")))


class SessionClosedError(RuntimeError):
    pass


@dataclass(slots=True)
class BrowserTabSessionInfo:
    session_id: str
    title: str
    browser_name: str = ""
    process_name: str = ""
    source_url: str = ""
    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_audio: bool = False
    audio_sample_rate: int = 0
    audio_channels: int = 0
    audio_bits_per_sample: int = 16
    last_seen_perf: float = 0.0


class _BridgeSession:
    def __init__(self, info: BrowserTabSessionInfo):
        self.info = info
        self.last_seen_perf = float(info.last_seen_perf or time.perf_counter())
        self.closed = False
        self.cond = threading.Condition()


class _BridgeHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, manager):
        self._bridge_manager = manager
        super().__init__(server_address, request_handler_class)

    def process_request(self, request, client_address):
        manager = getattr(self, "_bridge_manager", None)
        if manager is not None and manager._shutdown_requested.is_set():
            try:
                self.shutdown_request(request)
            except Exception:
                self.close_request(request)
            return
        try:
            super().process_request(request, client_address)
        except RuntimeError as exc:
            if (
                manager is not None
                and manager._shutdown_requested.is_set()
                and "can't create new thread" in str(exc).lower()
            ):
                try:
                    self.shutdown_request(request)
                except Exception:
                    self.close_request(request)
                return
            raise


class BrowserTabBridgeManager:
    def __init__(self, host: str = BRIDGE_HOST, port: int = BRIDGE_PORT):
        self.host = str(host)
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._cleanup_stop = threading.Event()
        self._shutdown_requested = threading.Event()
        self._lock = threading.RLock()
        self._sessions: dict[str, _BridgeSession] = {}
        self._revoked_sessions: dict[str, float] = {}

    def ensure_running(self) -> "BrowserTabBridgeManager":
        with self._lock:
            if self._server is not None:
                return self
            self._shutdown_requested.clear()
            server = _BridgeHTTPServer((self.host, self.port), self._make_handler(), self)
            server.daemon_threads = True
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever,
                name="browser-tab-bridge-http",
                daemon=True,
            )
            self._thread.start()
            self._cleanup_stop.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="browser-tab-bridge-cleanup",
                daemon=True,
            )
            self._cleanup_thread.start()
        return self

    def shutdown(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            cleanup_thread = self._cleanup_thread
            sessions = list(self._sessions.values())
            self._server = None
            self._thread = None
            self._cleanup_thread = None
            self._sessions = {}
            self._shutdown_requested.set()
            self._cleanup_stop.set()
        for session in sessions:
            try:
                session.closed = True
                with session.cond:
                    session.cond.notify_all()
            except Exception:
                continue
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        if cleanup_thread is not None and cleanup_thread.is_alive():
            try:
                cleanup_thread.join(timeout=2.0)
            except Exception:
                pass

    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(_CLEANUP_INTERVAL_S):
            cutoff = time.perf_counter() - _SESSION_TIMEOUT_S
            revoked_cutoff = time.perf_counter() - _REVOKED_SESSION_TTL_S
            doomed: list[str] = []
            with self._lock:
                for session_id, session in self._sessions.items():
                    if session.closed or session.last_seen_perf < cutoff:
                        doomed.append(session_id)
                stale_revoked = [
                    session_id
                    for session_id, revoked_t in self._revoked_sessions.items()
                    if float(revoked_t or 0.0) < revoked_cutoff
                ]
                for session_id in stale_revoked:
                    self._revoked_sessions.pop(session_id, None)
            for session_id in doomed:
                self.close_session(session_id, revoke=False)

    def upsert_session(
        self,
        payload: dict[str, Any],
        *,
        allow_create: bool = True,
    ) -> BrowserTabSessionInfo:
        now = time.perf_counter()
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            if not allow_create:
                raise SessionClosedError("capture session is not active")
            session_id = uuid.uuid4().hex
        info = BrowserTabSessionInfo(
            session_id=session_id,
            title=str(payload.get("title") or "").strip() or "Browser Tab",
            browser_name=str(payload.get("browser_name") or "").strip(),
            process_name=str(payload.get("process_name") or "").strip(),
            source_url=str(payload.get("source_url") or "").strip(),
            width=max(0, int(payload.get("width") or 0)),
            height=max(0, int(payload.get("height") or 0)),
            fps=max(1.0, float(payload.get("fps") or 30.0)),
            has_audio=bool(payload.get("has_audio", False)),
            audio_sample_rate=max(0, int(payload.get("audio_sample_rate") or 0)),
            audio_channels=max(0, int(payload.get("audio_channels") or 0)),
            audio_bits_per_sample=max(8, int(payload.get("audio_bits_per_sample") or 16)),
            last_seen_perf=now,
        )
        with self._lock:
            if allow_create:
                self._revoked_sessions.pop(session_id, None)
            elif session_id in self._revoked_sessions:
                raise SessionClosedError("capture session was stopped by HDRTVNet++")
            session = self._sessions.get(session_id)
            if session is None:
                session = _BridgeSession(info)
                self._sessions[session_id] = session
            else:
                session.info = info
                session.last_seen_perf = now
                session.closed = False
                with session.cond:
                    session.cond.notify_all()
        return info

    def touch_session(
        self,
        session_id: str,
        payload: dict[str, Any] | None = None,
    ) -> BrowserTabSessionInfo:
        next_payload = dict(payload or {})
        next_payload["session_id"] = str(session_id or "").strip()
        return self.upsert_session(next_payload, allow_create=False)

    def close_session(self, session_id: str, *, revoke: bool = True) -> None:
        session_id = str(session_id or "").strip()
        if not session_id:
            return
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if revoke:
                self._revoked_sessions[session_id] = time.perf_counter()
        if session is None:
            return
        session.closed = True
        with session.cond:
            session.cond.notify_all()

    def get_session(self, session_id: str) -> BrowserTabSessionInfo | None:
        session_id = str(session_id or "").strip()
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            info = session.info
            return BrowserTabSessionInfo(
                session_id=info.session_id,
                title=info.title,
                browser_name=info.browser_name,
                process_name=info.process_name,
                source_url=info.source_url,
                width=info.width,
                height=info.height,
                fps=info.fps,
                has_audio=info.has_audio,
                audio_sample_rate=info.audio_sample_rate,
                audio_channels=info.audio_channels,
                audio_bits_per_sample=info.audio_bits_per_sample,
                last_seen_perf=session.last_seen_perf,
            )

    def list_sessions(self) -> list[BrowserTabSessionInfo]:
        with self._lock:
            items = [
                BrowserTabSessionInfo(
                    session_id=session.info.session_id,
                    title=session.info.title,
                    browser_name=session.info.browser_name,
                    process_name=session.info.process_name,
                    source_url=session.info.source_url,
                    width=session.info.width,
                    height=session.info.height,
                    fps=session.info.fps,
                    has_audio=session.info.has_audio,
                    audio_sample_rate=session.info.audio_sample_rate,
                    audio_channels=session.info.audio_channels,
                    audio_bits_per_sample=session.info.audio_bits_per_sample,
                    last_seen_perf=session.last_seen_perf,
                )
                for session in self._sessions.values()
                if not session.closed
            ]
        items.sort(
            key=lambda item: (
                str(item.browser_name).lower(),
                str(item.title).lower(),
                str(item.source_url).lower(),
            )
        )
        return items

    def _make_handler(self):
        manager = self

        class _Handler(BaseHTTPRequestHandler):
            server_version = "HDRTVNetTabBridge/2.0"

            def log_message(self, _format, *args):
                del _format, args
                return

            def end_headers(self):
                origin = str(self.headers.get("Origin") or "").strip()
                if origin:
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Vary", "Origin")
                else:
                    self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type",
                )
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Max-Age", "600")
                if (
                    str(self.headers.get("Access-Control-Request-Private-Network") or "")
                    .strip()
                    .lower()
                    == "true"
                ):
                    self.send_header("Access-Control-Allow-Private-Network", "true")
                super().end_headers()

            def do_OPTIONS(self):
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "bridge_url": manager.address(),
                            "session_count": len(manager.list_sessions()),
                        },
                    )
                    return
                if parsed.path == "/sessions":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "sessions": [
                                {
                                    "session_id": item.session_id,
                                    "title": item.title,
                                    "browser_name": item.browser_name,
                                    "process_name": item.process_name,
                                    "source_url": item.source_url,
                                    "width": item.width,
                                    "height": item.height,
                                    "fps": item.fps,
                                    "has_audio": item.has_audio,
                                    "audio_sample_rate": item.audio_sample_rate,
                                    "audio_channels": item.audio_channels,
                                    "audio_bits_per_sample": item.audio_bits_per_sample,
                                    "last_seen_perf": item.last_seen_perf,
                                }
                                for item in manager.list_sessions()
                            ],
                        },
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

            def do_POST(self):
                parsed = urlparse(self.path)
                if parsed.path == "/session/start":
                    payload = self._read_json_body()
                    if payload is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
                        return
                    info = manager.upsert_session(payload)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "session_id": info.session_id,
                            "bridge_url": manager.address(),
                        },
                    )
                    return

                stop_match = re.fullmatch(r"/session/([^/]+)/stop", parsed.path)
                if stop_match:
                    manager.close_session(stop_match.group(1))
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return

                keepalive_match = re.fullmatch(r"/session/([^/]+)/keepalive", parsed.path)
                if keepalive_match:
                    payload = self._read_json_body()
                    if payload is None:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
                        return
                    try:
                        info = manager.touch_session(keepalive_match.group(1), payload)
                    except SessionClosedError as exc:
                        self._send_json(HTTPStatus.GONE, {"ok": False, "error": str(exc)})
                        return
                    except Exception as exc:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                        return
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "session_id": info.session_id,
                            "has_audio": info.has_audio,
                        },
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

            def _read_binary_body(self) -> bytes:
                try:
                    content_length = max(0, int(self.headers.get("Content-Length", "0")))
                except Exception:
                    content_length = 0
                return self.rfile.read(content_length) if content_length > 0 else b""

            def _read_json_body(self) -> dict[str, Any] | None:
                raw = self._read_binary_body()
                if not raw:
                    return {}
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    return None
                return payload if isinstance(payload, dict) else None

            def _send_json(
                self,
                status: int,
                payload: dict[str, Any],
            ) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return _Handler


_MANAGER = BrowserTabBridgeManager()


def ensure_browser_tab_bridge_running() -> BrowserTabBridgeManager:
    return _MANAGER.ensure_running()


def browser_tab_bridge_url() -> str:
    return ensure_browser_tab_bridge_running().address()


def list_browser_tab_sessions() -> list[BrowserTabSessionInfo]:
    return ensure_browser_tab_bridge_running().list_sessions()


def get_browser_tab_session(session_id: str) -> BrowserTabSessionInfo | None:
    return ensure_browser_tab_bridge_running().get_session(session_id)


def close_browser_tab_session(session_id: str) -> None:
    ensure_browser_tab_bridge_running().close_session(session_id)


def close_browser_tab_bridge() -> None:
    _MANAGER.shutdown()


atexit.register(close_browser_tab_bridge)
