"""A small HTTP server for chatting with a finetuned ohara model.

Built on the standard library's ``http.server`` rather than a web framework:
the whole surface is three endpoints, and a training box should not need a new
dependency stack to talk to the model it just trained.

Replies stream over Server-Sent Events, so tokens appear as they are sampled.
Generation is serialized behind a lock — there is one model and one KV cache, so
concurrent requests would interleave into each other's cache.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ohara.chat_engine import ChatEngine, SamplingConfig


STATIC_DIR = Path(__file__).parent / "static"
MAX_REQUEST_BYTES = 4 * 1024 * 1024


class ChatRequestHandler(BaseHTTPRequestHandler):
    """Serves the chat UI and streams model replies."""

    server_version = "ohara-webui"
    protocol_version = "HTTP/1.1"

    # Injected by create_server.
    engine: ChatEngine
    generation_lock: threading.Lock
    checkpoint_path: str | None
    default_sampling: SamplingConfig

    def log_message(self, format: str, *args: Any) -> None:
        # The default handler logs every asset request to stderr, which buries
        # the training output this server usually runs alongside.
        return

    # -- helpers ---------------------------------------------------------

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("request body is empty")
        if length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _serve_static(self, relative: str) -> None:
        # Resolve inside STATIC_DIR so a crafted path cannot escape the folder.
        target = (STATIC_DIR / relative).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "not found"}, status=404)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        content_type, _ = mimetypes.guess_type(target.name)
        self._send_bytes(target.read_bytes(), content_type or "application/octet-stream")

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server's required spelling
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/info":
            self._send_json(
                {
                    "model": self.engine.metadata(self.checkpoint_path),
                    "defaults": asdict(self.default_sampling),
                }
            )
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - http.server's required spelling
        if self.path.split("?", 1)[0] != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return
        try:
            payload = self._read_json()
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("'messages' must be a non-empty list")
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("each message must be an object")
                if message.get("role") not in {"user", "assistant", "system"}:
                    raise ValueError(f"unsupported role: {message.get('role')!r}")
                if not isinstance(message.get("content"), str):
                    raise ValueError("message content must be a string")
            defaults = self.default_sampling
            sampling = SamplingConfig(
                temperature=float(payload.get("temperature", defaults.temperature)),
                top_p=float(payload.get("top_p", defaults.top_p)),
                top_k=int(payload.get("top_k", defaults.top_k)),
                max_new_tokens=int(payload.get("max_new_tokens", defaults.max_new_tokens)),
            )
            seed = payload.get("seed")
            seed = int(seed) if seed is not None else None
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        # Length is unknown up front, so stream with chunked transfer encoding.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            with self.generation_lock:
                for delta in self.engine.generate_stream(messages, sampling, seed=seed):
                    self._send_event({"delta": delta})
            self._send_event({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            # The browser navigated away or hit stop; nothing left to send.
            return
        except Exception as error:  # noqa: BLE001 - surface any failure in the UI
            try:
                self._send_event({"error": f"{type(error).__name__}: {error}"})
            except (BrokenPipeError, ConnectionResetError):
                return
        finally:
            try:
                self._write_chunk(b"")
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _send_event(self, payload: dict[str, Any]) -> None:
        self._write_chunk(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))


def create_server(
    engine: ChatEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    checkpoint_path: str | None = None,
    sampling: SamplingConfig | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not start) a chat server bound to ``host:port``."""
    handler = type(
        "BoundChatRequestHandler",
        (ChatRequestHandler,),
        {
            "engine": engine,
            "generation_lock": threading.Lock(),
            "checkpoint_path": checkpoint_path,
            "default_sampling": sampling or SamplingConfig(),
        },
    )
    return ThreadingHTTPServer((host, port), handler)


def serve(
    engine: ChatEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    checkpoint_path: str | None = None,
    sampling: SamplingConfig | None = None,
) -> None:
    """Serve the chat UI until interrupted."""
    server = create_server(
        engine,
        host=host,
        port=port,
        checkpoint_path=checkpoint_path,
        sampling=sampling,
    )
    shown_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"ohara chat UI on http://{shown_host}:{port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.shutdown()
        server.server_close()
