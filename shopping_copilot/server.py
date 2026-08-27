"""Local development HTTP adapter. No authentication or production hardening."""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from conversation import update_state
from shared import SearchState
from .pipeline import ShoppingCopilot

MAX_BODY_BYTES = 65536


def create_server(copilot: ShoppingCopilot, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            logging.getLogger(__name__).info(format, *args)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "mode": "local_baseline"})
            else:
                self.send_json(404, {"error": "Unknown route"})

        def do_POST(self) -> None:
            if self.path not in {"/search", "/turn"}:
                self.send_json(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    self.send_json(413, {"error": f"Body must be 1–{MAX_BODY_BYTES} bytes"})
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Body must be a JSON object")
                if self.path == "/search":
                    if set(body) != {"state"}:
                        raise ValueError("/search expects exactly one state field")
                    state = SearchState.from_dict(body["state"])
                else:
                    if set(body) - {"previous_state", "query", "updates"}:
                        raise ValueError("Unknown /turn fields")
                    previous = body.get("previous_state")
                    previous = SearchState.from_dict(previous) if previous is not None else None
                    state = update_state(previous, body["query"], body.get("updates"))
                self.send_json(200, copilot.search(state))
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                logging.getLogger(__name__).exception("Search failed")
                self.send_json(500, {"error": "Internal search error"})

    return ThreadingHTTPServer((host, port), Handler)
