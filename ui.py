"""Browser board for the voting simulator.

Run:  python3 ui.py
Opens http://127.0.0.1:8000 in your browser. Ctrl+C to stop.
Set UI_PORT=8765 to use another port.

Serves board.html and one JSON endpoint, POST /api/run:
  {"voters": {"source": "random" | "cses_2017", "parties": [...], "n": 10000, "seed": 42},
   "systems": [{"name": ..., "method": "grades" | "votes", "split": "1", "levelling": "threshold"}, ...]}
All voting logic lives in election.py; this file only moves JSON around.
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from election import CSES_YEARS, PRESETS, SystemSpec, Voters, build_voters, run_board

HOST, PORT = "127.0.0.1", int(os.environ.get("UI_PORT", "8000"))
PAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board.html")

_voters_cache: "OrderedDict[str, Voters]" = OrderedDict()
_cache_lock = threading.Lock()
CACHE_SIZE = 4


def get_voters(settings: dict) -> Voters:
    """Same voter settings -> same electorate, without resimulating or rereading CSES."""
    key = json.dumps(settings, sort_keys=True)
    with _cache_lock:
        if key in _voters_cache:
            _voters_cache.move_to_end(key)
            return _voters_cache[key]
    v = build_voters(settings)
    with _cache_lock:
        _voters_cache[key] = v
        while len(_voters_cache) > CACHE_SIZE:
            _voters_cache.popitem(last=False)
    return v


def handle_run(request: dict) -> dict:
    """Pure function behind POST /api/run. Raises ValueError on bad input."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    settings = request.get("voters") or {}
    if not isinstance(settings, dict):
        raise ValueError("voters must be an object")
    systems = request.get("systems") or []
    if not isinstance(systems, list):
        raise ValueError("systems must be a list")
    specs = [SystemSpec.from_dict(s) for s in systems]
    return run_board(get_voters(settings), specs)


def presets_payload() -> dict:
    return {"presets": [p.to_dict() for p in PRESETS.values()], "cses_years": CSES_YEARS}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/api/presets":
            self._json(200, presets_payload())
            return
        if self.path.split("?")[0] != "/":
            self._json(404, {"error": "not found"})
            return
        with open(PAGE_PATH, "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/api/run":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length).decode() or "{}")
            self._json(200, handle_run(request))
        except (ValueError, FileNotFoundError, KeyError, TypeError) as e:
            self._json(400, {"error": str(e)})

    def log_message(self, fmt, *args):  # quieter console
        pass


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Voting board running at {url}  (Ctrl+C to stop)")
    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
