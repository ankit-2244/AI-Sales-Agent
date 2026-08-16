"""Local UI server. Serves frontend/ and proxies /chat + /health to the live API."""

from __future__ import annotations

import json
import ssl
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl._create_unverified_context()
        return ctx

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
UPSTREAM = "https://ai-sales-agent.fastapicloud.dev"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._proxy("GET", "/health", b"")
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/chat":
            self._send(404, json.dumps({"detail": "not found"}).encode())
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self._proxy("POST", "/chat", body)

    def _proxy(self, method: str, path: str, body: bytes) -> None:
        req = Request(
            UPSTREAM + path,
            data=body if method == "POST" else None,
            method=method,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 AI-Sales-Agent-UI",
            },
        )
        try:
            with urlopen(req, timeout=60, context=_ssl_context()) as resp:
                self._send(resp.status, resp.read(), resp.headers.get("Content-Type") or "application/json")
        except HTTPError as exc:
            self._send(exc.code, exc.read() or str(exc).encode())
        except URLError as exc:
            self._send(502, json.dumps({"detail": str(exc.reason)}).encode())


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 5500), Handler)
    print("Open  http://localhost:5500")
    print("API    proxied to", UPSTREAM)
    server.serve_forever()


if __name__ == "__main__":
    main()
