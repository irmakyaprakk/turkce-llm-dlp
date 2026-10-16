"""Sahte sohbet sayfasını 127.0.0.1:8765'te her yol için sunar (ör. /c/<id>)."""

import http.server
from pathlib import Path

PAGE = (Path(__file__).parent / "mock_chat.html").read_bytes()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
