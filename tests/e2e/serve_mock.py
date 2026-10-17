"""Sahte sohbet sayfasını 127.0.0.1:8765'te her yol için sunar (ör. /c/<id>).

POST istekleri "sohbet sunucusu" gibi kaydedilir; GET /__received kaydı döndürür. Böylece testler
sayfadan gerçekten hangi metnin çıktığını (ağ kilidinden geçeni) doğrulayabilir.
"""

import http.server
import json
from pathlib import Path

PAGE = (Path(__file__).parent / "mock_chat.html").read_bytes()
RECEIVED: list[str] = []


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/__received":
            body = json.dumps(RECEIVED, ensure_ascii=False).encode()
            content_type = "application/json"
        else:
            body = PAGE
            content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        RECEIVED.append(self.rfile.read(length).decode("utf-8", "replace"))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):
        pass


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
