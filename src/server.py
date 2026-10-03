"""
Built-in HTTP search server.

Serves the web search interface and exposes a JSON API for searching.
Uses only the Python standard library — no Flask, no Django, no dependencies.

Endpoints:
  GET  /              → serves the search UI (static/index.html)
  GET  /api/search    → JSON search results  (?q=query&limit=10)
  GET  /api/stats     → corpus statistics
  GET  /static/*      → static file serving
"""

import json
import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from src.engine import SearchEngine


class SearchHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the search engine."""

    engine: Optional[SearchEngine] = None
    static_dir: str = ""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif path == "/api/search":
            self._handle_search(query)
        elif path == "/api/stats":
            self._handle_stats()
        elif path.startswith("/static/"):
            filename = path[len("/static/"):]
            self._serve_static(filename)
        else:
            self._send_error(404, "Not found")

    def _handle_search(self, query: dict) -> None:
        q = query.get("q", [""])[0]
        limit = int(query.get("limit", ["10"])[0])

        if not q.strip():
            self._send_json({"query": "", "results": [], "total": 0})
            return

        results = self.engine.search(q, limit=limit)
        data = {
            "query": q,
            "total": len(results),
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "score": round(r.score, 4),
                    "snippet": r.snippet,
                }
                for r in results
            ],
        }
        self._send_json(data)

    def _handle_stats(self) -> None:
        self._send_json(self.engine.stats())

    def _serve_file(self, filename: str, content_type: str) -> None:
        filepath = os.path.join(self.static_dir, filename)
        if not os.path.isfile(filepath):
            self._send_error(404, "File not found")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _serve_static(self, filename: str) -> None:
        ext_map = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".ico": "image/x-icon",
        }
        ext = os.path.splitext(filename)[1]
        content_type = ext_map.get(ext, "text/plain")
        filepath = os.path.join(self.static_dir, filename)

        # Prevent directory traversal
        real_static = os.path.realpath(self.static_dir)
        real_path = os.path.realpath(filepath)
        if not real_path.startswith(real_static):
            self._send_error(403, "Forbidden")
            return

        if not os.path.isfile(filepath):
            self._send_error(404, "File not found")
            return

        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        """Suppress default logging — we log our own way."""
        pass


def start_server(engine: SearchEngine, static_dir: str, port: int = 8080) -> None:
    """Start the search server on the given port."""
    SearchHandler.engine = engine
    SearchHandler.static_dir = static_dir

    server = HTTPServer(("0.0.0.0", port), SearchHandler)
    print(f"  Search engine running at http://localhost:{port}")
    print(f"  API endpoint: http://localhost:{port}/api/search?q=your+query")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()
