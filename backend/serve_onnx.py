#!/usr/bin/env python3
"""
ONNX + Frontend Static Server
==============================
Serves the repo root (index.html + js/ + css/ + models/) with CORS headers.

This is needed for browser-based ONNX inference:
  • fetch('models/yolov8_sar.onnx') requires HTTP (not file://)
  • onnxruntime-web blocks cross-origin requests without CORS headers

Usage:
    python backend/serve_onnx.py           # port 5500
    python backend/serve_onnx.py --port 8080

Then open:  http://localhost:5500
"""

import sys
import argparse
import http.server
import socketserver
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.resolve()


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files with CORS headers so onnxruntime-web can load .onnx files."""

    # Type → MIME mapping for correct Content-Type
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".onnx": "application/octet-stream",
        ".pt":   "application/octet-stream",
        ".js":   "application/javascript",
        ".css":  "text/css",
        ".html": "text/html",
        ".json": "application/json",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        # ── CORS ────────────────────────────────────────────────────────
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        # ── Security: allow SharedArrayBuffer (needed by ort-wasm SIMD) ─
        self.send_header("Cross-Origin-Opener-Policy",  "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        # Suppress the noisy per-request logs; only log errors
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="SAR frontend + ONNX model server")
    parser.add_argument("--port", type=int, default=5500, help="Port (default: 5500)")
    parser.add_argument("--host", default="0.0.0.0",     help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Serve from repo root so paths like /models/yolov8_sar.onnx work
    import os
    os.chdir(REPO_ROOT)

    handler = CORSHTTPRequestHandler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"\n{'═' * 50}")
        print(f"  SAR Ship Detector — Frontend Server")
        print(f"{'═' * 50}")
        print(f"  Serving:   {REPO_ROOT}")
        print(f"  URL:       http://localhost:{args.port}")
        print(f"  CORS:      enabled (required for ONNX loading)")
        print(f"{'═' * 50}")
        print(f"\n  Open http://localhost:{args.port} in your browser")
        print(f"  Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


if __name__ == "__main__":
    main()
