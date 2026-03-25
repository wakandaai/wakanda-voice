#!/usr/bin/env python3
"""
Simple token server + frontend file server.

Serves the browser frontend and generates LiveKit access tokens.

Usage:
    python scripts/token_server.py
    # Then open http://localhost:7881
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parent.parent))

from livekit import api

logger = logging.getLogger(__name__)


class TokenAndFrontendHandler(SimpleHTTPRequestHandler):
    api_key = "devkey"
    api_secret = "secret"
    frontend_dir = str(Path(__file__).parent.parent / "frontend")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.frontend_dir, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/token":
            self.handle_token_request(parsed)
        else:
            super().do_GET()

    def handle_token_request(self, parsed):
        params = parse_qs(parsed.query)
        identity = params.get("identity", ["user"])[0]
        room_name = params.get("room", ["wakanda-room"])[0]

        token = (
            api.AccessToken(api_key=self.api_key, api_secret=self.api_secret)
            .with_identity(identity)
            .with_name(f"User {identity}")
            .with_grants(
                api.VideoGrants(
                    room_join=True, room=room_name,
                    can_publish=True, can_subscribe=True, can_publish_data=True,
                )
            )
            .to_jwt()
        )

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"token": token}).encode())

    def log_message(self, fmt, *args):
        logger.info(fmt % args)


def main():
    parser = argparse.ArgumentParser(description="LiveKit token server + frontend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7881)
    parser.add_argument("--api-key", default=os.environ.get("LIVEKIT_API_KEY", "devkey"))
    parser.add_argument("--api-secret", default=os.environ.get("LIVEKIT_API_SECRET", "secret"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [TOKEN] %(levelname)s %(message)s", datefmt="%H:%M:%S")

    TokenAndFrontendHandler.api_key = args.api_key
    TokenAndFrontendHandler.api_secret = args.api_secret

    server = HTTPServer((args.host, args.port), TokenAndFrontendHandler)
    logger.info(f"Frontend + token server at http://localhost:{args.port}")
    logger.info(f"Token endpoint: http://localhost:{args.port}/token?identity=user1&room=wakanda-room")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
