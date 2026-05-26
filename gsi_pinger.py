#listens for CS2 Game State Integration snapshots

from http.server import BaseHTTPRequestHandler, HTTPServer
import json

TOKEN = "odM6BOq8stAsOpRJK4hb"
PORT = 3000

_snapshot = None


class _GSIServer(HTTPServer):
    def __init__(self, address, token, handler):
        self.auth_token = token
        super().__init__(address, handler)


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _snapshot
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        if self._auth(payload):
            _snapshot = payload
            raise KeyboardInterrupt

    def _auth(self, payload):
        return payload.get("auth", {}).get("token") == self.server.auth_token

    def log_message(self, *args):
        pass

#Public function to start the server and get a snapshot
def get_snapshot():
    global _snapshot
    _snapshot = None
    server = _GSIServer(("localhost", PORT), TOKEN, _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return _snapshot
