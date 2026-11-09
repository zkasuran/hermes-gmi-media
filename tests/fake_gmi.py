"""Small scriptable local GMI server. Media are header bytes only."""
from collections import deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from urllib.parse import unquote

MEDIA = {"image": b"\x89PNG\r\n\x1a\n", "video": b"\x00\x00\x00\x18ftypmp42", "audio": b"ID3\x04\x00\x00\x00\x00\x00\x00"}


class FakeGmi:
    def __init__(self, records):
        self.models = {m["model"]: deepcopy(m) for m in records}
        self.requests = []
        self.post_replies = deque()
        self.poll_replies = deque()
        self.jobs = {}
        self.immediate = False
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.dispatch()

            def do_POST(self):
                self.dispatch()

            def dispatch(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.loads(raw) if raw else None
                path = unquote(self.path)
                with owner.lock:
                    owner.requests.append({"method": self.command, "path": path, "body": body,
                                           "Authorization": self.headers.get("Authorization")})
                    status, data = owner.reply(self.command, path, body)
                binary = isinstance(data, bytes)
                data = data if binary else json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, method, path, body):
        if path.startswith("/media/"):
            return 200, MEDIA[path.rsplit("/", 1)[1]]
        if path == "/models":
            return 200, {"model_ids": list(self.models)}
        if path.startswith("/models/"):
            record = self.models.get(path[len("/models/"):])
            return (200, {"data": record}) if record else (404, {"error": "Unknown model"})
        if method == "POST" and path == "/requests":
            if self.post_replies:
                return self.post_replies.popleft()
            rid = str(len(self.jobs) + 1)
            kind = self.models[body["model"]]["model_type"]
            result = {"request_id": rid, "status": "success",
                      "outcome": {"media_urls": [self.url + "/media/" + kind]}}
            self.jobs[rid] = result
            return 200, result if self.immediate else {"id": rid, "status": "queued"}
        if path.startswith("/requests/"):
            if self.poll_replies:
                return self.poll_replies.popleft()
            return 200, self.jobs[path.rsplit("/", 1)[1]]
        return 404, {"error": "Unknown route"}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
