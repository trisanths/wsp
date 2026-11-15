"""Serve a built hosted site locally, with the same rewrites Vercel applies.

    python3 studio/hosted/serve_dist.py [--dir studio/hosted/dist] [--port 8800]

Use it to test the exact files that deploy.sh publishes.
"""

import http.server
import json
import os
import sys


def arg(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


ROOT = os.path.abspath(arg("--dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")))
PORT = int(arg("--port", "8800"))
with open(os.path.join(ROOT, "vercel.json")) as handle:
    REWRITES = {item["source"]: item["destination"] for item in json.load(handle).get("rewrites", [])}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def translate_path(self, path):
        bare = path.split("?", 1)[0].split("#", 1)[0]
        if bare in REWRITES:
            path = REWRITES[bare]
        return super().translate_path(path)

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    # a page loads about 40 modules at once; the default backlog of 5 resets some of them
    request_queue_size = 256
    daemon_threads = True


if __name__ == "__main__":
    Server(("127.0.0.1", PORT), Handler).serve_forever()
