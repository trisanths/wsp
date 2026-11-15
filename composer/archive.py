"""Archive every request, reply, tool call, and result of a composer run.

Files go to composer/runs/<request>/<timestamp>/ as numbered JSON files,
plus log.txt. Every string passes through redact() first, so the OpenAI
key never reaches the disk.
"""

import json
import os
import time


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        out = {}
        for key in value:
            out[redact(str(key), secrets)] = redact(value[key], secrets)
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.append(redact(item, secrets))
        return out
    return value


class Archive:
    def __init__(self, runs_root, request_name, secrets, stamp=None):
        if stamp is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(runs_root, request_name, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.secrets = list(secrets)
        self.count = 0

    def write(self, kind, data):
        """Write one record. Return its path."""
        self.count += 1
        path = os.path.join(self.dir, f"{self.count:04d}_{kind}.json")
        record = {"kind": kind, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": data}
        with open(path, "w") as f:
            json.dump(redact(record, self.secrets), f, indent=1, default=str)
        return path

    def log(self, text):
        with open(os.path.join(self.dir, "log.txt"), "a") as f:
            f.write(redact(time.strftime("%H:%M:%S ") + text, self.secrets) + "\n")

    def write_named(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            json.dump(redact(data, self.secrets), f, indent=1, default=str)
        return path
