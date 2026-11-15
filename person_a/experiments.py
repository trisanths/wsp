# Append one line per run to experiments.jsonl so every result is traceable.
import json
import os
import time

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments.jsonl")


def log(what, config, result):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "what": what,
        "config": config,
        "result": result,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
