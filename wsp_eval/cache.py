"""Generation cache keyed by (model hash, rendered prompt).

The key also includes max_new_tokens and the backend name, because those
change the output. Stored in SQLite so several processes can share it.
"""
import hashlib
import sqlite3

from wsp_eval import config


def _connect():
    conn = sqlite3.connect(config.work_path("cache.sqlite"), timeout=60)
    conn.execute("CREATE TABLE IF NOT EXISTS generations ("
                 "key TEXT PRIMARY KEY, model_hash TEXT, backend TEXT, "
                 "max_new_tokens INTEGER, prompt TEXT, output TEXT)")
    return conn


def make_key(model_hash, backend, max_new_tokens, prompt):
    text = model_hash + "\n" + backend + "\n" + str(max_new_tokens) + "\n" + prompt
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_many(keys):
    """Return {key: output} for the keys that are cached."""
    found = {}
    conn = _connect()
    try:
        chunk_size = 500
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start:start + chunk_size]
            marks = ",".join(["?"] * len(chunk))
            rows = conn.execute("SELECT key, output FROM generations WHERE key IN (" + marks + ")", chunk)
            for key, output in rows:
                found[key] = output
    finally:
        conn.close()
    return found


def put_many(rows):
    """rows: list of (key, model_hash, backend, max_new_tokens, prompt, output)."""
    conn = _connect()
    try:
        conn.executemany("INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
