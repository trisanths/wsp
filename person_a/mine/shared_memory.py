# Shared mining memory on past.dev: every teammate's mining agent logs what it tried, and every
# agent can recall what the others learned.
#
# Scoping (important): every data point goes to ONE audience, "wsp-mining", read by ONE shared
# identity, "wsp-miner". Nothing is ever sent project-wide. Part 3's swarm runs (wsp/swarm/memory.py)
# assert a cold start in their own run audiences; a project-visible point would leak into them.
# Who wrote a point is kept in its label, its metadata and its text, not in the identity.
#
# Fails soft: with no PAST_API_KEY, or if past.dev is down, record() and recall() do nothing and
# the mining run carries on.
#
#   .venv/bin/python mine/shared_memory.py check     -> set up the audience and round-trip one test point
#   .venv/bin/python mine/shared_memory.py recall "math neurons in late MLP layers"
import datetime
import hashlib
import json
import os
import sys
import time

import requests

API = "https://api.past.dev/api/v1"
AUDIENCE = "wsp-mining"
READER = "wsp-miner"
MAX_CHARS = 1500

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"), override=True)
    except ImportError:
        pass


def now_iso():
    t = datetime.datetime.now(datetime.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (t.microsecond // 1000)


def clean_id(text):
    out = ""
    for ch in str(text):
        if ch.isalnum() or ch in "._-":
            out = out + ch
        else:
            out = out + "-"
    return out[:120]


class SharedMemory:
    def __init__(self, person, model, run_id):
        load_env()
        self.key = os.environ.get("PAST_API_KEY")
        self.person = clean_id(person or "unknown")
        self.model = clean_id(model or "unknown")
        self.run_id = clean_id(run_id)
        self.enabled = self.key is not None and self.key != ""
        self.buffer = []
        self.count = 0
        self.batches = 0
        self.last_error = None
        if self.enabled:
            self.enabled = self._ensure_audience()

    # --- HTTP -------------------------------------------------------------
    def _headers(self):
        return {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}

    def _call(self, method, path, body=None, tries=4):
        pause = 1.0
        for attempt in range(tries):
            try:
                r = requests.request(method, API + path, headers=self._headers(), json=body, timeout=20)
            except requests.RequestException as e:
                self.last_error = type(e).__name__
                time.sleep(pause)
                pause = pause * 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = pause
                if r.headers.get("Retry-After"):
                    try:
                        wait = float(r.headers["Retry-After"])
                    except ValueError:
                        pass
                time.sleep(wait)
                pause = pause * 2
                continue
            return r
        return None

    def _off(self, why):
        # Turn memory off for the rest of this run; the mining run continues without it.
        self.last_error = why
        self.enabled = False
        print("[shared-memory] off:", why)

    def _ensure_audience(self):
        r = self._call("PUT", "/audiences/" + AUDIENCE,
                       {"kind": "fixed", "name": "WSP mining: shared lessons across teammates",
                        "identities": [READER]})
        if r is None or r.status_code not in [200, 201]:
            code = "no response" if r is None else str(r.status_code)
            self._off("could not create/confirm audience (" + code + ")")
            return False
        return True

    # --- write --------------------------------------------------------------
    def record(self, kind, text, meta=None):
        # kind: "attempt" (a proposal and its test result), "lesson" (end-of-run takeaway),
        # "skill" (an agent-defined skill test), "note". Text: plain sentences with the numbers in.
        if not self.enabled:
            return
        self.count = self.count + 1
        point_id = clean_id("mine-" + self.run_id + "-" + kind + "-" + str(self.count))
        header = "[" + self.person + " | " + self.model + " | " + kind + "] "
        body = header + str(text).replace("\x00", " ").strip()
        if len(body) > MAX_CHARS:
            body = body[:MAX_CHARS]
        metadata = {"person": self.person, "model": self.model, "run_id": self.run_id, "kind": kind}
        if meta:
            metadata["meta"] = json.loads(json.dumps(meta, default=str))
        self.buffer.append({
            "id": point_id,
            "label": "mining " + kind + ": " + self.model,
            "timestamp": now_iso(),
            "audience": AUDIENCE,  # never project-wide
            "identity": READER,
            "content": body,
            "metadata": metadata,
        })
        if len(self.buffer) >= 10:
            self.flush()

    def flush(self):
        if not self.enabled or len(self.buffer) == 0:
            return {"sent": 0}
        items = self.buffer
        self.buffer = []
        for item in items:
            if item.get("audience") != AUDIENCE:
                raise ValueError("refusing to send a data point outside the " + AUDIENCE + " audience")
        self.batches = self.batches + 1
        key = "mine-" + self.run_id + "-b" + str(self.batches) + "-" + \
              hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:10]
        r = self._call("POST", "/ingest/batch", {"idempotencyKey": key, "items": items})
        if r is None:
            self._off("ingest failed: no response")
            return {"sent": 0}
        if r.status_code in [401, 402, 403]:
            self._off("ingest refused (" + str(r.status_code) + ")")
            return {"sent": 0}
        if r.status_code >= 400:
            self.last_error = "batch refused " + str(r.status_code)
            print("[shared-memory] batch dropped:", r.status_code, r.text[:200])
            return {"sent": 0}
        return {"sent": len(items), "ingestion_id": r.json().get("ingestionId")}

    def wait(self, ingestion_id, timeout=30):
        start = time.time()
        while time.time() - start < timeout:
            r = self._call("GET", "/ingest/" + ingestion_id)
            if r is not None and r.status_code == 200:
                status = r.json().get("status")
                if status == "completed":
                    return True
                if status == "failed":
                    self._call("POST", "/ingest/" + ingestion_id + "/retry")
            time.sleep(1.5)
        return False

    # --- read ---------------------------------------------------------------
    def recall(self, query, limit=20, level="low"):
        # Returns [{"text", "kind", "when", "source"}], best first. [] when memory is off.
        if not self.enabled:
            return []
        r = self._call("POST", "/recall", {"query": query, "identity": READER, "level": level, "limit": limit})
        if r is None or r.status_code != 200:
            return []
        out = []
        for doc in r.json().get("results", []):
            artifact = doc.get("artifact") or {}
            if artifact.get("supersededAt"):
                continue
            if artifact.get("kind") == "mention":
                continue
            source = ""
            sources = doc.get("sources") or []
            if len(sources) > 0:
                source = sources[0].get("sourceId", "")
            out.append({"text": doc.get("content", ""), "kind": artifact.get("kind"),
                        "when": doc.get("occurredAt"), "source": source})
        return out

    def recall_text(self, query, limit=15):
        # Recall formatted as a short bulleted block for an agent prompt.
        results = self.recall(query, limit=limit)
        if len(results) == 0:
            return ""
        lines = []
        for item in results:
            lines.append("- " + item["text"][:400])
        return "\n".join(lines)

    def delete(self, ids):
        if not self.enabled or len(ids) == 0:
            return False
        r = self._call("POST", "/data-points/delete", {"ids": ids})
        return r is not None and r.status_code == 202


def check():
    mem = SharedMemory("setup-check", "none", "check-" + str(int(time.time())))
    if not mem.enabled:
        print("FAILED: memory is off:", mem.last_error or "PAST_API_KEY missing from .env")
        return
    print("audience", AUDIENCE, "ready")
    mem.record("note", "Setup check: the shared mining memory round-trip test wrote this point.")
    sent = mem.flush()
    print("sent:", sent.get("sent"))
    ok = mem.wait(sent["ingestion_id"]) if sent.get("ingestion_id") else False
    print("ingested:", ok)
    found = False
    for item in mem.recall("shared mining memory round-trip setup check", limit=10):
        if item["source"].startswith("mine-" + mem.run_id):
            found = True
    print("recalled own test point:", found)
    deleted = mem.delete(["mine-" + mem.run_id + "-note-1"])
    print("test point deleted:", deleted)
    print("RESULT:", "OK" if (ok and found and deleted) else "PROBLEM (see above)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    elif len(sys.argv) > 2 and sys.argv[1] == "recall":
        m = SharedMemory("cli", "none", "cli")
        print(m.recall_text(sys.argv[2]) or "(nothing recalled)")
    else:
        print(__doc__ if __doc__ else "usage: shared_memory.py check | recall <query>")
