"""Experiment log rows (spec 4.3) and the per-run writer.

Each run writes outputs/runs/<run_id>/experiments.jsonl. One RunLog owns
that file. scripts/sync_log.py merges run files into the shared
experiments.jsonl; nothing appends to a shared log from two processes.

Every row has "type": "eval", "probe" or "event", carries "backend", and
carries "stub": true when the backend is the stub, so charts can watermark
STUB rows.

Eval and probe rows have every key of spec 4.3, plus "type", "stub",
"fidelity" and "graft_report". Fit is logged next to every score, using the
compile report's names:
- "fit_r2": {"<part id>@<host layer>": R squared or None}
- "fit": {"<part id>@<host layer>": "ridge" or "skipped"}
- "graft_report": the per-graft compile report entries (host_layer, fit,
  fit_r2, fit_pairs, part_params, glue_params, ...), in graft order.
A random-part control graft's key ends in "~random".

Event rows have type, kind, text, the run fields and the current counters.
"""

import datetime
import json
import os
import threading

STATUSES = ["ok", "collapsed", "invalid", "rejected_fit", "compile_error", "cache_hit",
            "infra_error", "budget_exhausted", "control"]


def now_iso():
    """The current UTC time, for example 2026-11-14T23:40:00.123Z."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (now.microsecond // 1000)


def new_eval_row():
    """A fresh eval row with every key at its default."""
    return {
        "timestamp": now_iso(),
        "type": "eval",
        "run_id": None,
        "method": None,
        "backend": None,
        "stub": False,
        "agent": None,
        "round": None,
        "hypothesis": "",
        "citations": [],
        "program": {},
        "program_hash": None,
        "describe": "",
        "split": "search",
        "fidelity": "full",
        "fit_r2": {},
        "fit": {},
        "graft_report": [],
        "heal_loss": [],
        "scores": {},
        "combined": None,
        "status": None,
        "cache_hit": False,
        "token": None,
        "eval_count": 0.0,
        "eval_budget": 0.0,
        "compute_s": {},
        "compute_used_s": 0.0,
        "compute_budget_s": 0.0,
        "extra": {},
    }


def new_probe_row():
    """A fresh probe row (a fit-only compile: the program with heal set to None)."""
    row = new_eval_row()
    row["type"] = "probe"
    row["split"] = None
    row["fidelity"] = None
    return row


def new_event_row(kind, text):
    """A fresh event row (analyst map, judge verdict, Exa call, memory op, ...)."""
    return {
        "timestamp": now_iso(),
        "type": "event",
        "kind": kind,
        "text": text,
        "run_id": None,
        "method": None,
        "backend": None,
        "stub": False,
        "agent": None,
        "round": None,
        "eval_count": 0.0,
        "eval_budget": 0.0,
        "compute_used_s": 0.0,
        "compute_budget_s": 0.0,
        "extra": {},
    }


class RunLog:
    """Writes rows to <run_dir>/experiments.jsonl and keeps them in memory."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "experiments.jsonl")
        self.lock = threading.Lock()
        self.rows = []

    def write(self, row):
        line = json.dumps(row, sort_keys=False) + "\n"
        with self.lock:
            self.rows.append(row)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)

    def all_rows(self):
        with self.lock:
            return list(self.rows)

    def write_json(self, name, data):
        """Write a JSON file (manifest.json, summary.json) into the run folder."""
        path = os.path.join(self.run_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        return path


def read_rows(path):
    """Read a JSONL log into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                rows.append(json.loads(line))
    return rows
