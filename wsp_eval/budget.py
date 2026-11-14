"""Shared evaluation budget counter.

One JSON file (work/budget.json), protected by a file lock so the swarm, the
baselines and the interp agent can all run at once. Each method uses its own
namespace (e.g. "swarm", "evo_merge", "lora", "full_ft", "interp"); the
comparison is valid only when the namespaces being compared used the same
number of search-split eval() calls.

Unit of budget: one eval() call on the search split, cached or not.
Held-out calls are counted separately and never limited.
"""
import fcntl
import json
import os
from contextlib import contextmanager

from wsp_eval import config


class BudgetExhausted(Exception):
    pass


def _budget_file():
    return config.work_path("budget.json")


@contextmanager
def _locked_state():
    lock_path = _budget_file() + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = {"namespaces": {}}
        if os.path.exists(_budget_file()):
            with open(_budget_file()) as f:
                state = json.load(f)
        yield state
        tmp = _budget_file() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, _budget_file())
        fcntl.flock(lock, fcntl.LOCK_UN)


def _namespace_entry(state, namespace):
    if namespace not in state["namespaces"]:
        state["namespaces"][namespace] = {
            "eval_calls": 0, "heldout_calls": 0, "items_scored": 0,
            "generations_run": 0, "limit": None,
        }
    return state["namespaces"][namespace]


def set_limit(namespace, limit):
    """Set the maximum number of search-split eval() calls for a namespace (None = unlimited)."""
    with _locked_state() as state:
        entry = _namespace_entry(state, namespace)
        entry["limit"] = limit


def check(namespace, split):
    """Raise BudgetExhausted if one more search call would go over the limit."""
    if split != "search":
        return
    with _locked_state() as state:
        entry = _namespace_entry(state, namespace)
        if entry["limit"] is not None and entry["eval_calls"] + 1 > entry["limit"]:
            raise BudgetExhausted("namespace " + repr(namespace) + " used " + str(entry["eval_calls"]) +
                                  " of " + str(entry["limit"]) + " eval calls")


def charge(namespace, split, items_scored, generations_run):
    """Record one eval() call. Returns the updated namespace entry."""
    with _locked_state() as state:
        entry = _namespace_entry(state, namespace)
        if split == "search":
            entry["eval_calls"] += 1
        else:
            entry["heldout_calls"] += 1
        entry["items_scored"] += items_scored
        entry["generations_run"] += generations_run
        result = dict(entry)
    return result


def usage(namespace=None):
    with _locked_state() as state:
        if namespace is None:
            return json.loads(json.dumps(state["namespaces"]))
        return dict(_namespace_entry(state, namespace))


def reset(namespace):
    with _locked_state() as state:
        if namespace in state["namespaces"]:
            limit = state["namespaces"][namespace]["limit"]
            del state["namespaces"][namespace]
            _namespace_entry(state, namespace)["limit"] = limit
