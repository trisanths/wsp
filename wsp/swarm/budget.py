"""The two-axis budget (spec section 5) and the one shared ledger.

Axis 1 is eval tokens, counted in full-eval equivalents (FEE). A full
search-split eval costs 1.0; lower fidelities cost a fraction (config
"fidelities"). Axis 2 is compute seconds: compile, fit, heal and eval time.

Every arm of a comparison gets its own Budget with the same caps, and a
second Budget instance (pool "heldout") holds its held-out allowance. Every
charge from every arm goes into one ledger file, outputs/ledger.jsonl.

The budget fails closed: a spend that would go over the eval cap is
refused. Compute is checked before work starts (compute_remaining() > 0)
and charged after it, so a job may overshoot by at most the jobs in flight.
"""

import json
import os
import threading
import time

SCALE = 1000  # eval amounts are kept as integer thousandths, so threads never see float drift

_PATH_LOCKS = {}
_PATH_LOCKS_GUARD = threading.Lock()


def lock_for_path(path):
    """One lock per ledger file, shared by every Ledger object in this process."""
    key = os.path.abspath(path)
    with _PATH_LOCKS_GUARD:
        if key not in _PATH_LOCKS:
            _PATH_LOCKS[key] = threading.Lock()
        return _PATH_LOCKS[key]


def to_units(amount):
    return int(round(float(amount) * SCALE))


class Ledger:
    """The append-only JSONL file that records every charge from every arm.

    Appends hold a lock per file. Each row is one write of one line, which
    keeps lines whole when several runs in one process share the file.
    """

    def __init__(self, path):
        self.path = path
        folder = os.path.dirname(path)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        self.lock = lock_for_path(path)

    def append(self, row):
        line = json.dumps(row, sort_keys=True) + "\n"
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)

    def read(self):
        rows = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line != "":
                    rows.append(json.loads(line))
        return rows


class Budget:
    """The two-axis budget interface. LocalBudget below; ModalBudget implements the same.

    Subclasses must implement the six methods marked "required". Every
    implementation must be thread-safe, because step_many calls it from a
    thread pool, and must fail closed: when the budget store cannot be
    reached, spend_eval returns None and exhausted() returns True.

    Ledger rows have these keys: run_id, method, pool, axis ("eval" or
    "compute"), amount (FEE or seconds), token (eval rows only), who, phase,
    time (unix seconds).
    """

    def __init__(self, eval_budget, compute_budget_s, run_id="", method="", pool="search"):
        self.eval_budget = float(eval_budget)
        self.compute_budget_s = float(compute_budget_s)
        self.run_id = run_id
        self.method = method
        self.pool = pool

    def spend_eval(self, who, amount=1.0):
        """Required. Spend `amount` FEE for one eval and return its token.

        The token is an int, unique within this budget, given out in spend
        order starting at 0, so it doubles as the eval index. Returns None
        (and spends nothing) when the spend would go over the cap. Token 0
        is falsy, so test the result with `is None`. Never refunds.
        """
        raise NotImplementedError

    def charge_compute(self, seconds, who, phase):
        """Required. Record compute seconds already used. Never refused.

        phase is one of "compile", "probe", "heal", "eval".
        """
        raise NotImplementedError

    def compute_remaining(self):
        """Required. Seconds left on the compute axis. Negative after an overshoot."""
        raise NotImplementedError

    def eval_remaining(self):
        """Required. FEE left on the eval axis, as a float."""
        raise NotImplementedError

    def exhausted(self):
        """Required. True when either axis is spent."""
        raise NotImplementedError

    def ledger_rows(self):
        """Required. The ledger rows this budget has written, oldest first."""
        raise NotImplementedError

    # Helpers built on the required methods.

    def eval_used(self):
        return round(self.eval_budget - self.eval_remaining(), 6)

    def compute_used(self):
        return self.compute_budget_s - self.compute_remaining()

    def can_spend(self, amount):
        """True when `amount` FEE is still available (a hint; spend_eval decides)."""
        return to_units(self.eval_remaining()) >= to_units(amount)

    def counters(self):
        """The counters every log row carries."""
        return {
            "eval_count": self.eval_used(),
            "eval_budget": self.eval_budget,
            "compute_used_s": self.compute_used(),
            "compute_budget_s": self.compute_budget_s,
        }

    def overshoot_s(self):
        """Compute seconds charged beyond the cap (0 when within it)."""
        over = -self.compute_remaining()
        if over < 0:
            return 0.0
        return over


class LocalBudget(Budget):
    """A budget held in memory behind one lock, for runs on this machine."""

    def __init__(self, eval_budget, compute_budget_s, run_id="", method="", pool="search", ledger=None):
        Budget.__init__(self, eval_budget, compute_budget_s, run_id, method, pool)
        self.eval_total_units = to_units(eval_budget)
        self.eval_spent_units = 0
        self.compute_spent = 0.0
        self.next_token = 0
        self.ledger = ledger
        self.rows = []
        self.lock = threading.Lock()

    @classmethod
    def from_config(cls, config, pool, ledger):
        """Build the search or held-out budget for a run from its config."""
        if pool == "heldout":
            return cls(config["heldout_eval_budget"], config["heldout_compute_budget_s"],
                       config["run_id"], config["method"], pool, ledger)
        return cls(config["eval_budget"], config["compute_budget_s"],
                   config["run_id"], config["method"], pool, ledger)

    def record(self, axis, amount, token, who, phase):
        """Write one ledger row. Called with self.lock held."""
        row = {"run_id": self.run_id, "method": self.method, "pool": self.pool, "axis": axis,
               "amount": amount, "token": token, "who": who, "phase": phase, "time": time.time()}
        self.rows.append(row)
        if self.ledger is not None:
            self.ledger.append(row)

    def spend_eval(self, who, amount=1.0):
        units = to_units(amount)
        if units < 0:
            raise ValueError("an eval cannot cost a negative amount")
        with self.lock:
            if self.eval_spent_units + units > self.eval_total_units:
                return None
            self.eval_spent_units += units
            token = self.next_token
            self.next_token += 1
            self.record("eval", units / SCALE, token, who, "eval")
            return token

    def charge_compute(self, seconds, who, phase):
        seconds = float(seconds)
        if seconds <= 0:
            return
        with self.lock:
            self.compute_spent += seconds
            self.record("compute", seconds, None, who, phase)

    def compute_remaining(self):
        with self.lock:
            return self.compute_budget_s - self.compute_spent

    def eval_remaining(self):
        with self.lock:
            return (self.eval_total_units - self.eval_spent_units) / SCALE

    def exhausted(self):
        with self.lock:
            if self.eval_spent_units >= self.eval_total_units:
                return True
            if self.compute_spent >= self.compute_budget_s:
                return True
            return False

    def ledger_rows(self):
        with self.lock:
            return list(self.rows)
