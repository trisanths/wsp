# Test of the catalog lock. Three writers replace catalog/<model>.jsonl with rows they read earlier:
# the agent's confirm_math tool (agent.py), mine/regate.py, and mine/enrich_rows.py. Each must hold the
# lock of client.write_part and read the file again under it, or a row that another lane or a sweep
# appended in the meantime is lost.
#   .venv/bin/python mine/tests/test_catalog_lock.py
# The real agent.Session.tool("confirm_math"), client.Miner.write_part, and regate.regate run on a
# temporary person_a folder. The engine is a stand-in; nothing here talks to Modal.
import fcntl
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import agent  # noqa: E402
import client  # noqa: E402
import common  # noqa: E402
import regate  # noqa: E402

SHORT = "toy"


def engine_call(method, *args):
    if method == "ablate":
        payload, part = args
        drops = {}
        for name in payload["skills"]:
            for split in ["search", "heldout"]:
                drops[name + "|" + split] = 0.3 if name == "math" else 0.002
        return {"drops": drops, "ablated": {}, "seconds": 1.0}
    if method == "random_drops":
        return {"drops": [0.01, 0.02, 0.03, 0.02, 0.02], "seconds": 1.0}
    raise RuntimeError("Class has no method " + method)


def make_miner():
    m = object.__new__(client.Miner)
    m.short, m.hf_id, m.revision, m.params_b, m.modal_seconds = SHORT, "org/Toy", "rev", 1.0, 0.0
    m.state = {"revision": "rev", "structure": {"n_layers": 8, "d_model": 64, "layers_detail": []}, "skipped": []}
    m.call = engine_call
    return m


def write_row(miner, layer, label):
    part = {"kind": "mlp_cluster", "layer": layer, "indices": [1, 2, 3]}
    res = miner.screen(part, "math")
    assert res["gate"]["passed"], res["gate"]
    return miner.write_part(part, label, "math", res, ["weights"], "why", "screened")


def rows():
    with open(common.catalog_path(SHORT)) as f:
        return [json.loads(line) for line in f if line.strip()]


def lock_is_held():
    # flock locks belong to the open file, so a second open of the lock file in this process is
    # refused while the first one holds the lock
    with open(common.catalog_path(SHORT) + ".lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def test_confirm_math():
    miner = make_miner()
    other = make_miner()      # another lane, or a sweep, on the same model
    first = write_row(miner, 2, "agent math part")
    session = object.__new__(agent.Session)
    session.miner, session.short = miner, SHORT
    seen = {}

    def confirm_while_another_lane_writes(part):
        # the engine call of confirm_math takes minutes; the lock must be free for the other writers
        assert not lock_is_held()
        seen["second"] = write_row(other, 5, "second lane part")
        return {"drop": 0.5, "passed": True, "seconds": 1.0}
    miner.confirm = confirm_while_another_lane_writes
    real_replace = os.replace

    def replace(src, dst):
        if dst == common.catalog_path(SHORT):
            seen["locked"] = lock_is_held()
        real_replace(src, dst)
    os.replace = replace
    try:
        out = session.tool("confirm_math", {"catalog_id": first["id"]})
    finally:
        os.replace = real_replace
    assert out == {"drop": 0.5, "passed": True, "seconds": 1.0}, out
    assert seen["locked"] is True and not lock_is_held()
    got = rows()
    # the row that the other lane appended during the engine call is still there, unchanged
    assert [r["id"] for r in got] == [first["id"], seen["second"]["id"]], [r["id"] for r in got]
    assert got[1] == seen["second"] and got[1]["tier"] == "screened"
    assert got[0]["tier"] == "confirmed" and got[0]["signature"]["confirm"] == out
    assert dict(got[0], tier="screened", signature=dict(got[0]["signature"], confirm=None)) == first

    # the row left the catalog during the engine call (regate.py, an audit): nothing is written
    third = write_row(miner, 6, "third part")
    state = {}

    def confirm_while_the_row_leaves(part):
        kept = [r for r in rows() if r["id"] != third["id"]]
        with open(common.catalog_path(SHORT), "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        state["before"] = open(common.catalog_path(SHORT)).read()
        return {"drop": 0.5, "passed": True, "seconds": 1.0}
    miner.confirm = confirm_while_the_row_leaves
    out = session.tool("confirm_math", {"catalog_id": third["id"]})
    assert out["passed"] and "left the catalog" in out["note"], out
    assert open(common.catalog_path(SHORT)).read() == state["before"]

    # a failed check changes nothing, and an unknown id is refused
    miner.confirm = lambda part: {"drop": 0.1, "passed": False, "seconds": 1.0}
    failed = session.tool("confirm_math", {"catalog_id": seen["second"]["id"]})
    assert failed == {"drop": 0.1, "passed": False, "seconds": 1.0}
    assert open(common.catalog_path(SHORT)).read() == state["before"]
    assert session.tool("confirm_math", {"catalog_id": "nothing"}) == {"error": "unknown catalog_id"}


def test_regate():
    miner = make_miner()
    targets = ["math", "code", "facts", "multilingual"]
    before = rows()
    assert len(before) == 2
    seen = {"gate": [], "replace": []}
    real_gate = regate.light_gate
    real_replace = os.replace

    def gate(drops, random_drops, target, all_targets):
        seen["gate"].append(lock_is_held())
        return real_gate(drops, random_drops, target, all_targets)

    def replace(src, dst):
        if dst == common.catalog_path(SHORT):
            seen["replace"].append(lock_is_held())
        real_replace(src, dst)
    regate.light_gate = gate
    os.replace = replace
    try:
        keep, rejected = regate.regate(SHORT, targets)
    finally:
        regate.light_gate = real_gate
        os.replace = real_replace
    # the lock is held from the read to the replace, and it is free afterwards
    assert seen == {"gate": [True, True], "replace": [True]}, seen
    assert not lock_is_held()
    assert [r["id"] for r in keep] == [r["id"] for r in before] and rejected == [] and rows() == keep
    # a writer that comes after regate still appends
    last = write_row(miner, 7, "after regate")
    assert [r["id"] for r in rows()] == [r["id"] for r in before] + [last["id"]]

    # a row that no longer passes moves to the rejected file (here: the third row)
    calls = []

    def strict_gate(drops, random_drops, target, all_targets):
        calls.append(target)
        return {"passed": len(calls) != 3, "reasons": [] if len(calls) != 3 else ["test"]}
    regate.light_gate = strict_gate
    try:
        keep, rejected = regate.regate(SHORT, targets)
    finally:
        regate.light_gate = real_gate
    assert [r["id"] for r in rejected] == [last["id"]] and rejected[0]["rejected_reasons"] == ["test"]
    with open(os.path.join(common.ROOT, "catalog", SHORT + ".rejected.jsonl")) as f:
        assert [json.loads(line)["id"] for line in f] == [last["id"]]
    assert [r["id"] for r in rows()] == [r["id"] for r in before]


def main():
    real_root = common.ROOT
    old_lane = os.environ.pop("WSP_LANE", None)
    tmp = tempfile.mkdtemp(prefix="catalog-lock-")
    common.ROOT = tmp
    try:
        test_confirm_math()
        test_regate()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        if old_lane is not None:
            os.environ["WSP_LANE"] = old_lane


if __name__ == "__main__":
    main()
    print("ok")
