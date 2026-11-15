# Backfill the evidence fields of existing catalog rows from what each row stores. No GPU call.
#   .venv/bin/python mine/enrich_rows.py --model <short_name> [--model <short_name> ...]
# For each row of catalog/<short_name>.jsonl that the mining pipeline wrote, it recomputes
# signature.z, z_heldout, random_n, random_mean, random_std, random_heldout and guard, and the
# top-level selection_data, and it sets enriched_at. Every existing field stays, and so does the row
# order. The file is replaced in one step (temp file, then os.replace). A second run gives the same
# file apart from enriched_at. The *.rejected.jsonl files are never read or written.
#
#   --routing   also fills "routing" for expert rows that have none, through the GPU engine. This is
#               the only path that calls the engine, and it costs GPU time.
#   --force     run although the model looks live (see below).
#   --live-since <commit>=<YYYY-MM-DDTHH:MM:SS>   see "Built-in sets of old rows" below.
#
# Live runs. A mining run appends rows to the catalog, and it writes to logs/<short_name>/ all the
# time: the state file after each screen, experiments.jsonl after each engine call, and the run
# transcript after each agent message. A model counts as live when any file in logs/<short_name>/
# changed in the last 10 minutes, and the script refuses it unless --force is given. A fresh git
# checkout also gives the log files new times, so the check can refuse a model that is not running.
# Wait 10 minutes, or pass --force when you know that no run is active. The script holds the catalog
# lock of client.write_part while it replaces the file, so a row that a run appends is not lost.
#
# Built-in sets of old rows. A row does not store which version of the built-in skill file the
# engine scored it with. A row that client.write_part wrote with builtin_basis "engine" keeps the
# descriptions the engine reported. For every other row the script takes the version of
# mine/data/builtin_skills.json (and of build_data.py) from git history: the last commit of that file
# at or before the row's "created" time, or the first commit when the row is older than all of them.
# This is an estimate. The engine uses the file from its last deploy, and the deploy time is not in
# git. When you know when a version went live, pass --live-since <commit>=<time>, and that time is
# used in place of the commit time. "created" has no time zone; it is read as local time of this machine.
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import evidence  # noqa: E402

LIVE_SECONDS = 600
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
_at = {}


def parse_time(text):
    try:
        return time.mktime(time.strptime(str(text), TIME_FORMAT))
    except ValueError:
        return None


# ---------- live check ----------

def live_files(short, now=None):
    # log files of this model that changed in the last LIVE_SECONDS
    now = now or time.time()
    folder = common.logs_dir(short)
    recent = []
    if not os.path.isdir(folder):
        return recent
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) < LIVE_SECONDS:
            recent.append(name)
    return recent


# ---------- built-in skill file versions ----------

def git(args):
    try:
        out = subprocess.run(["git", "-C", common.MINE_DIR] + args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def builtin_versions(live_since=None):
    # [{"rev", "epoch", "time_from"}] oldest first; [] when there is no git history to read
    rel = os.path.relpath(common.BUILTIN_SKILLS, common.MINE_DIR)
    out = git(["log", "--format=%H %ct", "--", rel])
    versions = []
    for line in reversed((out or "").splitlines()):
        pieces = line.split()
        if len(pieces) == 2:
            versions.append({"rev": pieces[0], "epoch": float(pieces[1]), "time_from": "commit"})
    for prefix in live_since or {}:
        hit = [v for v in versions if v["rev"].startswith(prefix)]
        if len(hit) != 1:
            raise SystemExit("--live-since: '" + prefix + "' names " + str(len(hit)) + " versions of the built-in file")
        hit[0]["epoch"] = live_since[prefix]
        hit[0]["time_from"] = "--live-since"
    return versions


def pick_version(versions, created_epoch):
    # the last version at or before the row's creation; the first one for a row older than all of
    # them (the file was in use before its first commit); the newest when the row has no time
    if not versions:
        return None
    if created_epoch is None:
        return versions[-1]
    chosen = versions[0]
    for v in versions:
        if v["epoch"] <= created_epoch:
            chosen = v
    return chosen


def builtin_at(rev):
    # the built-in sources as of one commit: descriptions from the skill file, dataset ids from build_data.py
    if rev not in _at:
        rel = os.path.relpath(common.BUILTIN_SKILLS, common.MINE_DIR)
        text = git(["show", rev + ":./" + rel])
        source = git(["show", rev + ":./build_data.py"])
        if text is None:
            _at[rev] = None
        else:
            data = json.loads(text)
            described = {}
            for name in data:
                described[name] = {"description": data[name].get("description"), "dataset": data[name].get("dataset")}
            builders = evidence.builder_datasets(source) if source else {}
            _at[rev] = evidence.builtin_sources(described, builders)
    return _at[rev]


# ---------- one row ----------

def agent_skills_on_disk(short):
    # the frozen tests of this model, from every lane: {name: {"path", "hash", "description"}}
    out = {}
    folder = common.skills_dir(short)
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        out[data.get("name", name[:-5])] = {"path": os.path.relpath(path, common.ROOT), "hash": data.get("hash"),
                                             "description": data.get("description")}
    return out


def from_mining(rec):
    sig = rec.get("signature")
    return (isinstance(sig, dict) and "effect" in sig and "random_drops" in sig
            and isinstance(rec.get("skill"), dict) and "name" in rec["skill"])


def stored_builtin(old):
    # the built-in descriptions that the engine reported when the row was written
    out = {}
    if old.get("source") == "builtin":
        out[old["skill"]] = {"source": "builtin", "description": old.get("description"),
                             "datasets": old.get("datasets") or []}
    for name in old.get("off_target_data") or {}:
        if old["off_target_data"][name].get("source") == "builtin":
            out[name] = dict(old["off_target_data"][name])
    return out


def enrich_row(rec, agent, versions, healdata, stamp):
    # Changes rec in place. Returns False for a row that the mining pipeline did not write.
    if not from_mining(rec):
        return False
    sig = rec["signature"]
    target = rec["skill"]["name"]
    sig.update(evidence.signature_fields(sig["effect"], sig["random_drops"], target))

    agent = dict(agent)
    if rec["skill"].get("source") == "agent":
        agent[target] = {"path": rec["skill"].get("path"), "hash": rec["skill"].get("hash"),
                         "description": rec["skill"].get("description")}
    old = rec.get("selection_data") if isinstance(rec.get("selection_data"), dict) else {}
    version = None
    if old.get("builtin_basis") == "engine":
        builtin = stored_builtin(old)
        basis = "engine"
    else:
        version = pick_version(versions, parse_time(rec.get("created")))
        builtin = builtin_at(version["rev"]) if version else None
        basis = "git history"
        if builtin is None:
            version = None
            builtin = evidence.local_builtin()
            basis = "local file"
    new = evidence.selection_data(target, sig["effect"], evidence.make_lookup(agent, builtin), healdata, basis)
    if version:
        new["builtin_version"] = {"commit": version["rev"][:12], "time_from": version["time_from"],
                                  "since": time.strftime(TIME_FORMAT, time.localtime(version["epoch"]))}
    merged = dict(old)
    merged.update(new)
    for key in ["note", "builtin_version"]:
        if key in merged and key not in new:
            del merged[key]  # a note of an earlier run that no longer holds
    rec["selection_data"] = merged
    rec["enriched_at"] = stamp
    return True


# ---------- routing (the only engine path) ----------

def measure_routing(short, rows):
    # {row id: (routing, note)} for expert rows without routing. Calls the GPU engine.
    wanted = [r for r in rows if from_mining(r) and r.get("type") == "expert" and r.get("routing") is None]
    out = {}
    if not wanted:
        return out
    import torch
    from client import Miner

    try:
        miner = Miner(short)
    except (SystemExit, Exception) as e:
        for r in wanted:
            out[r["id"]] = (None, "no routing: could not reach the engine: " + type(e).__name__ + ": " + str(e)[:160])
        return out
    for r in wanted:
        try:
            idx = torch.load(os.path.join(common.ROOT, r["units"]))["indices"].tolist()
            part = {"kind": "expert", "layer": int(r["layer"]), "indices": idx}
            extra = {}
            if r["skill"].get("source") == "agent":
                with open(os.path.join(common.ROOT, r["skill"]["path"])) as f:
                    data = json.load(f)
                extra[r["skill"]["name"]] = {"hash": data["hash"], "items": data["items"], "control": data["control"]}
        except Exception as e:
            out[r["id"]] = (None, "no routing: could not read the part or its skill test: " + type(e).__name__
                            + ": " + str(e)[:160])
            continue
        out[r["id"]] = miner.expert_routing(part, r["skill"]["name"], extra)
        print("  routing", r["id"], "ok" if out[r["id"]][0] is not None else out[r["id"]][1])
    print("  engine seconds:", round(miner.modal_seconds, 1))
    return out


# ---------- one model ----------

def read_rows(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                raise SystemExit(path + " line " + str(n) + " is not JSON; nothing was written")
    return rows


def write_rows(path, rows):
    tmp = path + ".enrich.tmp"
    with open(tmp, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def enrich_model(short, routing=False, force=False, versions=None, healdata=None, stamp=None):
    # Returns a short status text; it starts with "refused" or "skipped" when nothing was written.
    if short.endswith(".rejected"):
        return "refused: rejected files are never touched"
    path = common.catalog_path(short)
    if not os.path.exists(path):
        return "skipped: no catalog file"
    recent = live_files(short)
    if recent and not force:
        return ("refused: looks live, changed in the last " + str(LIVE_SECONDS // 60) + " minutes: "
                + ", ".join(recent[:4]) + " (pass --force when no run is active)")
    stamp = stamp or time.strftime(TIME_FORMAT)
    measured = {}
    if routing:
        measured = measure_routing(short, read_rows(path))
    with open(path + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            rows = read_rows(path)  # read again under the lock: a run may have appended a row
            agent = agent_skills_on_disk(short)
            done = 0
            for rec in rows:
                if not enrich_row(rec, agent, versions or [], healdata, stamp):
                    continue
                done += 1
                if rec.get("id") in measured:
                    rec["routing"] = measured[rec["id"]][0]
                    rec.pop("routing_note", None)
                    if measured[rec["id"]][1]:
                        rec["routing_note"] = measured[rec["id"]][1]
            write_rows(path, rows)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return "enriched " + str(done) + " of " + str(len(rows)) + " rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="short name; repeat for more models")
    ap.add_argument("--routing", action="store_true", help="fill routing for expert rows (calls the GPU engine)")
    ap.add_argument("--force", action="store_true", help="run although logs/<model>/ changed in the last 10 minutes")
    ap.add_argument("--live-since", action="append", default=[], metavar="COMMIT=TIME",
                    help="when the engine began to use the built-in file of that commit, e.g. 1ceeff6=2026-11-14T20:50:00")
    args = ap.parse_args()
    live_since = {}
    for item in args.live_since:
        prefix, _, text = item.partition("=")
        when = parse_time(text)
        if not prefix or when is None:
            raise SystemExit("--live-since needs <commit>=<YYYY-MM-DDTHH:MM:SS>, got '" + item + "'")
        live_since[prefix] = when
    versions = builtin_versions(live_since)
    if not versions:
        print("no git history for the built-in skill file: built-in descriptions come from the local file")
    healdata = evidence.load_healdata()
    if healdata is None:
        print("wsp/healdata.py not found: pool and reported_benchmarks stay unchecked (null)")
    failed = False
    for short in args.model:
        status = enrich_model(short, routing=args.routing, force=args.force, versions=versions, healdata=healdata)
        print("[" + short + "]", status)
        if status.startswith("refused"):
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
