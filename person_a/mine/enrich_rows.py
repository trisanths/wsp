# Backfill the evidence fields of existing catalog rows from what each row stores. No GPU call.
#   .venv/bin/python mine/enrich_rows.py --model <short_name> [--model <short_name> ...]
# For each row of catalog/<short_name>.jsonl that the mining pipeline wrote, it recomputes
# signature.z, z_heldout, z_flag, random_n, random_mean, random_std, random_distinct, random_heldout
# and guard, and the top-level selection_data, and it sets enriched_at. Every existing field stays,
# and so does the row order; the only fields it takes away are a routing_note or selection_note whose
# field it has now filled. The file is replaced in one step (temp file, then os.replace). A second
# run gives the same file apart from enriched_at. The *.rejected.jsonl files are never read or written.
#
#   --routing         also fills "routing" for expert rows that have none, through the GPU engine.
#                     This is the only path that calls the engine, and it costs GPU time.
#   --retry-routing   with --routing: also ask again for rows whose routing_note names a final reason.
#   --force           run although the model looks live, or although its logs folder is not here.
#   --live-since <commit>=<YYYY-MM-DDTHH:MM:SS>   see "Built-in sets of old rows" below.
#
# Where and when to run it. Run it on the machine that mines the model, after the run has stopped
# and the catalog is pushed. The script rewrites every line of the catalog file, so a teammate who
# appends rows to the same file on another machine gets a merge conflict on the next push.
#
# Live runs. A mining run appends rows to the catalog, and it writes to logs/<short_name>/ all the
# time: the state file after each screen, experiments.jsonl after each engine call, and the run
# transcript after each agent message. A model counts as live when any file in logs/<short_name>/
# changed in the last 10 minutes, and the script refuses it unless --force is given. It also refuses
# a model that has no logs/<short_name>/ folder on this machine: the logs of a model that a teammate
# mines are on the teammate's machine, so there is no evidence either way. A fresh git checkout also
# gives the log files new times, so the check can refuse a model that is not running. Wait 10
# minutes, or pass --force when you know that no run is active.
#
# The lock. The script holds the catalog lock of client.write_part while it replaces the file, so a
# row that write_part appends in that time is not lost. Two other writers do not take that lock: the
# agent's confirm_math tool (agent.py) and regate.py read the catalog and later replace it with the
# rows they read. If the backfill runs between their read and their write, they write the old rows
# back and the new fields are gone, without a message. --force does not protect against that.
#
# Built-in sets of old rows. A row does not store which version of the built-in skill file the
# engine scored it with. A row that client.write_part wrote with builtin_basis "engine" keeps what
# the engine reported. For every other row the script reads the versions of
# mine/data/builtin_skills.json (and of build_data.py) from git history. The engine uses the file
# from its last deploy, and the deploy time is not in git. So a version counts as possibly live from
# its commit time on (the script assumes that nobody deployed a file before its commit), and the
# first version counts as live before that. A row made after the commit of a newer version can have
# been scored with that version or with the one before it. Such a row gets builtin_basis
# "git history, version uncertain": no description for the skills on which the versions differ,
# reported_benchmarks as the union over the versions, and what each version says in
# builtin_candidates. To settle it, ask the person who deployed and pass
# --live-since <commit>=<time the engine began to use that version>. A row without a "created" time
# is treated as possibly made under every version. "created" has no time zone; it is read as local
# time of this machine.
#
# Agent skills. Lanes on one model share mine/skills/<short_name>/, and a later lane can replace
# the frozen test file of a skill with another test of the same name. selection_data.path_matches
# says whether the file at selection_data.path still has the hash the row was selected on. For an
# off-target agent skill the hash is written only when one test of that name is known on this
# machine (the file, the catalog rows, the state files); else it is null with hash_candidates.
#
# Routing (--routing). The engine is built for the revision that the row stores, not for the one in
# this machine's state file. A row gets routing null and a routing_note that starts with
# "no routing, final: ", with no engine call, when the row has no revision, when its part file or
# frozen test cannot be read, when the frozen test file does not have the row's hash, or when the
# state file of that revision shows module_list experts in the layer. The routing step writes
# nothing to logs/<short_name>/.
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
UNCERTAIN = "git history, version uncertain"
# keys of selection_data that an earlier run can have written and that may no longer hold
PASSING_KEYS = ["note", "builtin_version", "builtin_candidates", "builtin_note", "path_hash", "path_note"]
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


def state_files(short):
    # the state files of this model on this machine, one per lane
    folder = common.logs_dir(short)
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("state") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(folder, name)) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


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


def candidate_versions(versions, created_epoch):
    # The versions the engine can have held when a row was made, oldest first. The first version is
    # one of them until a later one is known to be live (the file was in use before its first
    # commit). A later version with a --live-since time at or before the row replaces all older
    # ones. A later version with only a commit time at or before the row is added: it was live or
    # it was not yet deployed. A version whose time is after the row is left out. A row with no
    # time can come from any version.
    if not versions:
        return []
    if created_epoch is None:
        return list(versions)
    out = [versions[0]]
    for v in versions[1:]:
        if v["epoch"] > created_epoch:
            continue
        if v["time_from"] == "--live-since":
            out = [v]
        else:
            out.append(v)
    return out


def version_label(version):
    return {"commit": version["rev"][:12], "time_from": version["time_from"],
            "since": time.strftime(TIME_FORMAT, time.localtime(version["epoch"]))}


def builtin_at(rev):
    # the built-in sources as of one commit: descriptions from the skill file, dataset ids from build_data.py
    if rev not in _at:
        rel = os.path.relpath(common.BUILTIN_SKILLS, common.MINE_DIR)
        text = git(["show", rev + ":./" + rel])
        source = git(["show", rev + ":./build_data.py"])
        if text is None:
            _at[rev] = None
        else:
            builders = evidence.builder_datasets(source) if source else {}
            _at[rev] = evidence.builtin_sources(evidence.describe_builtin(json.loads(text)), builders)
    return _at[rev]


# ---------- one row ----------

def agent_skills_on_disk(short):
    # the frozen test files of this model: {name: {"path", "hash", "description"}}. One file per
    # name: when two lanes froze a test of the same name, the file holds the later one.
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


def agent_tests(short, rows, states):
    # {name: {"path", "hash", "description"}} for evidence.make_lookup. The hash of a name is given
    # only when every record of this machine agrees on it: the file on disk, the catalog rows whose
    # target it is, and the state files of the lanes. With more than one hash, which test a row's
    # lane scored as an off-target skill is not known: hash None and the candidates.
    out = agent_skills_on_disk(short)
    hashes = {}
    for name in out:
        if out[name].get("hash"):
            hashes.setdefault(name, set()).add(out[name]["hash"])
    for rec in rows:
        skill = rec.get("skill")
        if isinstance(skill, dict) and skill.get("source") == "agent" and skill.get("name") and skill.get("hash"):
            hashes.setdefault(skill["name"], set()).add(skill["hash"])
    for state in states:
        skills = state.get("agent_skills")
        for name in skills if isinstance(skills, dict) else {}:
            if isinstance(skills[name], dict) and skills[name].get("hash"):
                hashes.setdefault(name, set()).add(skills[name]["hash"])
    for name in hashes:
        spec = dict(out.get(name) or {"path": None, "description": None})
        found = sorted(hashes[name])
        spec["hash"] = found[0] if len(found) == 1 else None
        if len(found) > 1:
            spec["hash_candidates"] = found
        out[name] = spec
    return out


def from_mining(rec):
    sig = rec.get("signature")
    return (isinstance(sig, dict) and "effect" in sig and "random_drops" in sig
            and isinstance(rec.get("skill"), dict) and "name" in rec["skill"])


def stored_builtin(old):
    # the built-in sets that the engine reported when the row was written
    out = {}
    if old.get("source") == "builtin":
        out[old["skill"]] = {"source": "builtin", "description": old.get("description"),
                             "datasets": old.get("datasets") or []}
        if old.get("items_sha"):
            out[old["skill"]]["items_sha"] = old["items_sha"]
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

    def build(builtin, basis):
        return evidence.selection_data(target, sig["effect"], evidence.make_lookup(agent, builtin), healdata, basis)

    if old.get("builtin_basis") == "engine":
        new = build(stored_builtin(old), "engine")
    else:
        candidates = candidate_versions(versions, parse_time(rec.get("created")))
        if not candidates or any(builtin_at(v["rev"]) is None for v in candidates):
            new = build(evidence.local_builtin(), "local file")
        elif len(candidates) == 1:
            new = build(builtin_at(candidates[0]["rev"]), "git history")
            new["builtin_version"] = version_label(candidates[0])
        else:
            # no single version is written as fact when the deploy time is not known
            per = [build(builtin_at(v["rev"]), UNCERTAIN) for v in candidates]
            new = evidence.merge_uncertain(per, [version_label(v) for v in candidates])
    if new["source"] == "agent":
        new.update(evidence.path_check(os.path.join(common.ROOT, str(new.get("path"))), new.get("hash")))
    merged = dict(old)
    merged.update(new)
    for key in PASSING_KEYS:
        if key in merged and key not in new:
            del merged[key]  # a note of an earlier run that no longer holds
    rec["selection_data"] = merged
    rec.pop("selection_note", None)  # write_part left it when it could not build selection_data
    rec["enriched_at"] = stamp
    return True


# ---------- routing (the only engine path) ----------

def routing_plan(short, rows, states, retry=False):
    # Decides for each expert row without routing what to do, with no engine call. Returns
    # (calls, out): calls is a list of (row, part, extra_skills) to send to the engine, and out is
    # {row id: (None, note)} for the rows that no engine call can serve.
    import torch

    calls = []
    out = {}
    for r in rows:
        if not (from_mining(r) and r.get("type") == "expert" and r.get("routing") is None):
            continue
        if not retry and str(r.get("routing_note") or "").startswith(evidence.FINAL):
            continue  # the answer would be the same: no call, and the row stays as it is
        target = r["skill"]["name"]
        if not r.get("revision"):
            out[r["id"]] = (None, evidence.FINAL + "the row has no revision, so the weights it was mined on are not known")
            continue
        try:
            idx = torch.load(os.path.join(common.ROOT, r["units"]))["indices"].tolist()
            part = {"kind": "expert", "layer": int(r["layer"]), "indices": idx}
        except Exception as e:
            out[r["id"]] = (None, evidence.FINAL + "could not read the part file: " + type(e).__name__ + ": " + str(e)[:160])
            continue
        extra = {}
        if r["skill"].get("source") == "agent":
            try:
                with open(os.path.join(common.ROOT, r["skill"]["path"])) as f:
                    data = json.load(f)
                extra[target] = {"hash": data["hash"], "items": data["items"], "control": data["control"]}
            except Exception as e:
                out[r["id"]] = (None, evidence.FINAL + "could not read the frozen test of the skill: "
                                + type(e).__name__ + ": " + str(e)[:160])
                continue
            if data["hash"] != r["skill"].get("hash"):
                # another lane replaced the file: its items are not the text this row was selected on
                out[r["id"]] = (None, evidence.FINAL + "the frozen test file has hash " + str(data["hash"])
                                + ", which differs from the row's hash " + str(r["skill"].get("hash")))
                continue
        elif target not in common.BUILTIN_TARGETS + common.CONTROL_ONLY:
            out[r["id"]] = (None, evidence.FINAL + "the target skill is neither a built-in set of the engine"
                            " nor an agent skill with a frozen test")
            continue
        reason = None
        for state in states:
            structure = state.get("structure")
            if state.get("revision") == r["revision"] and isinstance(structure, dict):
                reason = reason or evidence.client_unsupported(structure.get("layers_detail"), r["layer"])
        if reason:
            out[r["id"]] = (None, evidence.FINAL + reason)
            continue
        calls.append((r, part, extra))
    return calls, out


def measure_routing(short, rows, retry=False):
    # {row id: (routing, note)} for expert rows without routing. Calls the GPU engine, with one
    # engine handle for each revision that the rows name.
    calls, out = routing_plan(short, rows, state_files(short), retry)
    revisions = []
    for r, part, extra in calls:
        if r["revision"] not in revisions:
            revisions.append(r["revision"])
    print("  routing:", len(calls), "engine call(s) for", len(revisions), "revision(s);", len(out),
          "row(s) settled without a call")
    if not calls:
        return out
    import client

    seconds = 0.0
    for revision in revisions:
        group = [c for c in calls if c[0]["revision"] == revision]
        try:
            miner = client.Miner(short, revision=revision)
        except (SystemExit, Exception) as e:
            for r, part, extra in group:
                out[r["id"]] = (None, "no routing: could not reach the engine: " + type(e).__name__ + ": " + str(e)[:160])
            continue
        for r, part, extra in group:
            out[r["id"]] = miner.expert_routing(part, r["skill"]["name"], extra)
            print("  routing", r["id"], "ok" if out[r["id"]][0] is not None else out[r["id"]][1])
        seconds += miner.modal_seconds
    print("  engine seconds:", round(seconds, 1))
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


def enrich_model(short, routing=False, force=False, versions=None, healdata=None, stamp=None, retry_routing=False):
    # Returns a short status text; it starts with "refused" or "skipped" when nothing was written.
    if short.endswith(".rejected"):
        return "refused: rejected files are never touched"
    path = common.catalog_path(short)
    if not os.path.exists(path):
        return "skipped: no catalog file"
    if not os.path.isdir(common.logs_dir(short)) and not force:
        return ("refused: no logs/" + short + "/ folder on this machine, so a run on another machine cannot be"
                " ruled out (run the backfill where the model is mined, or pass --force when no run is active)")
    recent = live_files(short)
    if recent and not force:
        return ("refused: looks live, changed in the last " + str(LIVE_SECONDS // 60) + " minutes: "
                + ", ".join(recent[:4]) + " (pass --force when no run is active)")
    stamp = stamp or time.strftime(TIME_FORMAT)
    measured = {}
    if routing:
        measured = measure_routing(short, read_rows(path), retry_routing)
    with open(path + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            rows = read_rows(path)  # read again under the lock: a run may have appended a row
            agent = agent_tests(short, rows, state_files(short))
            done = 0
            uncertain = 0
            for rec in rows:
                if not enrich_row(rec, agent, versions or [], healdata, stamp):
                    continue
                done += 1
                if rec["selection_data"].get("builtin_basis") == UNCERTAIN:
                    uncertain += 1
                if rec.get("id") in measured:
                    rec["routing"] = measured[rec["id"]][0]
                    rec.pop("routing_note", None)
                    if measured[rec["id"]][1]:
                        rec["routing_note"] = measured[rec["id"]][1]
            write_rows(path, rows)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    status = "enriched " + str(done) + " of " + str(len(rows)) + " rows"
    if uncertain:
        status += ("; " + str(uncertain) + " with an uncertain built-in file version (reported_benchmarks is the"
                   " union over the versions; pass --live-since to settle it)")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="short name; repeat for more models")
    ap.add_argument("--routing", action="store_true", help="fill routing for expert rows (calls the GPU engine)")
    ap.add_argument("--retry-routing", action="store_true",
                    help="with --routing: ask again for rows whose routing_note names a final reason")
    ap.add_argument("--force", action="store_true",
                    help="run although logs/<model>/ changed in the last 10 minutes or is not on this machine")
    ap.add_argument("--live-since", action="append", default=[], metavar="COMMIT=TIME",
                    help="when the engine began to use the built-in file of that commit: "
                         "<commit>=<YYYY-MM-DDTHH:MM:SS>, local time")
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
        status = enrich_model(short, routing=args.routing, force=args.force, versions=versions, healdata=healdata,
                              retry_routing=args.retry_routing)
        print("[" + short + "]", status)
        if status.startswith("refused"):
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
