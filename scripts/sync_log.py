"""Merge every run log into the shared experiments.jsonl at the repo root.

Example:
    python scripts/sync_log.py                    # outputs/runs/*/experiments.jsonl -> experiments.jsonl
    python scripts/sync_log.py --modal            # also merge the Modal wsp-exp Dict

Each run writes only its own outputs/runs/<run_id>/experiments.jsonl, so no
two processes ever append to one file (spec section 4.3). This script is the
only writer of the shared file. It reads the shared file as it is now, every
run file, and (with --modal) the rows in the Modal Dict, then writes the
union sorted by timestamp.

De-duplication key: (run_id, type, split, token or seq, timestamp). Rows that
spent an eval token use the token; the split tells the search and held-out
pools apart, since both number their tokens from 0. Other rows (probes, events, cache hits,
invalid programs) use seq, the row's 0-based line number in its run file,
or the number after the last ":" in a Modal Dict key ("<run_id>:<seq>").
seq is written into each merged row, so running the sync again finds the
same keys and adds nothing twice. The file is replaced in one step
(written to a temp file, then renamed).
"""

import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def read_jsonl(path):
    """Rows of a JSONL file, plus the count of lines that were not valid JSON (a partial last write)."""
    rows = []
    bad = 0
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


def run_files(outputs_dir):
    """Every outputs/runs/*/experiments.jsonl, in a fixed order."""
    pattern = os.path.join(outputs_dir, "runs", "*", "experiments.jsonl")
    return sorted(glob.glob(pattern))


def with_seq(rows):
    """Copies of rows with "seq" set to the line number where a row has none."""
    out = []
    for index in range(len(rows)):
        row = rows[index]
        if row.get("seq") is None:
            row = dict(row)
            row["seq"] = index
        out.append(row)
    return out


def dedupe_key(row):
    """(run_id, type, split, ("token", n) or ("seq", n), timestamp).

    The split is in the key because the search and held-out pools both
    number their tokens from 0: a run's held-out token 0 is a different row
    from its search token 0, even when both carry the same timestamp.
    """
    token = row.get("token")
    if token is not None:
        ident = ("token", token)
    else:
        ident = ("seq", row.get("seq"))
    return (row.get("run_id"), row.get("type"), row.get("split"), ident, row.get("timestamp"))


def sort_key(row):
    """Timestamp first; within one timestamp, run id then seq keep a run's own order."""
    seq = row.get("seq")
    if not isinstance(seq, int):
        seq = -1
    return (str(row.get("timestamp") or ""), str(row.get("run_id") or ""), seq)


def merge(sources):
    """Union of several row lists, the first copy of each key kept, sorted by timestamp.

    Returns (rows, duplicates dropped).
    """
    seen = set()
    merged = []
    duplicates = 0
    for rows in sources:
        for row in rows:
            key = json.dumps(dedupe_key(row), sort_keys=True, default=str)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            merged.append(row)
    merged.sort(key=sort_key)
    return merged, duplicates


def write_jsonl(path, rows):
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp, path)
    return path


def seq_from_key(key):
    """The seq at the end of a Modal Dict key "<run_id>:<seq>", or None."""
    text = str(key)
    if ":" not in text:
        return None
    tail = text.rsplit(":", 1)[1]
    if tail.isdigit():
        return int(tail)
    return None


def rows_from_dict_items(items):
    """Log rows from (key, value) pairs of the wsp-exp Dict; values that are not dicts are skipped."""
    rows = []
    for key, value in items:
        if not isinstance(value, dict):
            continue
        row = dict(value)
        if row.get("seq") is None:
            row["seq"] = seq_from_key(key)
        rows.append(row)
    return rows


def modal_rows(dict_name):
    """Rows from the Modal Dict, or (None, reason) when modal or the Dict is not available."""
    try:
        import modal
    except ImportError:
        return None, "modal is not installed; skipped the " + dict_name + " merge"
    from wsp.swarm.config import load_dotenv

    load_dotenv()
    try:
        handle = modal.Dict.from_name(dict_name)
        items = list(handle.items())
    except Exception as error:
        return None, "could not read Modal Dict " + dict_name + " (" + type(error).__name__ + "); skipped it"
    return rows_from_dict_items(items), None


def sync(outputs_dir, out_path, extra_sources=None):
    """Merge the shared file, every run file and any extra row lists into out_path. Returns counts."""
    existing, bad_existing = read_jsonl(out_path)
    sources = [existing]
    bad = bad_existing
    files = run_files(outputs_dir)
    for path in files:
        rows, bad_rows = read_jsonl(path)
        bad += bad_rows
        sources.append(with_seq(rows))
    if extra_sources is not None:
        for rows in extra_sources:
            sources.append(rows)
    total = 0
    for rows in sources:
        total += len(rows)
    merged, duplicates = merge(sources)
    write_jsonl(out_path, merged)
    return {"path": out_path, "run_files": len(files), "rows_read": total, "rows_written": len(merged),
            "duplicates": duplicates, "bad_lines": bad, "already_there": len(existing)}


def modal_note(extra):
    if len(extra) == 0:
        return ""
    return " and the Modal Dict"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge run logs into the shared experiments.jsonl.")
    parser.add_argument("--outputs-dir", default=os.path.join(REPO_ROOT, "outputs"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "experiments.jsonl"))
    parser.add_argument("--modal", action="store_true", help="also merge the Modal wsp-exp Dict")
    parser.add_argument("--modal-dict", default="wsp-exp")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extra = []
    if args.modal:
        rows, reason = modal_rows(args.modal_dict)
        if rows is None:
            print(reason)
        else:
            print("Modal Dict " + args.modal_dict + ": " + str(len(rows)) + " rows")
            extra.append(rows)
    else:
        print("Modal Dict merge off (pass --modal to include " + args.modal_dict + ")")
    result = sync(args.outputs_dir, args.out, extra)
    print("read " + str(result["rows_read"]) + " rows (" + str(result["already_there"]) + " already in the shared log, " +
          "the rest from " + str(result["run_files"]) + " run files" + modal_note(extra) + "); dropped " +
          str(result["duplicates"]) + " duplicates; wrote " + str(result["rows_written"]) + " rows to " + result["path"])
    if result["bad_lines"] > 0:
        print("skipped " + str(result["bad_lines"]) + " lines that were not valid JSON (a run may still be writing)")
    return result


if __name__ == "__main__":
    main()
