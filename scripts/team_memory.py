"""Read and write the team's shared past.dev memory from the command line.

Every teammate's agents share one past.dev audience ("wsp-mining", see
person_a/mine/shared_memory.py). This script lets any agent or person
recall what the others learned and record a finding, without writing
client code. The key comes from PAST_API_KEY in the repo's .env file. It
is never printed.

    .venv/bin/python scripts/team_memory.py recall "which experts carry math"
    .venv/bin/python scripts/team_memory.py record finding "Run 14b: ..." --who head --topic composition
    .venv/bin/python scripts/team_memory.py record-file notes.jsonl --who head

record-file takes JSON lines: {"kind": "finding", "text": "...", "topic": "..."}.
Kinds: finding (a measured result), lesson (a takeaway), plan (what is
running or decided), note. Put the numbers and the control in the text.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "person_a", "mine"))


def load_key():
    """Put PAST_API_KEY from the repo .env into the environment when it is not set."""
    if os.environ.get("PAST_API_KEY"):
        return
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("PAST_API_KEY="):
                os.environ["PAST_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def memory(who, topic):
    import shared_memory

    # shared_memory.load_env reads person_a/.env with override. Keep the repo key when that file has none.
    key = os.environ.get("PAST_API_KEY")
    mem = shared_memory.SharedMemory(who, topic, "t" + str(int(time.time())))
    if not mem.enabled and key:
        os.environ["PAST_API_KEY"] = key
        mem.key = key
        mem.enabled = mem._ensure_audience()
    return mem


def send(mem, rows):
    sent = 0
    for row in rows:
        mem.record(row.get("kind", "note"), row["text"], {"topic": row.get("topic", "")})
    while len(mem.buffer) > 0:
        out = mem.flush()
        sent += out.get("sent", 0)
        if out.get("sent", 0) == 0:
            break
    return sent


def main():
    parser = argparse.ArgumentParser(description="Shared team memory on past.dev")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("recall")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=15)
    p = sub.add_parser("record")
    p.add_argument("kind", choices=["finding", "lesson", "plan", "note"])
    p.add_argument("text")
    p.add_argument("--who", default="head")
    p.add_argument("--topic", default="sapien")
    p = sub.add_parser("record-file")
    p.add_argument("path")
    p.add_argument("--who", default="head")
    p.add_argument("--topic", default="sapien")
    args = parser.parse_args()

    load_key()
    if args.command == "recall":
        mem = memory("cli", "recall")
        if not mem.enabled:
            print("memory is off:", mem.last_error or "PAST_API_KEY missing")
            return 1
        results = mem.recall(args.query, limit=args.limit)
        if len(results) == 0:
            print("(nothing recalled)")
        for item in results:
            print("- " + item["text"][:700].replace("\n", " "))
        return 0

    mem = memory(args.who, args.topic)
    if not mem.enabled:
        print("memory is off:", mem.last_error or "PAST_API_KEY missing")
        return 1
    if args.command == "record":
        rows = [{"kind": args.kind, "text": args.text, "topic": args.topic}]
    else:
        rows = []
        with open(args.path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    print("sent", send(mem, rows), "of", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
