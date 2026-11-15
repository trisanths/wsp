"""The scrolling agent log: what each agent tried, why, and what it scored.

Examples:
    python scripts/agent_log.py outputs/runs/<run_id>                 # print every eval row
    python scripts/agent_log.py outputs/runs/<run_id> --follow        # keep printing new rows
    python scripts/agent_log.py outputs/compare/<name> --html outputs/agent_log.html

Inputs are experiments.jsonl files, run folders, or compare folders (every
run in the manifest). Each eval row shows the eval count, agent, hypothesis,
program (describe), fit_r2 per graft, score and status. The terminal shows
rows oldest first; the HTML page shows them newest first.

Scores carry their split (SEARCH or HELDOUT) and STUB when they come from the
stub backend, so no stub or search number reads as a result. The HTML page
is one static file: inline CSS, no scripts, no external assets, light and
dark mode.
"""

import argparse
import datetime
import html
import json
import os
import sys
import time


# Reading.

def parse_line(line):
    line = line.strip()
    if line == "":
        return None
    try:
        return json.loads(line)
    except ValueError:
        return None


def read_rows(path):
    """Read a JSONL log, skipping blank lines and a half-written last line."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = parse_line(line)
            if row is not None:
                rows.append(row)
    return rows


def log_paths(path):
    """The experiments.jsonl files behind a file, a run folder or a compare folder."""
    if not os.path.isdir(path):
        return [path]
    own = os.path.join(path, "experiments.jsonl")
    if os.path.exists(own):
        return [own]
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        return [own]
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    paths = []
    for record in manifest.get("runs", []):
        log_path = record.get("log_path")
        if log_path is not None:
            paths.append(log_path)
    return paths


def eval_rows(rows):
    out = []
    for row in rows:
        if row.get("type") == "eval":
            out.append(row)
    return out


def time_order(rows):
    """Oldest first, by timestamp; rows with the same timestamp keep their log order."""
    return sorted(rows, key=timestamp_of)


def timestamp_of(row):
    value = row.get("timestamp")
    if value is None:
        return ""
    return str(value)


def is_stub_row(row):
    return row.get("stub") is True or row.get("backend") == "stub"


def has_stub(rows):
    for row in rows:
        if is_stub_row(row):
            return True
    return False


# Formatting shared by both views.


def split_label(row):
    if row.get("split") == "heldout":
        return "HELDOUT"
    return "SEARCH"


def score_text(row):
    """The combined score with its labels, for example "0.3111 SEARCH STUB", or "-" when none ran."""
    value = row.get("combined")
    if value is None:
        return "-"
    text = "%.4f" % value + " " + split_label(row)
    if row.get("fidelity") not in (None, "full"):
        text += " (" + str(row["fidelity"]) + ")"
    if is_stub_row(row):
        text += " STUB"
    return text


def fit_text(row):
    """Per-graft fit_r2, for example "stub_qmath_l14_c0@8 0.79", or "-" with no grafts."""
    fits = row.get("fit_r2") or {}
    if len(fits) == 0:
        return "-"
    parts = []
    for key in sorted(fits):
        value = fits[key]
        if value is None:
            parts.append(key + " skipped")
        else:
            parts.append(key + " " + "%.2f" % value)
    return ", ".join(parts)


def eval_text(row):
    value = row.get("eval_count")
    if value is None:
        return "-"
    return "%.2f" % float(value)


def status_text(row):
    status = str(row.get("status"))
    extra = row.get("extra") or {}
    if status == "control" and extra.get("control"):
        status += " (" + str(extra["control"]) + ")"
    return status


# Terminal view.

def terminal_lines(row, show_run=False):
    """Three lines for one eval row: counts and score, the program, the hypothesis."""
    head = "[" + eval_text(row).rjust(6) + "] " + str(row.get("agent")).ljust(13) + " "
    head += status_text(row).ljust(14) + " " + score_text(row).ljust(27) + " fit " + fit_text(row)
    if show_run:
        head += "  (" + str(row.get("run_id")) + ")"
    lines = [head, "          " + str(row.get("describe") or "-")]
    hypothesis = row.get("hypothesis")
    if hypothesis:
        lines.append("          why: " + str(hypothesis))
    return lines


def print_rows(rows, out, show_run=False):
    for row in rows:
        for line in terminal_lines(row, show_run):
            out.write(line + "\n")
    out.flush()


def header_text(paths, rows):
    text = "agent log: " + ", ".join(paths) + " (" + str(len(rows)) + " eval rows)"
    if has_stub(rows):
        text += " [STUB]"
    return text


class LogTail:
    """Reads new complete rows from a growing JSONL file. A partial last line waits for its end."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = b""

    def poll(self):
        """Return the rows appended since the last poll ([] if the file does not exist yet).

        Reads bytes, so the offset is exact and a line cut mid-character waits whole.
        """
        if not os.path.exists(self.path):
            return []
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
            self.offset = handle.tell()
        data = self.pending + data
        lines = data.split(b"\n")
        self.pending = lines[-1]
        rows = []
        for line in lines[:-1]:
            row = parse_line(line.decode("utf-8", errors="replace"))
            if row is not None:
                rows.append(row)
        return rows


def follow(path, out, poll_s=1.0, max_polls=None):
    """Print eval rows as they are appended to one run file. Stops on Ctrl-C (or after max_polls)."""
    tail = LogTail(path)
    out.write("following " + path + " (Ctrl-C to stop)\n")
    polls = 0
    try:
        while max_polls is None or polls < max_polls:
            print_rows(eval_rows(tail.poll()), out)
            polls += 1
            if max_polls is None or polls < max_polls:
                time.sleep(poll_s)
    except KeyboardInterrupt:
        out.write("stopped\n")
    return tail


# HTML view.

PAGE_CSS = """
:root { color-scheme: light; --bg: #fcfcfb; --surface: #ffffff; --text: #0b0b0b; --muted: #52514e;
  --line: #e4e3df; --head: #f0efec; --banner-bg: #fff4d6; --banner-text: #5c4000; --banner-line: #fab219;
  --good: #0ca30c; --warn: #fab219; --bad: #d03b3b; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { color-scheme: dark; --bg: #1a1a19; --surface: #222220; --text: #f0efec;
    --muted: #c3c2b7; --line: #383835; --head: #2b2b29; --banner-bg: #3a2f10; --banner-text: #ffe2a3;
    --banner-line: #fab219; }
}
:root[data-theme="dark"] { color-scheme: dark; --bg: #1a1a19; --surface: #222220; --text: #f0efec;
  --muted: #c3c2b7; --line: #383835; --head: #2b2b29; --banner-bg: #3a2f10; --banner-text: #ffe2a3;
  --banner-line: #fab219; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
p.meta { color: var(--muted); margin: 0 0 16px; }
.banner { background: var(--banner-bg); color: var(--banner-text); border: 1px solid var(--banner-line);
  border-radius: 6px; padding: 10px 14px; margin: 0 0 16px; font-weight: 600; }
.wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; min-width: 900px; }
th, td { text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { background: var(--head); color: var(--muted); font-weight: 600; font-size: 12px; position: sticky; top: 0; }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
td.fit, td.program { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12px; }
td.why { color: var(--muted); max-width: 420px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; background: var(--line); }
.dot.ok { background: var(--good); }
.dot.collapsed, .dot.infra_error, .dot.compile_error { background: var(--bad); }
.dot.rejected_fit, .dot.budget_exhausted { background: var(--warn); }
"""


def status_class(row):
    status = str(row.get("status"))
    safe = ""
    for char in status:
        if char.isalnum() or char == "_":
            safe += char
    return safe


def html_row(row, show_run):
    cells = []
    cells.append('<td class="num">' + html.escape(eval_text(row)) + "</td>")
    if show_run:
        cells.append("<td>" + html.escape(str(row.get("run_id"))) + "</td>")
    cells.append("<td>" + html.escape(str(row.get("agent"))) + "</td>")
    cells.append('<td class="why">' + html.escape(str(row.get("hypothesis") or "")) + "</td>")
    cells.append('<td class="program">' + html.escape(str(row.get("describe") or "-")) + "</td>")
    cells.append('<td class="fit">' + html.escape(fit_text(row)) + "</td>")
    cells.append('<td class="num">' + html.escape(score_text(row)) + "</td>")
    cells.append('<td><span class="dot ' + status_class(row) + '"></span>' + html.escape(status_text(row)) + "</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def build_html(rows, sources, title="Agent log"):
    """The whole page as a string. rows are eval rows in time order; the page lists them newest first."""
    run_ids = set()
    for row in rows:
        run_ids.add(row.get("run_id"))
    show_run = len(run_ids) > 1
    heads = ["eval (FEE)"]
    if show_run:
        heads.append("run")
    for name in ["agent", "hypothesis", "program", "fit_r2", "score", "status"]:
        heads.append(name)
    head_cells = ""
    for name in heads:
        head_cells += "<th>" + html.escape(name) + "</th>"
    body_rows = []
    for row in reversed(rows):
        body_rows.append(html_row(row, show_run))
    made = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = ['<!doctype html>', '<html lang="en">', "<head>", '<meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             "<title>" + html.escape(title) + "</title>", "<style>" + PAGE_CSS + "</style>", "</head>",
             "<body>", "<main>", "<h1>" + html.escape(title) + "</h1>",
             '<p class="meta">' + str(len(rows)) + " eval rows, newest first. Built " + made + " from " +
             html.escape(", ".join(sources)) + ". SEARCH scores are search-split checks, not results; "
             "HELDOUT rows are scored once at the end.</p>"]
    if has_stub(rows):
        parts.append('<div class="banner" role="note">STUB: these rows come from the stub backend. '
                     "They test the plumbing and are not results.</div>")
    parts.append('<div class="wrap"><table><thead><tr>' + head_cells + "</tr></thead><tbody>")
    parts.append("\n".join(body_rows))
    parts.append("</tbody></table></div>")
    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def write_html(path, rows, sources, title="Agent log"):
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(build_html(rows, sources, title))
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Show the agent log in the terminal or as a static HTML page.")
    parser.add_argument("paths", nargs="+", help="experiments.jsonl files, run folders or compare folders")
    parser.add_argument("--follow", action="store_true", help="keep printing rows appended to one run file")
    parser.add_argument("--poll-s", type=float, default=1.0, help="seconds between reads with --follow")
    parser.add_argument("--html", default=None, help="write a static HTML page to this path")
    parser.add_argument("--title", default="Agent log")
    return parser.parse_args(argv)


def main(argv=None, out=None):
    args = parse_args(argv)
    if out is None:
        out = sys.stdout
    paths = []
    for path in args.paths:
        for log_path in log_paths(path):
            paths.append(log_path)
    if args.follow:
        if len(paths) != 1:
            raise SystemExit("agent_log: --follow takes exactly one run file")
        follow(paths[0], out, args.poll_s)
        return None
    rows = []
    for path in paths:
        if not os.path.exists(path):
            out.write("missing log: " + path + "\n")
            continue
        for row in eval_rows(read_rows(path)):
            rows.append(row)
    rows = time_order(rows)
    if args.html is not None:
        written = write_html(args.html, rows, paths, args.title)
        out.write("wrote " + written + "\n")
        return written
    run_ids = set()
    for row in rows:
        run_ids.add(row.get("run_id"))
    out.write(header_text(paths, rows) + "\n")
    print_rows(rows, out, len(run_ids) > 1)
    return rows


if __name__ == "__main__":
    main()
