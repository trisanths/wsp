"""Side-by-side demo for a developer: the full Qwen3-30B and the code-slimmed model on the same requests.

Request this demos: "I need a coding assistant as good as Qwen3-30B at code, at half the memory. I do not care
about anything else." (demo/requests/code_half_memory.json). The slimmed model is slim_64: 64 of 128 experts per
layer kept by code saliency, no heal (results/README.md, Run 19; modal_round19.py).

Eight prompts go through both models with vLLM: six coding requests a developer would send (function with
tests, bug fix, regex explanation, SQL from a schema, class refactor, bash one-liner) and two non-code
requests (history, medicine) that show what was removed. Greedy decoding, the chat template of
wsp.skills_eval.render_prompt, 512 new tokens.

Coding prompts are graded by running code: the model's answer is cut into code blocks, and hidden checks run
in a subprocess in the container (Python asserts, SQLite against a fixture, bash against a fixture log). The
regex prompt and the two non-code prompts have no test; they get a keyword check that is only a hint, and the
text is there to read. One run, greedy, so one sample per prompt: it shows behaviour, it is not a benchmark.

Smoke (one prompt, one cell):

    modal run modal_demo.py --smoke

Everything (one H200 container per cell):

    modal run --detach modal_demo.py

Rows go to /vol/results/demo/<cell>.json on the volume and, after each cell, to results/demo_code_<cell>.json
here. The side-by-side files results/demo_code_outputs.json and results/demo_code_outputs.md are written when
both cells are present. "modal run modal_demo.py --collect-only" rebuilds them from the volume.
"""

import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

import modal

app = modal.App("wsp-demo")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"

# The image of modal_serve.py (a script that ran on 2026-11-15).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

RESULT_DIR = "/vol/results/demo"
LOCAL_DIR = "results"
SLIM_ROOT = "/vol/slim_models"
MOE_BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_NEW_TOKENS = 512
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
CODE_VERSION = "demo_v1"

CELLS = {
    "full_30b": {"kind": "hub", "model": MOE_BASE, "label": "Qwen3-30B-A3B full (128 experts per layer)"},
    "slim_64": {"kind": "path", "dir": "round19/moe_saliency_64", "tokenizer": MOE_BASE,
                "label": "slim_64 (64 experts per layer by code saliency, no heal)"},
}

# ---------------------------------------------------------------- prompts and fixtures

CART_CODE = '''class Cart:
    def __init__(self):
        self.items = []

    def add(self, name, price, qty):
        self.items.append((name, price, qty))

    def total(self, coupon=None):
        t = 0
        for it in self.items:
            t = t + it[1] * it[2]
        if coupon == "TEN":
            t = t - t * 0.1
        elif coupon == "FIVE":
            t = t - t * 0.05
        elif coupon == "HALF":
            t = t - t * 0.5
        return t

    def count(self):
        c = 0
        for it in self.items:
            c = c + it[2]
        return c
'''

AVERAGE_CODE = '''def moving_average(values, window):
    result = []
    for i in range(len(values) - window):
        result.append(sum(values[i:i + window]) / window)
    return result
'''

REGEX = (r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])-(?P<day>0[1-9]|[12]\d|3[01])"
         r"(?:T(?P<time>[01]\d|2[0-3]):[0-5]\d)?$")

SQL_SCHEMA = """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL, created_at TEXT);"""

SQL_ROWS = """
INSERT INTO customers VALUES (1,'Ana','US'),(2,'Ben','US'),(3,'Chloe','DE'),(4,'Dev','US'),(5,'Eli','US'),(6,'Fay','US');
INSERT INTO orders VALUES
 (1,1,100.0,'2024-02-01'),(2,1,50.0,'2024-11-30'),(3,1,900.0,'2023-12-31'),
 (4,2,400.0,'2024-05-05'),(5,3,5000.0,'2024-06-06'),(6,4,120.0,'2024-01-01'),
 (7,4,80.0,'2024-12-31'),(8,5,60.0,'2024-03-03'),(9,5,70.0,'2025-01-01'),
 (10,6,250.0,'2024-07-07'),(11,2,10.0,'2024-08-08');
"""

SQL_REFERENCE = """SELECT c.name, SUM(o.total) AS total_spend FROM customers c JOIN orders o ON o.customer_id = c.id
WHERE c.country = 'US' AND o.created_at >= '2024-01-01' AND o.created_at < '2025-01-01'
GROUP BY c.id ORDER BY total_spend DESC LIMIT 3"""

LOG_LINES = (["10.0.0.7"] * 9 + ["10.0.0.3"] * 7 + ["192.168.1.20"] * 6 + ["172.16.5.5"] * 4
             + ["10.0.0.99"] * 3 + ["8.8.8.8"] * 2 + ["1.1.1.1"])
LOG_TOP5 = [("10.0.0.7", 9), ("10.0.0.3", 7), ("192.168.1.20", 6), ("172.16.5.5", 4), ("10.0.0.99", 3)]

PROMPTS = [
    {"id": "code_function_tests", "kind": "code", "title": "Write a function with tests",
     "text": "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs and "
             "returns the merged list of non-overlapping intervals, sorted by start. Intervals that touch, such "
             "as [1, 2] and [2, 3], are merged. Include pytest tests."},
    {"id": "code_fix_bug", "kind": "code", "title": "Fix a bug in a pasted snippet",
     "text": "This function should return one average per full window, but it returns one value too few. "
             "Fix it and say what was wrong.\n\n```python\n" + AVERAGE_CODE + "```"},
    {"id": "code_explain_regex", "kind": "code", "title": "Explain a regex",
     "text": "Explain what this regular expression matches, piece by piece, and give two strings it accepts "
             "and two it rejects.\n\n```\n" + REGEX + "\n```"},
    {"id": "code_sql", "kind": "code", "title": "SQL from a schema",
     "text": "Here is a SQLite schema:\n\n```sql\n" + SQL_SCHEMA + "\n```\n\nWrite one query that returns the "
             "three US customers with the highest total spend on orders created in 2024. Return two columns in "
             "this order: name, total_spend. Highest spend first."},
    {"id": "code_refactor", "kind": "code", "title": "Refactor a class",
     "text": "Refactor this class for readability. Use a dataclass for the line items, a dict for the coupons, "
             "and a generator sum. Keep the public API (add, total, count, and the constructor) and the "
             "behaviour exactly the same, including that an unknown coupon gives no discount. Give the full "
             "refactored code in one block.\n\n```python\n" + CART_CODE + "```"},
    {"id": "code_bash", "kind": "code", "title": "A bash one-liner",
     "text": "Give me a bash one-liner that prints the 5 most frequent IP addresses in access.log (the IP is "
             "the first field of each line), each with its count, most frequent first."},
    {"id": "other_history", "kind": "other", "title": "History question (removed skill)",
     "text": "Why did the Western Roman Empire end in 476, and what part did Odoacer play in it?",
     "hint": ["odoacer", "romulus", "476", "deposed|deposition|overthrew"]},
    {"id": "other_medical", "kind": "other", "title": "Medical question (removed skill)",
     "text": "A 58-year-old man has crushing chest pain that spreads to his left arm. His ECG shows ST "
             "elevation in leads II, III, and aVF. What is the diagnosis, which artery is most likely blocked, "
             "and what are the first treatments?",
     "hint": ["inferior", "right coronary|rca", "aspirin", "pci|percutaneous|reperfusion|thrombolysis"]},
]

# ---------------------------------------------------------------- grading


def code_blocks(text):
    """[(language, code)] for each fenced block. An unclosed last fence (cut by the token limit) still counts."""
    blocks = []
    for match in re.finditer(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", text, re.S):
        blocks.append((match.group(1).lower(), match.group(2)))
    return blocks


def run_process(args, cwd, timeout=20, stdin_text=None):
    """(returncode, stdout, stderr). A timeout gives returncode -9."""
    try:
        done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, input=stdin_text)
        return done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired:
        return -9, "", "timeout"


def run_python(code, cwd, timeout=20):
    path = os.path.join(cwd, "snippet.py")
    with open(path, "w") as f:
        f.write(code)
    return run_process([sys.executable, path], cwd, timeout)


def python_blocks(text):
    return [code for lang, code in code_blocks(text) if lang in ("python", "py", "python3", "")]


def pick_block_with(text, needle, languages=("python", "py", "python3", "")):
    """The last code block of the given languages that contains needle, or None."""
    found = None
    for lang, code in code_blocks(text):
        if lang in languages and needle in code:
            found = code
    return found


def function_source(code, name):
    """Imports (without pytest) and the function `name` from a module source, as one string; None if absent."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    parts = []
    found = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + [getattr(node, "module", None) or ""]
            if any(item.split(".")[0] == "pytest" for item in names):
                continue
            parts.append(ast.get_source_segment(code, node))
        elif isinstance(node, ast.FunctionDef) and node.name == name:
            parts.append(ast.get_source_segment(code, node))
            found = True
    return "\n".join(parts) if found else None


def verdict(passed, detail):
    return {"passed": passed, "detail": detail}


def grade_merge_intervals(text):
    code = pick_block_with(text, "def merge_intervals")
    has_tests = "def test_" in text
    source = function_source(code, "merge_intervals") if code else None
    if not source:
        return verdict(False, "no merge_intervals function found in a code block")
    checks = (
        "assert [list(x) for x in merge_intervals([[1,3],[2,6],[8,10],[15,18]])] == [[1,6],[8,10],[15,18]]\n"
        "assert [list(x) for x in merge_intervals([[1,4],[4,5]])] == [[1,5]]\n"
        "assert [list(x) for x in merge_intervals([])] == []\n"
        "assert [list(x) for x in merge_intervals([[5,6],[1,2]])] == [[1,2],[5,6]]\n"
        "assert [list(x) for x in merge_intervals([[1,10],[2,3]])] == [[1,10]]\n"
        "print('ok')\n")
    with tempfile.TemporaryDirectory() as cwd:
        rc, out, err = run_python(source + "\n" + checks, cwd)
    if rc != 0:
        return verdict(False, "hidden asserts failed: " + err.strip().splitlines()[-1][:160] if err.strip()
                       else "hidden asserts failed")
    if not has_tests:
        return verdict(False, "function correct, but the answer has no test functions")
    return verdict(True, "hidden asserts pass and the answer includes test functions")


def grade_moving_average(text):
    code = pick_block_with(text, "def moving_average")
    source = function_source(code, "moving_average") if code else None
    if not source:
        return verdict(False, "no moving_average function found in a code block")
    checks = ("assert moving_average([1,2,3,4], 2) == [1.5, 2.5, 3.5]\n"
              "assert moving_average([1,2,3], 3) == [2.0]\n"
              "assert moving_average([1], 2) == []\n"
              "assert moving_average([4,4,4,4,4], 1) == [4,4,4,4,4]\nprint('ok')\n")
    with tempfile.TemporaryDirectory() as cwd:
        rc, out, err = run_python(source + "\n" + checks, cwd)
    if rc != 0:
        return verdict(False, "hidden asserts failed")
    return verdict(True, "hidden asserts pass")


def keyword_check(text, groups):
    """(hits, total) where each group is a regex alternation matched case-insensitively."""
    lowered = text.lower()
    hits = [group for group in groups if re.search(group, lowered)]
    return hits, len(groups)


def grade_regex(text):
    hits, total = keyword_check(text, [r"year", r"month", r"day", r"optional|\?", r"23|hour", r"31"])
    return verdict(None, f"keyword hint only, no test: {len(hits)} of {total} expected terms present")


def grade_sql(text):
    blocks = [code for lang, code in code_blocks(text) if lang in ("sql", "sqlite", "")]
    if not blocks:
        return verdict(False, "no sql code block")
    query = blocks[0].strip()
    db = sqlite3.connect(":memory:")
    db.executescript(SQL_SCHEMA + SQL_ROWS)
    expected = [tuple(row) for row in db.execute(SQL_REFERENCE).fetchall()]
    try:
        cursor = db.execute(query.rstrip(";"))
        got = [tuple(row) for row in cursor.fetchall()]
        columns = [item[0] for item in cursor.description]
    except Exception as error:
        return verdict(False, "query failed in SQLite: " + str(error)[:160])
    if got == expected and columns[:2] == ["name", "total_spend"]:
        return verdict(True, f"rows match the reference {expected}")
    return verdict(False, f"got columns {columns} rows {got}, expected {expected}")


def grade_cart(text):
    code = pick_block_with(text, "class Cart")
    if not code:
        return verdict(False, "no Cart class found in a code block")
    uses_dataclass = "dataclass" in code
    checks = ("__name__ = 'checked'\n" + code + "\n"
              "c = Cart()\nc.add('a', 10, 2)\nc.add('b', 5, 1)\n"
              "assert c.total() == 25\nassert abs(c.total('TEN') - 22.5) < 1e-9\n"
              "assert abs(c.total('FIVE') - 23.75) < 1e-9\nassert abs(c.total('HALF') - 12.5) < 1e-9\n"
              "assert c.total('BOGUS') == 25\nassert c.count() == 3\nassert Cart().total() == 0\nprint('ok')\n")
    with tempfile.TemporaryDirectory() as cwd:
        rc, out, err = run_python(checks, cwd)
    if rc != 0:
        return verdict(False, "behaviour changed or code failed: " + (err.strip().splitlines() or ["?"])[-1][:160])
    return verdict(True, "behaviour unchanged" + ("; uses a dataclass" if uses_dataclass else "; no dataclass"))


def grade_bash(text):
    candidates = [code for lang, code in code_blocks(text) if lang in ("bash", "sh", "shell", "zsh", "")]
    if not candidates:
        return verdict(False, "no bash code block")
    lines = [line.strip() for line in candidates[0].strip().splitlines() if line.strip()]
    lines = [line[2:] if line.startswith("$ ") else line for line in lines if not line.startswith("#")]
    command = "\n".join(lines)
    with tempfile.TemporaryDirectory() as cwd:
        with open(os.path.join(cwd, "access.log"), "w") as f:
            for number, ip in enumerate(LOG_LINES):
                f.write(f'{ip} - - [01/Jan/2026:00:00:{number % 60:02d} +0000] "GET /x HTTP/1.1" 200 12\n')
        rc, out, err = run_process(["bash", "-c", command], cwd)
    got = []
    for line in out.strip().splitlines():
        ip = re.search(r"\d+\.\d+\.\d+\.\d+", line)
        count = re.search(r"(?<![\d.])(\d+)(?![\d.])", line.replace(ip.group(0), "") if ip else line)
        if ip and count:
            got.append((ip.group(0), int(count.group(1))))
    if got == LOG_TOP5:
        return verdict(True, "output matches the top 5 with counts")
    return verdict(False, f"rc {rc}, parsed {got}, expected {LOG_TOP5}; stderr {err.strip()[:100]}")


GRADERS = {"code_function_tests": grade_merge_intervals, "code_fix_bug": grade_moving_average,
           "code_explain_regex": grade_regex, "code_sql": grade_sql, "code_refactor": grade_cart,
           "code_bash": grade_bash}


def grade_prompt(prompt, text):
    """{"passed": True/False/None, "detail": str}. None means no test exists for this prompt."""
    if prompt["id"] in GRADERS:
        try:
            return GRADERS[prompt["id"]](text)
        except Exception as error:
            return verdict(False, "grader error: " + repr(error)[:160])
    hits, total = keyword_check(text, prompt.get("hint", []))
    return verdict(None, f"keyword hint only, no test: {len(hits)} of {total} expected terms present")


# ---------------------------------------------------------------- the GPU function


def select_prompts(smoke):
    return PROMPTS[:1] if smoke else PROMPTS


def log(cell, message):
    print(f"[{cell}] {message}", flush=True)


def safe_reload():
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


def weights_gib(path):
    total = 0
    for name in os.listdir(path):
        if name.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(path, name))
    return total / 1024 ** 3


def demo_cell(cell, smoke=False):
    import transformers
    import vllm
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    spec = CELLS[cell]
    name = f"smoke_{cell}" if smoke else cell
    started = time.time()
    if spec["kind"] == "hub":
        local = snapshot_download(spec["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        tokenizer_id = spec["model"]
    else:
        local = os.path.join(SLIM_ROOT, spec["dir"])
        tokenizer_id = spec["tokenizer"]
        if not os.path.isdir(local):
            raise FileNotFoundError(f"{local} is not on this volume")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
    prompts = select_prompts(smoke)
    rendered = [skills_eval.render_prompt(tokenizer, item["text"]) for item in prompts]
    log(name, f"vllm {vllm.__version__}; weights on disk {weights_gib(local):.2f} GiB; {len(prompts)} prompts")
    llm = LLM(model=local, tokenizer=tokenizer_id, dtype="bfloat16", seed=0, gpu_memory_utilization=GPU_UTILIZATION,
              max_model_len=MAX_MODEL_LEN, enable_prefix_caching=False)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.generate(rendered, params, use_tqdm=False)
    rows = []
    for item, out in zip(prompts, outputs):
        text = out.outputs[0].text
        graded = grade_prompt(item, text)
        rows.append({"id": item["id"], "kind": item["kind"], "title": item["title"], "prompt": item["text"],
                     "output": text, "new_tokens": len(out.outputs[0].token_ids),
                     "finish_reason": out.outputs[0].finish_reason, "passed": graded["passed"],
                     "grade_detail": graded["detail"]})
        log(name, f"{item['id']}: passed={graded['passed']} ({graded['detail'][:120]}) tokens={rows[-1]['new_tokens']} "
                  f"finish={rows[-1]['finish_reason']}")
    return {"cell": name, "label": spec["label"], "model_path": local, "weights_gib_on_disk": round(weights_gib(local), 2),
            "vllm": vllm.__version__, "code_version": CODE_VERSION, "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy, chat template", "smoke": smoke, "seconds": round(time.time() - started, 1),
            "rows": rows}


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=2 * 3600, single_use_containers=True)
def run_cell(cell, smoke=False):
    safe_reload()
    try:
        result = demo_cell(cell, smoke)
    except Exception as error:
        log(cell, "failed: " + repr(error)[:400])
        result = {"cell": cell, "smoke": smoke, "error": repr(error)[:500], "rows": []}
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, f"{result['cell']}.json"), "w") as f:
        json.dump(result, f, indent=2)
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)
    return result


@app.function(image=modal.Image.debian_slim(python_version="3.12"), volumes={"/vol": volume}, timeout=600)
def read_results():
    safe_reload()
    found = {}
    if os.path.isdir(RESULT_DIR):
        for name in os.listdir(RESULT_DIR):
            if name.endswith(".json") and not name.startswith("smoke_"):
                with open(os.path.join(RESULT_DIR, name)) as f:
                    found[name[:-5]] = json.load(f)
    return found


# ---------------------------------------------------------------- the side-by-side files


def fence(text):
    """A code fence longer than any run of backticks in text."""
    longest = max([len(m) for m in re.findall(r"`+", text)] + [2])
    ticks = "`" * (longest + 1)
    return f"{ticks}text\n{text.rstrip()}\n{ticks}"


def mark(passed):
    return {True: "pass", False: "FAIL", None: "no test"}[passed]


def render_markdown(results):
    """results {cell: result}. A readable side-by-side file."""
    names = [name for name in CELLS if name in results and not results[name].get("error")]
    lines = ["# Code demo: full Qwen3-30B against the code-slimmed model", "",
             "Same eight prompts, greedy decoding, chat template, 512 new tokens, vLLM on one H200 each. "
             "One sample per prompt. Coding prompts are graded by running hidden checks on the answer; the regex "
             "prompt and the two non-code prompts have no test, only a keyword hint. Made by modal_demo.py.", ""]
    for name in names:
        r = results[name]
        lines.append(f"- {name}: {r['label']}; weights on disk {r['weights_gib_on_disk']} GiB.")
    lines += ["", "## Scores on the six coding prompts", "", "| prompt | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for item in PROMPTS:
        if item["kind"] != "code":
            continue
        cells = []
        for name in names:
            row = next((x for x in results[name]["rows"] if x["id"] == item["id"]), None)
            cells.append(mark(row["passed"]) if row else "missing")
        lines.append(f"| {item['title']} | " + " | ".join(cells) + " |")
    lines.append("")
    for item in PROMPTS:
        lines += [f"## {item['title']}", "", "Prompt:", "", fence(item["text"]), ""]
        for name in names:
            row = next((x for x in results[name]["rows"] if x["id"] == item["id"]), None)
            if not row:
                continue
            lines += [f"### {name}", "", f"Result: {mark(row['passed'])}. {row['grade_detail']}. "
                      f"{row['new_tokens']} new tokens, finish {row['finish_reason']}.", "", fence(row["output"]), ""]
    return "\n".join(lines) + "\n"


def write_side_by_side(results, directory=LOCAL_DIR):
    """Write demo_code_outputs.json and .md when every cell is present. Returns the two paths or None."""
    if not all(name in results and not results[name].get("error") for name in CELLS):
        return None
    os.makedirs(directory, exist_ok=True)
    side = {"request": "I need a coding assistant as good as Qwen3-30B at code, at half the memory.",
            "cells": {name: {k: v for k, v in results[name].items() if k != "rows"} for name in CELLS},
            "prompts": []}
    for item in PROMPTS:
        entry = {"id": item["id"], "kind": item["kind"], "title": item["title"], "prompt": item["text"]}
        for name in CELLS:
            row = next((x for x in results[name]["rows"] if x["id"] == item["id"]), None)
            entry[name] = None if row is None else {k: row[k] for k in
                                                    ("output", "new_tokens", "finish_reason", "passed", "grade_detail")}
        side["prompts"].append(entry)
    json_path = os.path.join(directory, "demo_code_outputs.json")
    md_path = os.path.join(directory, "demo_code_outputs.md")
    with open(json_path, "w") as f:
        json.dump(side, f, indent=2)
    with open(md_path, "w") as f:
        f.write(render_markdown(results))
    return json_path, md_path


@app.local_entrypoint()
def main(cells: str = "all", smoke: bool = False, collect_only: bool = False):
    if collect_only:
        results = read_results.remote()
        print("on the volume:", sorted(results))
        print("wrote", write_side_by_side(results))
        return
    names = ["full_30b"] if smoke else (list(CELLS) if cells == "all" else [c.strip() for c in cells.split(",") if c.strip()])
    for name in names:
        if name not in CELLS:
            raise SystemExit(f"unknown cell {name!r}; cells are {list(CELLS)}")
    if not names:
        raise SystemExit("no cells to run")
    print(f"cells: {names}" + (" (smoke)" if smoke else ""))
    handles = {name: run_cell.spawn(name, smoke) for name in names}
    results = {}
    for name, handle in handles.items():
        result = handle.get()
        results[result["cell"]] = result
        os.makedirs(LOCAL_DIR, exist_ok=True)
        with open(os.path.join(LOCAL_DIR, f"demo_code_{result['cell']}.json"), "w") as f:
            json.dump(result, f, indent=2)
        if result.get("error"):
            print(f"[{name}] failed: {result['error']}")
            continue
        for row in result["rows"]:
            print(f"[{result['cell']}] {row['id']}: passed={row['passed']} {row['grade_detail'][:120]}")
    if not smoke:
        print("wrote", write_side_by_side(results))
