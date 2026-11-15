"""Score one model on six skills: math, competition math, code (two sets), medicine, and general knowledge.

Any model that takes input_ids and attention_mask works: a plain Hugging
Face model, a wsp-compiled model from wsp.load, or a fused model from
wsp.fuse.load.

Skills and scoring:
  gsm8k      500 seeded GSM8K test items. Chat template, the question plus
             "Solve step by step and end with: The answer is N.", greedy,
             512 new tokens. The answer is the number after "The answer
             is", else after "####", else in the last \\boxed{}, else the
             last number (modal_graft.final_number plus the first rule).
  math500    HuggingFaceH4/MATH-500. Chat template, "put the final answer
             in \\boxed{}", greedy, 1024 new tokens. The last \\boxed{} is
             the answer; after a simple normalization, strings must be
             equal or both must be the same number or fraction.
  humaneval  openai_humaneval. The model completes the function from the
             prompt (chat template, one code block). The tests run in a
             fresh subprocess. pass@1, greedy.
  mbpp       MBPP sanitized test split (257 items), the same way.
  medmcqa    The 1000 round 10 MedMCQA validation items, by the
             log-probability of " A" to " D" after "Answer:".
  mmlu       The round 6 MMLU dev 1000 items, the same way.

The suite score is the macro average of the skill scores.

Code runs in a fresh Python subprocess per program, in a temporary
directory, with a 10 s timeout, CPU, memory, and file size limits where
the platform has them, no proxy variables in the environment, and the
socket and process functions replaced by a function that raises. The
model's if __name__ == "__main__": blocks are removed first. A program
passes only when it exits with status 0 after its last line wrote a
fresh token, so an early sys.exit(0) fails.
"""

import concurrent.futures
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import torch

SKILLS = ["gsm8k", "math500", "humaneval", "mbpp", "medmcqa", "mmlu"]

# MMLU test subjects that score one business domain. Each group gives two extra multiple-choice skills:
#   mmlu_<group>             every TEST item of the group's subjects
#   mmlu_general_ex_<group>  the general MMLU 1000 (test, shuffled with seed 0) minus those subjects
# The two never share an item, because the general set drops every item whose subject is in the group.
MMLU_GROUPS = {
    "medical": ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics"],
    "accounting": ["professional_accounting", "high_school_statistics"],
    "law": ["professional_law", "jurisprudence", "international_law"],
}
EXTRA_SKILLS = []
for _group in MMLU_GROUPS:
    EXTRA_SKILLS += ["mmlu_" + _group, "mmlu_general_ex_" + _group]
GENERATIVE = ["gsm8k", "math500", "humaneval", "mbpp"]
MAX_NEW_TOKENS = {"gsm8k": 512, "math500": 1024, "humaneval": 512, "mbpp": 512}
LETTERS = ["A", "B", "C", "D"]
GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
MATH_INSTRUCTION = "Solve step by step and put the final answer in \\boxed{}."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."
STOP_TOKENS = ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "<end_of_turn>", "<|end|>"]
NUMBER = r"-?[\d,]*\.?\d+"
OUTPUT_CHARS = 600


# ---------------------------------------------------------------- numbers


def parse_number(value):
    """Return a float from a number string such as '1,234.' or '-3', or None."""
    if value is None:
        return None
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def final_number(text):
    """Return the number after '####', else the last number in the text (as modal_graft.final_number)."""
    match = re.search(r"####\s*(" + NUMBER + ")", text)
    if match is not None:
        return parse_number(match.group(1))
    numbers = re.findall(NUMBER, text)
    if len(numbers) == 0:
        return None
    return parse_number(numbers[-1])


def gsm8k_prediction(text):
    """Return the predicted GSM8K number from a model output, or None.

    The rules go in order: the last "The answer is N", then "#### N", then
    the last \\boxed{} that holds a number, then the last number.
    """
    # The number may sit in $...$, \(...\), **...**, or \boxed{...} after "answer is".
    found = re.findall(r"answer is[:\s]*[\s$*\\(\[]*(?:\\boxed\{|\\text\{)?[\s$*\\]*(" + NUMBER + ")", text,
                       flags=re.IGNORECASE)
    if len(found) > 0:
        return parse_number(found[-1])
    match = re.search(r"####\s*(" + NUMBER + ")", text)
    if match is not None:
        return parse_number(match.group(1))
    boxed = last_boxed(text)
    if boxed is not None:
        value = to_number(normalize_answer(boxed))
        if value is not None:
            return value
    numbers = re.findall(NUMBER, text)
    if len(numbers) == 0:
        return None
    return parse_number(numbers[-1])


def same_number(a, b):
    """True when two floats are equal up to a relative 1e-6."""
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ---------------------------------------------------------------- boxed answers


def last_boxed(text):
    """Return the content of the last \\boxed{...} or \\fbox{...} in text, or None.

    Braces nest, so \\boxed{\\frac{1}{2}} gives \\frac{1}{2}. The form
    \\boxed 5 (no braces) gives the next word.
    """
    start = -1
    name = ""
    for candidate in ["\\boxed", "\\fbox"]:
        position = text.rfind(candidate)
        if position > start:
            start = position
            name = candidate
    if start < 0:
        return None
    rest = text[start + len(name):]
    if rest.startswith(" ") and not rest.lstrip().startswith("{"):
        word = rest.strip().split()
        if len(word) == 0:
            return None
        return word[0].rstrip("$.")
    rest = rest.lstrip()
    if not rest.startswith("{"):
        return None
    depth = 0
    for index, char in enumerate(rest):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return rest[1:index]
    return None


def normalize_answer(answer):
    """Return a simple normal form of a math answer string, for string equality."""
    s = answer.strip()
    # Units come after the number in \text{ cm} or \mbox{ inches}^2 (a
    # space first): keep what comes before them, as the MATH reference does.
    for unit in ["\\text{ ", "\\mbox{ "]:
        if unit in s:
            left = s.split(unit)[0].strip()
            if left != "":
                s = left
    s = s.replace("\\$", "")
    for wrapper in ["\\text", "\\textbf", "\\mathrm", "\\mbox", "\\mathbf"]:
        s = re.sub(re.escape(wrapper) + r"\{([^{}]*)\}", r"\1", s)
    for junk in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\displaystyle", "$"]:
        s = s.replace(junk, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for degree in ["^{\\circ}", "^\\circ", "\\circ", "°"]:
        s = s.replace(degree, "")
    s = s.replace("\\%", "").replace("%", "")
    # x = 5 and y=\frac{1}{2} keep only the value.
    match = re.match(r"^[a-zA-Z]\s*=\s*(.+)$", s)
    if match is not None:
        s = match.group(1)
    # x \in [-2,7] keeps only the set.
    match = re.match(r"^[a-zA-Z]\s*\\in\s+(.+)$", s)
    if match is not None:
        s = match.group(1)
    s = s.replace(" ", "")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    return s


def to_number(s):
    """Return the value of a normalized answer that is a number or a fraction, else None."""
    if re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
        s = s.replace(",", "")
    if re.match(r"^-?\d+(\.\d+)?$", s) or re.match(r"^-?\.\d+$", s):
        return float(s)
    match = re.match(r"^(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$", s)
    if match is None:
        match = re.match(r"^(-?)(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", s)
    if match is None:
        return None
    denominator = float(match.group(3))
    if denominator == 0:
        return None
    value = float(match.group(2)) / denominator
    if match.group(1) == "-":
        value = -value
    return value


def answers_match(predicted, gold):
    """True when a predicted math answer equals the gold answer after normalization."""
    if predicted is None or gold is None:
        return False
    a = normalize_answer(predicted)
    b = normalize_answer(gold)
    if a == b:
        return True
    return same_number(to_number(a), to_number(b))


# ---------------------------------------------------------------- code


def extract_code(text, name=None):
    """Return the Python code in a model output.

    The first fenced block that defines the function name (when given)
    wins, else the first fenced block that defines any function, else the
    first fenced block. An unclosed fence (a cut-off output) gives
    everything after it. Text without a fence comes back as it is.
    """
    blocks = re.findall(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) > 0:
        if name is not None:
            for block in blocks:
                if defines_function(block, name):
                    return block
        for block in blocks:
            if "def " in block:
                return block
        return blocks[0]
    if "```" in text:
        rest = text.split("```", 1)[1]
        lines = rest.split("\n")
        if len(lines) > 0 and re.match(r"^[ \t]*[A-Za-z0-9_+-]*[ \t]*$", lines[0]):
            lines = lines[1:]
        return "\n".join(lines)
    return text


def strip_main_block(code):
    """Remove each top-level if __name__ == "__main__": block from code.

    The test program runs as __main__, so such a block would run the
    model's own demo code: an input() call or a wrong example assert there
    fails a correct function, and a sys.exit(0) there ends the program
    before the tests.
    """
    lines = code.split("\n")
    out = []
    skipping = False
    for line in lines:
        if skipping:
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                continue
            skipping = False
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            skipping = True
            continue
        out.append(line)
    return "\n".join(out)


def defines_function(code, name):
    return re.search(r"^\s*def\s+" + re.escape(name) + r"\s*\(", code, flags=re.MULTILINE) is not None


def indent_body(code):
    """Indent a function body by 4 spaces when its first code line is not indented."""
    lines = code.split("\n")
    first = None
    for line in lines:
        if line.strip() != "":
            first = line
            break
    if first is None or first.startswith(" ") or first.startswith("\t"):
        return code
    out = []
    for line in lines:
        if line.strip() == "":
            out.append(line)
        else:
            out.append("    " + line)
    return "\n".join(out)


def assemble_humaneval(prompt, output, test, entry_point):
    """Return the program that tests one HumanEval completion.

    When the output defines the entry point, the program is the prompt's
    text before that def (imports and helper functions) plus the output's
    code. Otherwise the output is a function body and goes after the
    prompt. The item's test and check(entry_point) come last.
    """
    code = strip_main_block(extract_code(output, entry_point))
    if defines_function(code, entry_point):
        match = re.search(r"^def\s+" + re.escape(entry_point) + r"\s*\(", prompt, flags=re.MULTILINE)
        header = prompt
        if match is not None:
            header = prompt[:match.start()]
        program = header + "\n" + code
    else:
        program = prompt + indent_body(code)
    return program + "\n\n" + test + "\n\ncheck(" + entry_point + ")\n"


def assemble_mbpp(output, test_list, test_imports=None):
    """Return the program that tests one MBPP completion: imports, the code, then the asserts."""
    code = strip_main_block(extract_code(output))
    lines = []
    if test_imports is not None:
        for line in test_imports:
            lines.append(line)
    lines.append(code)
    lines.append("")
    for test in test_list:
        lines.append(test)
    return "\n".join(lines) + "\n"


# The child process runs this before the program. It sets its own resource
# limits and replaces the network and process functions with one that raises.
GUARD = '''
def _wsp_sandbox(cpu_seconds, memory_bytes):
    try:
        import resource
    except ImportError:
        resource = None
    if resource is not None:
        limits = [("RLIMIT_CPU", cpu_seconds), ("RLIMIT_AS", memory_bytes), ("RLIMIT_DATA", memory_bytes),
                  ("RLIMIT_FSIZE", 16 * 1024 * 1024), ("RLIMIT_CORE", 0)]
        for name, value in limits:
            if not hasattr(resource, name):
                continue
            try:
                resource.setrlimit(getattr(resource, name), (value, value))
            except (ValueError, OSError):
                pass
    import os
    import shutil
    import socket
    import subprocess

    def blocked(*args, **kwargs):
        raise OSError("blocked in the wsp code sandbox")

    for name in ["socket", "create_connection", "create_server", "getaddrinfo", "socketpair", "fromfd"]:
        if hasattr(socket, name):
            setattr(socket, name, blocked)
    for name in ["system", "popen", "kill", "killpg", "fork", "forkpty", "execv", "execve", "execvp", "execvpe",
                 "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl",
                 "spawnle", "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp", "remove", "unlink", "rmdir",
                 "removedirs", "rename", "renames", "truncate", "chmod", "chown", "setuid"]:
        if hasattr(os, name):
            setattr(os, name, blocked)
    for name in ["rmtree", "move", "chown"]:
        setattr(shutil, name, blocked)
    for name in ["Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"]:
        setattr(subprocess, name, blocked)


_wsp_sandbox({cpu}, {memory})
del _wsp_sandbox
'''


def sandbox_env(tmp):
    """A small environment for the child: no proxy variables, no Python path, home in the temp dir."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp, "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    return env


def hoist_future_imports(program):
    """Return (the from __future__ import lines, the rest of the program).

    Python needs those imports at the top of the file, before the guard.
    """
    futures = []
    rest = []
    for line in program.split("\n"):
        if re.match(r"^from\s+__future__\s+import\s", line):
            futures.append(line)
        else:
            rest.append(line)
    return futures, "\n".join(rest)


def stop_group(process):
    """Kill the program's process group: the program and anything it started."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def run_program(program, timeout=10.0, memory_mb=4096):
    """Run one Python program in a fresh subprocess. Return a dict with passed, status, and stderr.

    status is "passed", "failed", or "timeout". A program passes only when
    it exits with status 0 after its last line ran: the last line writes a
    fresh random token to a file, so a program that calls sys.exit(0) or
    os._exit(0) before its tests does not pass. stdout goes nowhere and
    stderr goes to a file in the temporary directory, so a program that
    prints without end cannot fill the memory of this process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program.py")
        err_path = os.path.join(tmp, "stderr.txt")
        done_path = os.path.join(tmp, "done.txt")
        token = uuid.uuid4().hex
        guard = GUARD.replace("{cpu}", str(int(timeout) + 1)).replace("{memory}", str(int(memory_mb) * 1024 * 1024))
        futures, body = hoist_future_imports(program)
        text = ""
        for line in futures:
            text += line + "\n"
        text += guard + "\n" + body + "\n\n"
        text += "with open(" + repr(done_path) + ", 'w') as _wsp_done:\n"
        text += "    _wsp_done.write(" + repr(token) + ")\n"
        with open(path, "w") as f:
            f.write(text)
        timed_out = False
        with open(err_path, "wb") as err_file:
            process = subprocess.Popen([sys.executable, "-I", path], cwd=tmp, env=sandbox_env(tmp),
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err_file,
                                       start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            # The child has its own session, so this stops it and anything
            # it started, also after a normal exit.
            stop_group(process)
            if timed_out:
                process.wait()
        error = read_tail(err_path, 2000)
        finished = False
        if os.path.exists(done_path):
            with open(done_path) as f:
                finished = f.read() == token
    if timed_out:
        return {"passed": False, "status": "timeout", "returncode": None, "stderr": error[-500:]}
    if process.returncode == 0 and finished:
        return {"passed": True, "status": "passed", "returncode": 0, "stderr": error[-500:]}
    if process.returncode == 0:
        error = error + "\n[wsp] the program exited before its last line ran"
    return {"passed": False, "status": "failed", "returncode": process.returncode, "stderr": error[-500:]}


def read_tail(path, count):
    """Return the last count bytes of a file as text."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - count))
        data = f.read()
    return data.decode("utf-8", errors="replace")


def run_programs(programs, timeout=10.0, workers=8):
    """Run many programs, several at a time. Return their results in order."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for program in programs:
            futures.append(pool.submit(run_program, program, timeout))
        for future in futures:
            results.append(future.result())
    return results


# ---------------------------------------------------------------- items


def gsm8k_items(count=500, seed=0):
    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(test)))
    random.Random(seed).shuffle(order)
    items = []
    for index in order[:count]:
        row = test[index]
        items.append({"id": f"gsm8k:{index}", "question": row["question"], "answer": row["answer"],
                      "gold": final_number(row["answer"])})
    return items


def math500_items():
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = []
    for index in range(len(rows)):
        row = rows[index]
        items.append({"id": row.get("unique_id") or f"math500:{index}", "problem": row["problem"],
                      "gold": row["answer"], "subject": row.get("subject"), "level": row.get("level")})
    return items


def humaneval_items():
    from datasets import load_dataset

    rows = load_dataset("openai/openai_humaneval", split="test")
    items = []
    for index in range(len(rows)):
        row = rows[index]
        items.append({"id": row["task_id"], "prompt": row["prompt"], "test": row["test"],
                      "entry_point": row["entry_point"]})
    return items


def mbpp_items():
    from datasets import load_dataset

    rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    items = []
    for index in range(len(rows)):
        row = rows[index]
        text = row.get("prompt") or row.get("text")
        items.append({"id": f"mbpp:{row['task_id']}", "text": text, "test_list": list(row["test_list"]),
                      "test_imports": list(row.get("test_imports") or [])})
    return items


def mcq_prompt(question, choices):
    """The round 6 multiple-choice prompt: question, lettered choices, then 'Answer:'."""
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def medmcqa_items(count=1000):
    """The round 10 items: MedMCQA validation, shuffled with seed 7, the first 1000."""
    from datasets import load_dataset

    med = load_dataset("openlifescienceai/medmcqa", split="validation")
    order = list(range(len(med)))
    random.Random(7).shuffle(order)
    items = []
    for index in order[:count]:
        row = med[index]
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        items.append({"id": f"medmcqa:{index}", "prompt": mcq_prompt(row["question"], choices),
                      "gold": int(row["cop"]), "subject": row.get("subject_name") or "unknown",
                      "question": row["question"], "choices": choices})
    return items


def mmlu_items(count=1000):
    """The round 6 dev items: MMLU test, shuffled with seed 0, the first 1000."""
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    items = []
    for index in order[:count]:
        row = test[index]
        items.append({"id": f"mmlu_dev:{index}", "prompt": mcq_prompt(row["question"], row["choices"]),
                      "gold": int(row["answer"]), "subject": row["subject"]})
    return items


def _mmlu_test_rows():
    """Every row of the MMLU test split, in dataset order. Tests replace this."""
    from datasets import load_dataset

    return load_dataset("cais/mmlu", "all", split="test")


def mmlu_subject_items(subjects, rows=None):
    """Every MMLU TEST item whose subject is in subjects, in dataset order. Raises when a subject has no item."""
    if rows is None:
        rows = _mmlu_test_rows()
    wanted = set(subjects)
    items = []
    seen = set()
    for index in range(len(rows)):
        row = rows[index]
        if row["subject"] not in wanted:
            continue
        seen.add(row["subject"])
        items.append({"id": f"mmlu_test:{index}", "prompt": mcq_prompt(row["question"], row["choices"]),
                      "gold": int(row["answer"]), "subject": row["subject"],
                      "question": row["question"], "choices": list(row["choices"])})
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError("no MMLU test items for subject(s): " + ", ".join(missing))
    return items


def mmlu_general_items(exclude_subjects=(), count=1000, rows=None):
    """The general MMLU set minus some subjects: the test split shuffled with seed 0 (as mmlu_items), the items
    whose subject is not excluded, the first count. With no exclusion it equals mmlu_items(count)."""
    if rows is None:
        rows = _mmlu_test_rows()
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    dropped = set(exclude_subjects)
    items = []
    for index in order:
        row = rows[index]
        if row["subject"] in dropped:
            continue
        items.append({"id": f"mmlu_dev:{index}", "prompt": mcq_prompt(row["question"], row["choices"]),
                      "gold": int(row["answer"]), "subject": row["subject"],
                      "question": row["question"], "choices": list(row["choices"])})
        if len(items) >= count:
            break
    return items


# ---------------------------------------------------------------- clinic scorers (medical set of the clinic demo)

# The six MMLU subjects the clinic demo scores as one medical group. mmlu_general_ex_med6 is the general MMLU 1000
# (test, seed 0) minus these subjects, so the two sets never share an item.
MED6_SUBJECTS = ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics",
                 "college_biology"]
CLINIC_SKILLS = ["medmcqa", "medqa", "pubmedqa", "mmlu_med6", "mmlu_general_ex_med6"]
PUBMEDQA_CHOICES = ["yes", "no", "maybe"]


def medqa_items(split="test", rows=None):
    """MedQA USMLE 4-option items (GBaker/MedQA-USMLE-4-options). The test split has 1273 items."""
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    items = []
    for index in range(len(rows)):
        row = rows[index]
        options = row["options"]
        choices = [options[letter] for letter in LETTERS] if isinstance(options, dict) else list(options)
        items.append({"id": f"medqa_{split}:{index}", "prompt": mcq_prompt(row["question"], choices),
                      "question": row["question"], "choices": choices, "gold": LETTERS.index(row["answer_idx"]),
                      "subject": "medqa"})
    return items


def pubmedqa_test_ids():
    """{pubid: label} of the 500 official PubMedQA test items (wsp/pubmedqa_test.py)."""
    from wsp import pubmedqa_test

    return dict(pubmedqa_test.TEST)


def pubmedqa_question(row):
    """The question text of a PubMedQA row: the abstract sections, then the question."""
    contexts = row["context"]["contexts"] if isinstance(row["context"], dict) else row["context"]
    return "Abstract: " + " ".join(str(c).strip() for c in contexts) + "\nQuestion: " + row["question"].strip()


def pubmedqa_row_item(row, prefix="pubmedqa"):
    """An item with choices yes, no, maybe (gold index from final_decision), or None when the row has no label."""
    label = row.get("final_decision")
    if label not in PUBMEDQA_CHOICES:
        return None
    question = pubmedqa_question(row)
    return {"id": f"{prefix}:{row['pubid']}", "prompt": mcq_prompt(question, PUBMEDQA_CHOICES), "question": question,
            "choices": list(PUBMEDQA_CHOICES), "gold": PUBMEDQA_CHOICES.index(label), "subject": "pubmedqa"}


def pubmedqa_items(rows=None, test_ids=None):
    """PubMedQA labeled TEST: the 500 labeled items of the official test set (pubids in the ground-truth file)."""
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if test_ids is None:
        test_ids = pubmedqa_test_ids()
    wanted = {str(k) for k in test_ids}
    items = []
    for index in range(len(rows)):
        row = rows[index]
        if str(row["pubid"]) not in wanted:
            continue
        item = pubmedqa_row_item(row)
        if item is not None:
            items.append(item)
    return items


def load_items(skill):
    """Return the eval items of one skill."""
    if skill == "medqa":
        return medqa_items("test")
    if skill == "pubmedqa":
        return pubmedqa_items()
    if skill == "mmlu_med6":
        return mmlu_subject_items(MED6_SUBJECTS)
    if skill == "mmlu_general_ex_med6":
        return mmlu_general_items(MED6_SUBJECTS)
    for group, subjects in MMLU_GROUPS.items():
        if skill == "mmlu_" + group:
            return mmlu_subject_items(subjects)
        if skill == "mmlu_general_ex_" + group:
            return mmlu_general_items(subjects)
    if skill == "gsm8k":
        return gsm8k_items()
    if skill == "math500":
        return math500_items()
    if skill == "humaneval":
        return humaneval_items()
    if skill == "mbpp":
        return mbpp_items()
    if skill == "medmcqa":
        return medmcqa_items()
    if skill == "mmlu":
        return mmlu_items()
    raise ValueError(f"unknown skill: {skill!r}; the skills are {', '.join(SKILLS + EXTRA_SKILLS)}")


# ---------------------------------------------------------------- prompts


def render_prompt(tokenizer, user_text):
    """Return the prompt string for one user message.

    With a chat template, the template with the generation prompt (it holds
    any BOS). Without one, BOS plus "Question: ...\\nAnswer:". Encode the
    result with add_special_tokens=False.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    bos = tokenizer.bos_token or ""
    return bos + "Question: " + user_text + "\nAnswer:"


def user_text(skill, item):
    """Return the user message for a generative item."""
    if skill == "gsm8k":
        return item["question"] + "\n" + GSM8K_INSTRUCTION
    if skill == "math500":
        return item["problem"] + "\n" + MATH_INSTRUCTION
    if skill == "humaneval":
        return ("Complete the following Python function. " + CODE_INSTRUCTION + "\n\n```python\n"
                + item["prompt"] + "```")
    if skill == "mbpp":
        return (item["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(item["test_list"])
                + "\n\n" + CODE_INSTRUCTION)
    raise ValueError(f"{skill} is not a generative skill")


# ---------------------------------------------------------------- generation


class StopOnStrings:
    """A stopping criterion: a row is done once its new text holds any stop string.

    It returns one bool per row, so generate pads the rows that are done
    and stops when every row is done.
    """

    def __init__(self, tokenizer, stop_strings, prompt_length, tail_tokens=24):
        self.tokenizer = tokenizer
        self.stop_strings = list(stop_strings)
        self.prompt_length = prompt_length
        self.tail_tokens = tail_tokens

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for row in range(input_ids.shape[0]):
            new = input_ids[row, self.prompt_length:]
            text = self.tokenizer.decode(new[-self.tail_tokens:], skip_special_tokens=False)
            hit = False
            for stop in self.stop_strings:
                if stop in text:
                    hit = True
            done.append(hit)
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def stop_token_ids(model, tokenizer):
    """Return the token ids that end a turn: the EOS ids plus the end-of-turn tokens in the vocabulary."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    config = getattr(model, "generation_config", None)
    if config is not None and config.eos_token_id is not None:
        found = config.eos_token_id
        if isinstance(found, int):
            found = [found]
        for token_id in found:
            if token_id not in ids:
                ids.append(token_id)
    vocab = tokenizer.get_vocab()
    for token in STOP_TOKENS:
        if token in vocab and vocab[token] not in ids:
            ids.append(vocab[token])
    return ids


def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def model_device(model):
    return next(model.parameters()).device


def generate_texts(model, tokenizer, prompts, max_new_tokens=512, batch_size=32, stop_strings=None):
    """Greedy generation for many prompts with left padding. Return the new texts in prompt order.

    Prompts are encoded with add_special_tokens=False (render_prompt adds
    any BOS) and batched by length. A stop string ends a row early and the
    text is cut there.
    """
    from transformers import StoppingCriteriaList

    prepare_tokenizer(tokenizer)
    device = model_device(model)
    stop_ids = stop_token_ids(model, tokenizer)
    lengths = []
    for prompt in prompts:
        lengths.append(len(prompt))
    order = sorted(range(len(prompts)), key=lambda index: -lengths[index])
    texts = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        batch = []
        for index in chunk:
            batch.append(prompts[index])
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_length = input_ids.shape[1]
        criteria = None
        if stop_strings:
            criteria = StoppingCriteriaList([StopOnStrings(tokenizer, stop_strings, prompt_length)])
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids,
                                 stopping_criteria=criteria)
        for row, index in enumerate(chunk):
            text = tokenizer.decode(out[row, prompt_length:], skip_special_tokens=True)
            if stop_strings:
                for stop in stop_strings:
                    cut = text.find(stop)
                    if cut >= 0:
                        text = text[:cut]
            texts[index] = text
    return texts


def letter_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    return ids


def score_choices(model, tokenizer, items, batch_size=16):
    """Score multiple-choice items by the log-prob of ' A' to ' D' after the prompt. Return records.

    The raw prompt goes in with the tokenizer's own special tokens, as in
    rounds 6 and 10, so the numbers compare with those rounds.
    """
    prepare_tokenizer(tokenizer)
    device = model_device(model)
    choice_ids = letter_ids(tokenizer)
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].float()
        probs = torch.softmax(logits[:, choice_ids], dim=-1).cpu()
        for row, item in enumerate(batch):
            pred = int(probs[row].argmax())
            values = []
            for value in probs[row].tolist():
                values.append(round(value, 5))
            records.append({"id": item["id"], "gold": item["gold"], "pred": pred, "probs": values,
                            "correct": pred == item["gold"]})
    return records


# ---------------------------------------------------------------- skills


def grade_outputs(skill, items, outputs, timeout=10.0, workers=8):
    """Grade the outputs of a generative skill. Return one record per item."""
    records = []
    if skill == "gsm8k" or skill == "math500":
        for item, output in zip(items, outputs):
            if skill == "gsm8k":
                pred = gsm8k_prediction(output)
                correct = same_number(pred, item["gold"])
            else:
                pred = last_boxed(output)
                correct = answers_match(pred, item["gold"])
            records.append({"id": item["id"], "gold": item["gold"], "pred": pred, "correct": correct,
                            "output": output[:OUTPUT_CHARS]})
        return records
    programs = []
    for item, output in zip(items, outputs):
        if skill == "humaneval":
            programs.append(assemble_humaneval(item["prompt"], output, item["test"], item["entry_point"]))
        else:
            programs.append(assemble_mbpp(output, item["test_list"], item["test_imports"]))
    results = run_programs(programs, timeout=timeout, workers=workers)
    for item, output, result in zip(items, outputs, results):
        records.append({"id": item["id"], "pred": result["status"], "correct": result["passed"],
                        "stderr": result["stderr"][-200:], "output": output[:OUTPUT_CHARS]})
    return records


def evaluate_skill(model, tokenizer, skill, items=None, batch_size=32, max_new_tokens=None, limit=None,
                   timeout=10.0, workers=8):
    """Score one skill. Return {"skill", "score", "correct", "n", "seconds", "records"}."""
    if skill not in SKILLS and skill not in EXTRA_SKILLS:
        raise ValueError(f"unknown skill: {skill!r}; the skills are {', '.join(SKILLS + EXTRA_SKILLS)}")
    if items is None:
        items = load_items(skill)
    if limit is not None:
        items = items[:limit]
    start = time.time()
    if skill in GENERATIVE:
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS[skill]
        prompts = []
        for item in items:
            prompts.append(render_prompt(tokenizer, user_text(skill, item)))
        outputs = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
        records = grade_outputs(skill, items, outputs, timeout=timeout, workers=workers)
    else:
        records = score_choices(model, tokenizer, items, batch_size=max(1, batch_size // 2))
    correct = 0
    for record in records:
        if record["correct"]:
            correct += 1
    score = 0.0
    if len(records) > 0:
        score = correct / len(records)
    return {"skill": skill, "score": score, "correct": correct, "n": len(records),
            "seconds": round(time.time() - start, 1), "records": records}


def evaluate_suite(model, tokenizer, skills=None, items=None, batch_size=32, limit=None, timeout=10.0, workers=8,
                   log=True):
    """Score every skill. Return {"scores": {skill: score}, "suite": macro average, "skills": {skill: result}}.

    items: None, or a dict of skill -> items, so many models score the
    same items without loading the datasets again.
    """
    if skills is None:
        skills = list(SKILLS)
    scores = {}
    results = {}
    for skill in skills:
        skill_items = None
        if items is not None and skill in items:
            skill_items = items[skill]
        result = evaluate_skill(model, tokenizer, skill, items=skill_items, batch_size=batch_size, limit=limit,
                                timeout=timeout, workers=workers)
        scores[skill] = result["score"]
        results[skill] = result
        if log:
            print(f"{skill}: {result['score']:.4f} ({result['correct']}/{result['n']}) in {result['seconds']} s")
    total = 0.0
    for skill in scores:
        total += scores[skill]
    suite = 0.0
    if len(scores) > 0:
        suite = total / len(scores)
    return {"scores": scores, "suite": suite, "skills": results}


def decode_speed(model, tokenizer, new_tokens=128, repeats=2,
                 prompt="Explain how a binary search works, step by step."):
    """Return greedy decode tokens per second for one prompt at batch 1 (best of repeats)."""
    prepare_tokenizer(tokenizer)
    device = model_device(model)
    text = render_prompt(tokenizer, prompt)
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    best = 0.0
    # One short warm-up run, then the timed runs.
    for run in range(repeats + 1):
        count = new_tokens
        if run == 0:
            count = 8
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=count,
                                 min_new_tokens=count, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - start
        made = out.shape[1] - input_ids.shape[1]
        if run > 0 and seconds > 0:
            speed = made / seconds
            if speed > best:
                best = speed
    return best


def model_params(model):
    """Return (params_total, params_run_per_token).

    Every distinct parameter counts once in the total. The input embedding
    table is a lookup, so it does not count as run per token, unless the
    output head shares it (tied weights), because the head multiplies by
    all of it.
    """
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    run = total
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            run -= embed.weight.numel()
    return total, run


def strip_outputs(result, keep_output=True):
    """Return a copy of a suite result fit for a JSON row: per-item records keep outputs only when asked."""
    out = {"scores": dict(result["scores"]), "suite": result["suite"], "skills": {}}
    for skill in result["skills"]:
        entry = result["skills"][skill]
        records = []
        for record in entry["records"]:
            row = dict(record)
            if not keep_output:
                row.pop("output", None)
                row.pop("stderr", None)
            records.append(row)
        out["skills"][skill] = {"score": entry["score"], "correct": entry["correct"], "n": entry["n"],
                                "seconds": entry["seconds"], "records": records}
    return out
