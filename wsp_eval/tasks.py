"""Benchmark tasks: how to load items, build the prompt, and score an output.

Every model sees exactly the same prompt for an item. Prompts are chat
messages; the backend renders them with the model's own chat template.

Task names:
  mmlupro                 all 14 MMLU-Pro categories
  mmlupro:<category>      one category, e.g. mmlupro:biology, mmlupro:computer_science
  bbeh                    all 23 BBEH tasks
  bbeh_mini               the official 460-item BBEH mini subset
  bbeh:<task>             one BBEH task, e.g. bbeh:boolean_expressions
  gsm8k
  mmlu, mmlu:<subject>
  humaneval               runs model code; needs WSP_ALLOW_CODE_EXEC=1
"""
import os
import random
import re
import subprocess
import sys
import tempfile

LETTERS = "ABCDEFGHIJ"

MMLUPRO_CATEGORIES = [
    "biology", "business", "chemistry", "computer science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
]

BBEH_TASKS = [
    "boardgame qa", "boolean expressions", "buggy tables", "causal understanding",
    "disambiguation qa", "dyck languages", "geometric shapes", "hyperbaton",
    "linguini", "movie recommendation", "multistep arithmetic", "nycc",
    "object counting", "object properties", "sarc triples", "shuffled objects",
    "spatial reasoning", "sportqa", "temporal sequence", "time arithmetic",
    "web of lies", "word sorting", "zebra puzzles",
]

# Suffix from the official BBEH evaluation setup.
BBEH_SUFFIX = (
    "\n\nThink step by step, and when you provide the final answer, please use "
    "the prefix \"The final answer is:\" without any modification, and provide "
    "the answer directly, with no formatting, no bolding, and no markup. For "
    "instance: \"The final answer is: 42\" or \"The final answer is: yes\". If "
    "the question is multiple choice with a single correct answer, the final "
    "answer must only be the letter corresponding to the correct answer. For "
    "example, \"The final answer is: (a)\""
)


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _user(text):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_letter(output, num_options):
    """Pull a multiple-choice letter out of a model output. Returns None if none found."""
    valid = LETTERS[:num_options]
    text = output.strip()

    match = re.search(r"[Aa]nswer(?: is)?\s*[:\-]?\s*\(?([A-J])\)?(?![a-zA-Z])", text)
    if match is not None and match.group(1) in valid:
        return match.group(1)

    match = re.search(r"\(([A-J])\)", text)
    if match is not None and match.group(1) in valid:
        return match.group(1)

    # Most common for direct prompts: the output starts with the letter.
    if len(text) > 0 and text[0] in valid:
        if len(text) == 1 or not text[1].isalpha():
            return text[0]

    for match in re.finditer(r"\b([A-J])\b", text):
        if match.group(1) in valid:
            return match.group(1)
    return None


def extract_number(output):
    """Final numeric answer for GSM8K-style outputs. Returns a string or None."""
    text = output.replace(",", "").replace("$", "")
    match = re.search(r"answer is[:\s]*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match is not None:
        return match.group(1)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) == 0:
        return None
    return numbers[-1]


def numbers_equal(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def bbeh_extract_answer(output):
    """Approximation of the official BBEH answer extraction."""
    text = output.strip()
    markers = ["The final answer is:", "The final answer is", "The answer is:", "The answer is", "final answer:"]
    for marker in markers:
        position = text.rfind(marker)
        if position >= 0:
            text = text[position + len(marker):]
            break
    text = text.strip()
    # Keep only the first line of the answer.
    lines = text.split("\n")
    text = lines[0].strip()
    text = text.replace("**", "").replace("$", "")
    text = text.strip().rstrip(".").strip()
    return text.lower()


def bbeh_fuzzy_match(prediction, reference):
    """Approximation of the official BBEH fuzzy matching."""
    prediction = prediction.strip().lower()
    reference = reference.strip().lower()
    if prediction == reference:
        return True
    # "(a)" versus "a"
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        if prediction[1] == reference:
            return True
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        if reference[1] == prediction:
            return True
    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    # Bracket issues
    if "[" + reference + "]" == prediction or "[" + prediction + "]" == reference:
        return True
    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task:
    name = ""
    max_new_tokens = 32

    def load_items(self):
        """Return a list of dicts with keys: id, messages, answer (plus anything score needs)."""
        raise NotImplementedError

    def score(self, output, item):
        """Return 1.0 for correct, 0.0 for wrong."""
        raise NotImplementedError


def multiple_choice_prompt(subject, question, options):
    lines = []
    lines.append("The following is a multiple choice question about " + subject + ". "
                 "Answer with only the letter of the correct option.")
    lines.append("")
    lines.append("Question: " + question.strip())
    lines.append("Options:")
    for i in range(len(options)):
        lines.append(LETTERS[i] + ". " + str(options[i]).strip())
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


class MMLUProTask(Task):
    max_new_tokens = 32

    def __init__(self, category=None):
        self.category = category
        if category is None:
            self.name = "mmlupro"
        else:
            self.name = "mmlupro:" + category.replace(" ", "_")

    def load_items(self):
        data = _load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        items = []
        for row in data:
            if self.category is not None and row["category"] != self.category:
                continue
            items.append({
                "id": "mmlupro-" + str(row["question_id"]),
                "messages": _user(multiple_choice_prompt(row["category"], row["question"], row["options"])),
                "answer": row["answer"],
                "num_options": len(row["options"]),
            })
        return items

    def score(self, output, item):
        letter = extract_letter(output, item["num_options"])
        if letter == item["answer"]:
            return 1.0
        return 0.0


class MMLUTask(Task):
    max_new_tokens = 32

    def __init__(self, subject=None):
        self.subject = subject
        if subject is None:
            self.name = "mmlu"
        else:
            self.name = "mmlu:" + subject

    def load_items(self):
        data = _load_dataset("cais/mmlu", "all", split="test")
        items = []
        index = 0
        for row in data:
            index += 1
            if self.subject is not None and row["subject"] != self.subject:
                continue
            subject_text = row["subject"].replace("_", " ")
            items.append({
                "id": "mmlu-" + str(index - 1),
                "messages": _user(multiple_choice_prompt(subject_text, row["question"], row["choices"])),
                "answer": LETTERS[row["answer"]],
                "num_options": len(row["choices"]),
            })
        return items

    def score(self, output, item):
        letter = extract_letter(output, item["num_options"])
        if letter == item["answer"]:
            return 1.0
        return 0.0


class BBEHTask(Task):
    max_new_tokens = int(os.environ.get("WSP_BBEH_MAX_TOKENS", "1024"))

    def __init__(self, subtask=None, mini=False):
        self.subtask = subtask
        self.mini = mini
        if mini:
            self.name = "bbeh_mini"
        elif subtask is None:
            self.name = "bbeh"
        else:
            self.name = "bbeh:" + subtask.replace(" ", "_")

    def load_items(self):
        # BBEH ships one split (named "train" on the hub); it is the test set.
        data = _load_dataset("BBEH/bbeh", split="train")
        items = []
        index = 0
        for row in data:
            index += 1
            if self.subtask is not None and row["task"] != self.subtask:
                continue
            if self.mini and str(row["mini"]) != "1":
                continue
            items.append({
                "id": "bbeh-" + str(index - 1),
                "messages": _user(row["input"] + BBEH_SUFFIX),
                "answer": row["target"],
                "subtask": row["task"],
            })
        return items

    def score(self, output, item):
        prediction = bbeh_extract_answer(output)
        if bbeh_fuzzy_match(prediction, item["answer"]):
            return 1.0
        return 0.0


def gsm8k_final_number(text):
    """The number after '####', else the last number in the text (the team's GSM8K rule)."""
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if match is None:
        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        if len(numbers) == 0:
            return None
        value = numbers[-1]
    else:
        value = match.group(1)
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


class GSM8KTask(Task):
    """GSM8K, scored the same way as the team's heal data and earlier runs.

    Splits (the 2026-11-14 update; identical to gsm8k_splits in wsp/data.py):
      search   200 items from GSM8K train (seed 0 shuffle of train indices)
      heal     the rest of GSM8K train (training data; never scored)
      heldout  the full GSM8K test set, run once at the end

    Prompt: the raw text "Question: <q>\nAnswer:" with no chat template, the
    same format the heal pass trains on (format_gsm8k in wsp/data.py).
    Greedy, 256 new tokens. The output is cut at the next "Question:", then
    the final number is compared with the gold number after "####".
    """
    name = "gsm8k"
    max_new_tokens = 256

    def _item(self, row, item_id):
        return {
            "id": item_id,
            "prompt": "Question: " + row["question"] + "\nAnswer:",
            "answer": gsm8k_final_number(row["answer"]),
            "question": row["question"],
            "solution": row["answer"],
        }

    def load_items(self):
        return self.split_items("heldout", 0)

    def split_items(self, split, seed):
        if split == "heldout":
            data = _load_dataset("openai/gsm8k", "main", split="test")
            items = []
            for index in range(len(data)):
                items.append(self._item(data[index], "gsm8k-test-" + str(index)))
            return items

        from wsp.data import SEARCH_SIZE
        train = _load_dataset("openai/gsm8k", "main", split="train")
        # Same order as wsp.data.gsm8k_splits(seed): shuffle the train indices.
        order = list(range(len(train)))
        random.Random(seed).shuffle(order)
        if split == "search":
            chosen = order[:SEARCH_SIZE]
        elif split == "heal":
            chosen = order[SEARCH_SIZE:]
        else:
            raise ValueError("unknown split " + split)
        items = []
        for index in chosen:
            items.append(self._item(train[index], "gsm8k-train-" + str(index)))
        return items

    def score(self, output, item):
        text = output.split("Question:")[0]
        predicted = gsm8k_final_number(text)
        gold = item["answer"]
        if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
            return 1.0
        return 0.0


def extract_code(output, prompt, entry_point):
    match = re.search(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
    if match is not None:
        code = match.group(1)
    else:
        code = output
    if ("def " + entry_point) not in code:
        code = prompt + code
    return code


class HumanEvalTask(Task):
    name = "humaneval"
    max_new_tokens = 512

    def load_items(self):
        data = _load_dataset("openai_humaneval", split="test")
        items = []
        for row in data:
            prompt = ("Complete the following Python function. Reply with the full function "
                      "in a ```python code block.\n\n" + row["prompt"])
            items.append({
                "id": row["task_id"],
                "messages": _user(prompt),
                "answer": None,
                "code_prompt": row["prompt"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            })
        return items

    def score(self, output, item):
        if os.environ.get("WSP_ALLOW_CODE_EXEC") != "1":
            raise RuntimeError("humaneval executes model-written code. Set WSP_ALLOW_CODE_EXEC=1 "
                               "only on a sandboxed machine.")
        code = extract_code(output, item["code_prompt"], item["entry_point"])
        program = code + "\n\n" + item["test"] + "\n\ncheck(" + item["entry_point"] + ")\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "program.py")
            with open(path, "w") as f:
                f.write(program)
            try:
                result = subprocess.run([sys.executable, "-I", path], cwd=tmp, capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                return 0.0
        if result.returncode == 0:
            return 1.0
        return 0.0


_registered = {}


def register_task(task):
    """Add a custom task (used by tests; teammates can add tasks the same way)."""
    _registered[task.name] = task


def get_task(name):
    if name in _registered:
        return _registered[name]
    if name == "mmlupro":
        return MMLUProTask()
    if name.startswith("mmlupro:"):
        category = name.split(":", 1)[1].replace("_", " ")
        if category not in MMLUPRO_CATEGORIES:
            raise ValueError("unknown MMLU-Pro category: " + category)
        return MMLUProTask(category)
    if name == "bbeh":
        return BBEHTask()
    if name == "bbeh_mini":
        return BBEHTask(mini=True)
    if name.startswith("bbeh:"):
        subtask = name.split(":", 1)[1].replace("_", " ")
        if subtask not in BBEH_TASKS:
            raise ValueError("unknown BBEH task: " + subtask)
        return BBEHTask(subtask)
    if name == "gsm8k":
        return GSM8KTask()
    if name == "mmlu":
        return MMLUTask()
    if name.startswith("mmlu:"):
        return MMLUTask(name.split(":", 1)[1])
    if name == "humaneval":
        return HumanEvalTask()
    if name == "ifeval":
        raise NotImplementedError("ifeval needs the instruction checkers from lm-eval; not wired in yet")
    raise ValueError("unknown task: " + name)


def task_for_training_set(trained_on):
    """Map a catalog `trained_on` string like 'MMLU-Pro biology SFT' to a task name, or None."""
    if trained_on in _registered:
        return trained_on
    text = trained_on.lower()
    if "mmlu-pro" in text or "mmlupro" in text or "mmlu pro" in text:
        for category in MMLUPRO_CATEGORIES:
            if category in text or category.replace(" ", "_") in text:
                return "mmlupro:" + category.replace(" ", "_")
        return "mmlupro"
    if "bbeh" in text or "big-bench extra hard" in text or "bigbench extra hard" in text:
        for subtask in BBEH_TASKS:
            if subtask in text or subtask.replace(" ", "_") in text:
                return "bbeh:" + subtask.replace(" ", "_")
        return "bbeh"
    if "gsm8k" in text:
        return "gsm8k"
    if "humaneval" in text:
        return "humaneval"
    if "mmlu" in text:
        return "mmlu"
    return None
