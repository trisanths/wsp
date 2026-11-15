"""Round 17: small bases plus skills. Insert specialist layers into one network.

Each written model is ONE network (wsp/insert.py): base layers 0..at-1,
a stitch into the donor width, the donor's own decoder layers, a stitch
back, then base layers at..end. The inserted section is a residual branch
with a scalar gate that starts at 0, so a fresh network equals the base.
Only the stitches, the gates, and LoRA r16 on the base and donor layers
train. Run 14 trained only the seam glue and the block did not help; here
the base learns to read the inserted layers through its own LoRA.

Bases: Qwen/Qwen2.5-0.5B-Instruct, unsloth/Llama-3.2-1B-Instruct,
LiquidAI/LFM2-1.2B (transformers 4.57 has lfm2).
Skill donors:
  math       Qwen/Qwen2.5-Math-1.5B-Instruct layers 6..21
  code       Qwen/Qwen2.5-Coder-1.5B-Instruct layers 6..21
  knowledge  Qwen/Qwen2.5-7B-Instruct layers 8..20

Arms per base (2 seeds where trained):
  base         the base alone
  lora         base + LoRA r16 heal only, same data and steps (the fine-tuning bar)
  math         base + math insert
  code         base + code insert
  knowledge    base + knowledge insert
  all3         base + all three inserts at three depths
  all3_random  all3 with random same-length donor ranges (control)

A single insert goes where the base depth matches the donor depth of its
first layer. all3 puts math at 25%, knowledge at 50%, and code at 75% of
the base depth.

Per insert, before the heal: a ridge fit starts stitch_in (an MLP stitch)
and stitch_out on offset-matched states, then stitch_in trains for 500
steps on the donor's own LM loss when base and donor share a vocabulary
(the stitch study objective), else on the downstream objective. The donor
sink goes in at each sequence's first real token.

Heal: 1500 steps, batch 8, seq 512, on a mixed set of 12000 texts, one
pass: GSM8K train solutions in chat format ending "The answer is N.";
code instructions in chat format (MBPP full train, then CodeAlpaca-20k);
general instructions (alpaca-cleaned) in chat format; MMLU auxiliary_train
pairs in the round 6 format. Stitch texts are held apart from heal texts.

Scores: GSM8K test 300 seeded items (chat template, "Solve step by step
and end with: The answer is N.", greedy, 512 new tokens, final-number
match); MBPP sanitized test, first 200 (code block run with its asserts in
a subprocess, 10 s timeout, pass@1); MMLU dev 1000 (round 6 items) and
MedMCQA 1000 (round 10 items) by the log-prob of A to D. The macro average
of the four is the suite score. wsp/skills_eval.py does the scoring when it
imports; else the small copy in this file does. Rows also hold total and
per-token params and decode tok/s at batch 1.

    python scripts/modal_fanout.py modal_round17.py round17 \\
        --profiles trisanth,trisanthsrini,ztrisanth,trisanthsrinivasan,trisanthcollege
"""

import json
import os
import random
import re
import time

import modal

app = modal.App("wsp-round17")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

BASES = {
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "lfm12": "LiquidAI/LFM2-1.2B",
}
DONORS = {
    "math": ("Qwen/Qwen2.5-Math-1.5B-Instruct", [6, 21]),
    "code": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", [6, 21]),
    "knowledge": ("Qwen/Qwen2.5-7B-Instruct", [8, 20]),
}
ALL3_DEPTH = {"math": 0.25, "knowledge": 0.50, "code": 0.75}
ARMS = ["base", "lora", "math", "code", "knowledge", "all3", "all3_random"]
SEEDS = [0, 1]

HEAL = {"steps": 1500, "batch_size": 8, "seq_len": 512}
LRS = {"stitch": 3e-4, "gate": 1e-2, "part_lora": 1e-4, "base_lora": 1e-4}
LORA_RANK = 16
STITCH = {"kind": "mlp", "steps": 500, "lr": 3e-4, "batch_size": 8, "seq_len": 256}
HEAL_PER_POOL = 3000
STITCH_DOMAIN = 1000
STITCH_GENERAL = 300
STITCH_EVAL = 64

EVAL_SKILLS = ["gsm8k", "mbpp", "mmlu", "medmcqa"]
GSM8K_COUNT = 300
MBPP_COUNT = 200
GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."
LETTERS = ["A", "B", "C", "D"]


def cells():
    out = []
    for base_key in BASES:
        for arm in ARMS:
            seeds = SEEDS
            if arm == "base":
                seeds = [0]
            for seed in seeds:
                out.append((base_key, arm, seed))
    return out


# ---------------------------------------------------------------- data


def pin_chat_date(tokenizer):
    """Make the chat template ignore today's date. Return True when it changed.

    The Llama 3.2 template puts strftime_now() in the system turn, so the
    heal texts and the scoring prompts would change with the day a cell
    runs. With the call gone, the template uses its fixed fallback date,
    and every arm and rerun sees the same prompts.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or "strftime_now is defined" not in template:
        return False
    tokenizer.chat_template = template.replace("strftime_now is defined", "false")
    return True


def chat_text(tokenizer, user, assistant):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def with_bos(tokenizer, text):
    """Put the BOS string in front of a raw text when the tokenizer adds one by itself.

    The heal encodes with add_special_tokens=False, since chat texts hold
    their BOS already. Raw MCQ texts need it to match how they are scored.
    """
    if tokenizer.bos_token is None:
        return text
    ids = tokenizer("x").input_ids
    if len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
        return tokenizer.bos_token + text
    return text


def gsm8k_solution(answer):
    """GSM8K train answer text without calculator marks, ending 'The answer is N.'."""
    text = re.sub(r"<<[^>]*>>", "", answer)
    parts = text.split("####")
    body = parts[0].strip()
    final = ""
    if len(parts) > 1:
        final = parts[1].strip().replace(",", "")
    return body + "\nThe answer is " + final + "."


def math_pairs(count):
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    pairs = []
    for index in order[:count]:
        row = rows[index]
        pairs.append((row["question"] + "\n" + GSM8K_INSTRUCTION, gsm8k_solution(row["answer"])))
    return pairs


def code_pairs(count):
    """MBPP full train split (disjoint from the sanitized test), then CodeAlpaca-20k."""
    from datasets import load_dataset

    pairs = []
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    for index in range(len(mbpp)):
        row = mbpp[index]
        user = (row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"]) + "\n\n"
                + CODE_INSTRUCTION)
        pairs.append((user, "```python\n" + row["code"].strip() + "\n```"))
    alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    order = list(range(len(alpaca)))
    random.Random(1).shuffle(order)
    for index in order:
        if len(pairs) >= count:
            break
        row = alpaca[index]
        user = row["instruction"]
        if row.get("input"):
            user = user + "\n\n" + row["input"]
        pairs.append((user, row["output"]))
    random.Random(2).shuffle(pairs)
    return pairs[:count]


def general_pairs(count):
    from datasets import load_dataset

    rows = load_dataset("yahma/alpaca-cleaned", split="train")
    order = list(range(len(rows)))
    random.Random(4).shuffle(order)
    pairs = []
    for index in order[:count]:
        row = rows[index]
        user = row["instruction"]
        if row.get("input"):
            user = user + "\n\n" + row["input"]
        pairs.append((user, row["output"]))
    return pairs


def mcq_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def mmlu_pairs(count):
    """MMLU auxiliary_train pairs in the round 6 format: (prompt, " X")."""
    from datasets import load_dataset

    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    order = list(range(len(aux)))
    random.Random(1).shuffle(order)
    pairs = []
    for index in order:
        if len(pairs) >= count:
            break
        row = aux[index]
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mcq_prompt(row["question"], row["choices"])
        if len(prompt) > 1500:
            continue
        pairs.append((prompt, " " + LETTERS[int(row["answer"])]))
    return pairs


def build_data(tokenizer):
    """Return (heal_texts, stitch_texts, sizes).

    stitch_texts maps math, code, knowledge to plain texts (no chat
    template, so a donor with another tokenizer reads them too). The data
    seeds are fixed, so every arm and seed sees the same texts, and heal
    and stitch texts do not overlap.
    """
    total = HEAL_PER_POOL + STITCH_DOMAIN
    pools = {"math": math_pairs(total), "code": code_pairs(total), "mmlu": mmlu_pairs(total),
             "general": general_pairs(HEAL_PER_POOL + 3 * STITCH_GENERAL)}
    heal = []
    for name in ["math", "code", "general"]:
        for user, assistant in pools[name][:HEAL_PER_POOL]:
            heal.append(chat_text(tokenizer, user, assistant))
    for prompt, answer in pools["mmlu"][:HEAL_PER_POOL]:
        heal.append(with_bos(tokenizer, prompt + answer))
    random.Random(5).shuffle(heal)

    stitch = {}
    spare = pools["general"][HEAL_PER_POOL:]
    sources = {"math": "math", "code": "code", "knowledge": "mmlu"}
    for number, skill in enumerate(["math", "code", "knowledge"]):
        texts = []
        for user, assistant in pools[sources[skill]][HEAL_PER_POOL:total]:
            if skill == "knowledge":
                texts.append(user + assistant)
            else:
                texts.append(user + "\n" + assistant)
        for user, assistant in spare[number * STITCH_GENERAL:(number + 1) * STITCH_GENERAL]:
            texts.append(user + "\n" + assistant)
        random.Random(6 + number).shuffle(texts)
        stitch[skill] = texts
    sizes = {}
    for name in pools:
        sizes[name] = len(pools[name])
    sizes["heal"] = len(heal)
    return heal, stitch, sizes


# ---------------------------------------------------------------- plan


def layer_count(model_id):
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return config.num_hidden_layers


def matched_at(first, donor_layers, base_layers):
    """The base insertion point at the donor depth of the first inserted layer, in [1, base_layers - 1]."""
    at = int(round(first / donor_layers * base_layers))
    if at < 1:
        at = 1
    if at > base_layers - 1:
        at = base_layers - 1
    return at


def insert_plan(arm, seed, base_layers):
    """Return a list of dicts: skill, donor, first, last, at."""
    from wsp.assemble import random_block_range

    skills = [arm]
    if arm in ("all3", "all3_random"):
        skills = ["math", "knowledge", "code"]
    plan = []
    for number, skill in enumerate(skills):
        donor, layers = DONORS[skill]
        first = layers[0]
        last = layers[1]
        donor_layers = layer_count(donor)
        if arm in ("all3", "all3_random"):
            at = int(round(ALL3_DEPTH[skill] * base_layers))
        else:
            at = matched_at(first, donor_layers, base_layers)
        if arm == "all3_random":
            first, last = random_block_range(donor_layers, first, last, seed + 100 + number)
        plan.append({"skill": skill, "donor": donor, "first": int(first), "last": int(last), "at": int(at)})
    return plan


def same_vocab(tokenizer_a, tokenizer_b):
    return tokenizer_a.get_vocab() == tokenizer_b.get_vocab()


def slim_stitch(result):
    out = {}
    for key in result:
        if key == "losses":
            continue
        out[key] = result[key]
    losses = result.get("losses") or []
    if len(losses) > 0:
        out["loss_first10"] = sum(losses[:10]) / len(losses[:10])
        out["loss_last20"] = sum(losses[-20:]) / len(losses[-20:])
    return out


def build_inserts(base, tokenizer, plan, stitch_texts, seed):
    """Pre-train each stitch on the plain base, build each insertion, then attach them all.

    Returns a list of per-insert reports. Each donor is loaded once, used
    for the sink, the ridge fit, and the stitch training, then freed.
    """
    import torch
    import transformers

    from wsp.insert import attach_insertions, build_insertion, ridge_stitch
    from wsp.stitch import donor_sink, train_stitch

    insertions = []
    reports = []
    for entry in plan:
        start = time.time()
        donor = transformers.AutoModelForCausalLM.from_pretrained(entry["donor"], dtype=torch.bfloat16)
        donor = donor.to("cuda").eval()
        donor_tokenizer = transformers.AutoTokenizer.from_pretrained(entry["donor"])
        texts = stitch_texts[entry["skill"]]
        train_texts = texts[:-STITCH_EVAL]
        eval_texts = texts[-STITCH_EVAL:]
        with torch.no_grad():
            sink = donor_sink(donor, donor_tokenizer, entry["first"], train_texts)
        stitch, w_out, ridge = ridge_stitch(base, tokenizer, entry["at"], donor, donor_tokenizer, entry["first"],
                                            train_texts[:600], kind=STITCH["kind"])
        stitch = stitch.to("cuda")
        objective = "downstream"
        if same_vocab(tokenizer, donor_tokenizer):
            objective = "donor_lm"
        torch.manual_seed(seed)
        trained = train_stitch(base, tokenizer, entry["at"] - 1, donor, donor_tokenizer, entry["first"], stitch,
                               train_texts, objective=objective, steps=STITCH["steps"], lr=STITCH["lr"],
                               batch_size=STITCH["batch_size"], seq_len=STITCH["seq_len"], seed=seed,
                               eval_texts=eval_texts, sink=sink, log_every=100)
        insertion = build_insertion(base, donor, [entry["first"], entry["last"]], entry["at"],
                                    donor_id=entry["donor"], stitch_in=stitch, stitch_out=w_out,
                                    part_lora_rank=LORA_RANK, sink=sink)
        insertions.append(insertion)
        report = dict(entry)
        report["objective"] = objective
        report["ridge"] = ridge
        report["stitch"] = slim_stitch(trained)
        report["seconds"] = round(time.time() - start, 1)
        reports.append(report)
        print(json.dumps({"insert": entry, "objective": objective, "ridge_r2_in": ridge["r2_in"],
                          "stitch_after": report["stitch"].get("after")}, default=str))
        del donor
        torch.cuda.empty_cache()
    attach_insertions(base, insertions)
    return reports


# ---------------------------------------------------------------- scoring (fallback copy)


def skills_eval_module():
    """Return wsp.skills_eval when it imports and has what we need, else None."""
    try:
        from wsp import skills_eval
    except Exception as error:
        print(f"wsp.skills_eval does not import ({type(error).__name__}: {error}); using the local copy")
        return None
    for name in ["evaluate_skill", "gsm8k_items", "mbpp_items", "mmlu_items", "medmcqa_items", "decode_speed"]:
        if not hasattr(skills_eval, name):
            print(f"wsp.skills_eval has no {name}; using the local copy")
            return None
    return skills_eval


def final_number(text):
    """The number after 'The answer is', else after '####', else the last number."""
    number = r"-?[\d,]*\.?\d+"
    found = re.findall(r"answer is[:\s]*(?:\\boxed\{)?[\s$*\\(]*(" + number + ")", text, flags=re.IGNORECASE)
    if len(found) > 0:
        value = found[-1]
    else:
        match = re.search(r"####\s*(" + number + ")", text)
        if match is not None:
            value = match.group(1)
        else:
            numbers = re.findall(number, text)
            if len(numbers) == 0:
                return None
            value = numbers[-1]
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def local_items(skill):
    from datasets import load_dataset

    items = []
    if skill == "gsm8k":
        test = load_dataset("openai/gsm8k", "main", split="test")
        order = list(range(len(test)))
        random.Random(0).shuffle(order)
        for index in order[:GSM8K_COUNT]:
            row = test[index]
            items.append({"id": f"gsm8k:{index}", "question": row["question"], "gold": final_number(row["answer"])})
    elif skill == "mbpp":
        rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for index in range(min(MBPP_COUNT, len(rows))):
            row = rows[index]
            items.append({"id": f"mbpp:{row['task_id']}", "text": row.get("prompt") or row.get("text"),
                          "test_list": list(row["test_list"]), "test_imports": list(row.get("test_imports") or [])})
    elif skill == "mmlu":
        test = load_dataset("cais/mmlu", "all", split="test")
        order = list(range(len(test)))
        random.Random(0).shuffle(order)
        for index in order[:1000]:
            row = test[index]
            items.append({"id": f"mmlu_dev:{index}", "prompt": mcq_prompt(row["question"], row["choices"]),
                          "gold": int(row["answer"])})
    else:
        med = load_dataset("openlifescienceai/medmcqa", split="validation")
        order = list(range(len(med)))
        random.Random(7).shuffle(order)
        for index in order[:1000]:
            row = med[index]
            choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
            items.append({"id": f"medmcqa:{index}", "prompt": mcq_prompt(row["question"], choices),
                          "gold": int(row["cop"])})
    return items


def render_prompt(tokenizer, text):
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (tokenizer.bos_token or "") + "Question: " + text + "\nAnswer:"


def local_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=32):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = [tokenizer.eos_token_id]
    vocab = tokenizer.get_vocab()
    for token in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>"]:
        if token in vocab and vocab[token] not in stop_ids:
            stop_ids.append(vocab[token])
    device = model.get_input_embeddings().weight.device
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
        for row in range(len(batch)):
            outputs.append(tokenizer.decode(out[row, input_ids.shape[1]:], skip_special_tokens=True))
    return outputs


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if len(blocks) > 0:
        return blocks[0]
    return text


def limit_child(seconds, memory_mb=4096):
    """Return a preexec function that caps the child's CPU time, memory, and file size."""

    def apply():
        import resource

        limits = [(resource.RLIMIT_CPU, seconds), (resource.RLIMIT_AS, memory_mb * 1024 * 1024),
                  (resource.RLIMIT_FSIZE, 16 * 1024 * 1024), (resource.RLIMIT_CORE, 0)]
        for name, value in limits:
            try:
                resource.setrlimit(name, (value, value))
            except (ValueError, OSError):
                pass

    return apply


def run_tests(program, timeout=10.0):
    """Run one program in a fresh subprocess. Return True when it exits with status 0 in time.

    The child gets its own session, so on a timeout we kill it and every
    process it started. Its output goes to /dev/null, not to pipes, so a
    grandchild that holds a pipe open cannot block us, and a program that
    prints without end cannot fill our memory. The last line writes a
    token to a file, so a program that exits early with status 0 fails.
    """
    import signal
    import subprocess
    import sys
    import tempfile
    import uuid

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "prog.py")
        done_path = os.path.join(folder, "done.txt")
        token = uuid.uuid4().hex
        with open(path, "w") as f:
            f.write(program)
            f.write("\n\nwith open(" + repr(done_path) + ", 'w') as _done:\n")
            f.write("    _done.write(" + repr(token) + ")\n")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": folder, "TMPDIR": folder}
        process = subprocess.Popen([sys.executable, "-I", path], cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                                   preexec_fn=limit_child(int(timeout) + 1))
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        if timed_out:
            process.wait()
            return False
        finished = False
        if os.path.exists(done_path):
            with open(done_path) as f:
                finished = f.read() == token
        return process.returncode == 0 and finished


def local_mcq(model, tokenizer, items, batch_size=16):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = []
    for letter in LETTERS:
        choice_ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    device = model.get_input_embeddings().weight.device
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(input_ids=encoded["input_ids"].to(device),
                           attention_mask=encoded["attention_mask"].to(device)).logits[:, -1, :].float()
        for row in range(len(batch)):
            pred = int(logits[row, choice_ids].argmax())
            records.append({"id": batch[row]["id"], "pred": pred, "correct": pred == batch[row]["gold"]})
    return records


def local_evaluate(model, tokenizer, skill, items):
    start = time.time()
    records = []
    if skill == "gsm8k":
        prompts = []
        for item in items:
            prompts.append(render_prompt(tokenizer, item["question"] + "\n" + GSM8K_INSTRUCTION))
        outputs = local_generate(model, tokenizer, prompts)
        for item, output in zip(items, outputs):
            pred = final_number(output)
            correct = pred is not None and item["gold"] is not None and abs(pred - item["gold"]) < 1e-6
            records.append({"id": item["id"], "pred": pred, "correct": correct})
    elif skill == "mbpp":
        prompts = []
        for item in items:
            text = (item["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(item["test_list"])
                    + "\n\n" + CODE_INSTRUCTION)
            prompts.append(render_prompt(tokenizer, text))
        outputs = local_generate(model, tokenizer, prompts)
        for item, output in zip(items, outputs):
            program = "\n".join(item["test_imports"]) + "\n" + extract_code(output) + "\n\n" + "\n".join(
                item["test_list"]) + "\n"
            records.append({"id": item["id"], "correct": run_tests(program)})
    else:
        records = local_mcq(model, tokenizer, items)
    correct = 0
    for record in records:
        if record["correct"]:
            correct += 1
    return {"skill": skill, "score": correct / max(1, len(records)), "correct": correct, "n": len(records),
            "seconds": round(time.time() - start, 1), "records": records}


def local_decode_speed(model, tokenizer, new_tokens=128):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer([render_prompt(tokenizer, "Explain how a binary search works, step by step.")],
                        return_tensors="pt", add_special_tokens=False)
    best = 0.0
    for run in range(3):
        count = new_tokens
        if run == 0:
            count = 8
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=encoded["input_ids"].to(device),
                                 attention_mask=encoded["attention_mask"].to(device), max_new_tokens=count,
                                 min_new_tokens=count, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        seconds = time.time() - start
        if run > 0 and seconds > 0:
            speed = (out.shape[1] - encoded["input_ids"].shape[1]) / seconds
            if speed > best:
                best = speed
    return best


def eval_items(evaluator):
    items = {}
    for skill in EVAL_SKILLS:
        if evaluator is None:
            items[skill] = local_items(skill)
        elif skill == "gsm8k":
            items[skill] = evaluator.gsm8k_items(count=GSM8K_COUNT, seed=0)
        elif skill == "mbpp":
            items[skill] = evaluator.mbpp_items()[:MBPP_COUNT]
        elif skill == "mmlu":
            items[skill] = evaluator.mmlu_items(1000)
        else:
            items[skill] = evaluator.medmcqa_items(1000)
    return items


def score_model(model, tokenizer, items, evaluator, row):
    """Score the four skills, fill row with scores and costs, and return the full results."""
    from wsp.insert import param_counts

    counts = param_counts(model)
    row["params"] = counts
    row["params_total_b"] = counts["total"] / 1e9
    row["params_per_token_b"] = counts["per_token"] / 1e9
    if evaluator is None:
        row["decode_tok_s"] = round(local_decode_speed(model, tokenizer), 1)
    else:
        row["decode_tok_s"] = round(evaluator.decode_speed(model, tokenizer), 1)
    scores = {}
    full = {}
    for skill in EVAL_SKILLS:
        if evaluator is None:
            result = local_evaluate(model, tokenizer, skill, items[skill])
        else:
            result = evaluator.evaluate_skill(model, tokenizer, skill, items=items[skill], batch_size=32)
        scores[skill] = result["score"]
        full[skill] = result
        print(f"{row['base']} {row['arm']} s{row['seed']} {skill}: {result['score']:.4f} in {result['seconds']} s")
    row["scores"] = scores
    total = 0.0
    for skill in scores:
        total += scores[skill]
    row["macro"] = total / len(scores)
    row["evaluator"] = "local copy"
    if evaluator is not None:
        row["evaluator"] = "wsp.skills_eval"
    return full


# ---------------------------------------------------------------- cells


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=10 * 3600)
def run_cell(base_key, arm, seed):
    import torch
    import transformers

    from wsp.insert import add_base_lora, gate_values, heal, save_inserted

    start = time.time()
    torch.manual_seed(seed)
    base_id = BASES[base_key]
    row = {"base": base_key, "base_id": base_id, "arm": arm, "seed": seed}
    evaluator = skills_eval_module()
    items = eval_items(evaluator)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_id)
    # Every arm, the base arm too, gets the same date-free template.
    row["chat_date_pinned"] = pin_chat_date(tokenizer)
    model = transformers.AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda").eval()

    if arm != "base":
        heal_texts, stitch_texts, sizes = build_data(tokenizer)
        row["data"] = sizes
        lrs = {"base_lora": LRS["base_lora"]}
        if arm != "lora":
            plan = insert_plan(arm, seed, layer_count(base_id))
            row["plan"] = plan
            row["inserts"] = build_inserts(model, tokenizer, plan, stitch_texts, seed)
            lrs = dict(LRS)
        # LoRALinear puts its factors on the device of the weight it wraps.
        add_base_lora(model, LORA_RANK)
        heal_start = time.time()
        torch.manual_seed(seed)
        losses = heal(model, tokenizer, heal_texts, lrs, steps=HEAL["steps"], batch_size=HEAL["batch_size"],
                      seq_len=HEAL["seq_len"], seed=seed, log_every=100, add_special_tokens=False)
        row["heal_seconds"] = round(time.time() - heal_start, 1)
        row["heal_loss_first20"] = sum(losses[:20]) / len(losses[:20])
        row["heal_loss_last50"] = sum(losses[-50:]) / len(losses[-50:])
        row["gates"] = gate_values(model)
        out_dir = f"/vol/compiled/round17/{base_key}_{arm}_s{seed}"
        try:
            save_inserted(model, tokenizer, out_dir, base_id, extra={"row": {"arm": arm, "seed": seed}})
            row["saved"] = out_dir
        except Exception as error:
            row["save_error"] = repr(error)[:400]
        volume.commit()
        model.eval()

    full = score_model(model, tokenizer, items, evaluator, row)
    row["seconds"] = round(time.time() - start, 1)
    os.makedirs("/vol/results/round17", exist_ok=True)
    with open(f"/vol/results/round17/{base_key}_{arm}_s{seed}.json", "w") as f:
        json.dump({"row": row, "full": full}, f, indent=2, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {len(mine)} cells: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        summary = {}
        for key in ["base", "arm", "seed", "macro", "scores", "params_total_b", "params_per_token_b", "decode_tok_s",
                    "gates", "seconds"]:
            summary[key] = row.get(key)
        print(json.dumps(summary))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round17_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
