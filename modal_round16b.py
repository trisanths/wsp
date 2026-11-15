"""Round 16b: skills as experts in one MoE network (wsp/skill_moe.py).

One written network: in every layer of the base, the MLP becomes a router
over the base MLP and the same layer's MLP from each specialist. Only the
routers (and an optional LoRA on the base attention) train, in a short
heal. Nothing runs beside the network at inference.

Base: Qwen/Qwen2.5-1.5B-Instruct (28 layers, hidden 1536, MLP 8960).
Experts: Qwen/Qwen2.5-Coder-1.5B-Instruct and Qwen/Qwen2.5-Math-1.5B-Instruct,
the same shape and vocabulary (151936). Math has rope_theta 10000 and 4096
positions; base and Coder have rope_theta 1e6. MLPs do not use rope, so
Math's MLPs are valid experts, but they were trained under attention with
another rope base, so the states they read differ somewhat from the ones
they saw in training. The MoE keeps the base config (rope 1e6).

Arms (2 seeds where trained; the seed changes the heal data order, the
router noise, and the LoRA init):
  base, coder, math   each model alone, its own tokenizer and chat template
                      (Math's template has another default system prompt,
                      "Please reason step by step, and put your final
                      answer within \\boxed{}."; the GSM8K grader reads a
                      boxed answer too; rows keep a prompt sample)
  ties                TIES merge of the three: task vectors against
                      Qwen/Qwen2.5-1.5B, keep the top 20% by magnitude,
                      sign consensus, disjoint mean, scale 1
  moe_soft            top_k 3 (soft mixture of all experts)
  moe_top1            top_k 1 (Switch style)
  moe_top2            top_k 2 (Mixtral style, renormalized)
  moe_top2_lora       moe_top2 plus LoRA r16 on the base attention
  lora_only           base plus LoRA r16 on attention and MLP, same data and steps

Routers start as one-hot on the base (bias [16, 0, 0], base row zero,
expert rows small seeded noise so Coder and Math do not tie), so a fresh
MoE equals the base to about 2e-7. Routers train with Adam eps 1e-16 (see
wsp/skill_moe.py: the init gradient is scaled by about exp(-16)).

Heal: 1000 steps, batch 8, seq 512, one pass over 8000 texts in the base
chat format, equal thirds: GSM8K train solutions ending "The answer is N.";
code (MBPP full train, task ids 601 to 974, disjoint from the test split,
plus Python rows of ise-uiuc/Magicoder-OSS-Instruct-75K); general
instructions (the first turn of HuggingFaceH4/ultrachat_200k train_sft).
Router lr 1e-3, LoRA lr 1e-4, load-balance weight 0.01 (MoE arms).

Scores: GSM8K test 300 seeded items (chat template, "Solve step by step and
end with: The answer is N.", greedy, 512 new tokens), MBPP sanitized test
200 seeded items (pass@1, tests in a subprocess, 10 s timeout), MMLU dev
1000 (round 6 items) and MedMCQA 1000 (round 10 items) by the log-prob of
A to D. wsp/skills_eval.py does the scoring when it imports; else the small
copy below. Rows also hold total and active params, decode tok/s at batch
1, and routing shares per expert per skill (prompt tokens of 32 items per
skill, one item per forward, so no padding).

    python scripts/modal_fanout.py modal_round16b.py round16b --profiles a,b,c

Writes results/round16b_shard<i>.json, full per-item records to
/vol/results/round16b/<arm>_s<seed>.json, and MoE routers to
/vol/written/round16b_<arm>_s<seed>.
"""

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import modal

app = modal.App("wsp-round16b")
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

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
PRETRAINED = "Qwen/Qwen2.5-1.5B"
CODER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MATH = "Qwen/Qwen2.5-Math-1.5B-Instruct"
EXPERT_NAMES = ["base", "coder", "math"]

ALONE = {"base": BASE, "coder": CODER, "math": MATH}
MOE_TOP_K = {"moe_soft": 3, "moe_top1": 1, "moe_top2": 2, "moe_top2_lora": 2}
TRAINED = ["moe_soft", "moe_top1", "moe_top2", "moe_top2_lora", "lora_only"]
SEEDS = [0, 1]

HEAL = {"steps": 1000, "batch_size": 8, "seq_len": 512}
ROUTER_LR = 1e-3
LORA_LR = 1e-4
LORA_RANK = 16
BALANCE_WEIGHT = 0.01
TIES_KEEP = 0.2
TIES_SCALE = 1.0

SKILLS = ["gsm8k", "mbpp", "mmlu", "medmcqa"]
COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000, "medmcqa": 1000}
ROUTING_ITEMS = 32
GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."
LETTERS = ["A", "B", "C", "D"]


def cells():
    out = []
    for arm in ["base", "coder", "math", "ties"]:
        out.append((arm, 0))
    for arm in TRAINED:
        for seed in SEEDS:
            out.append((arm, seed))
    return out


# ---------------------------------------------------------------- eval (small copy)
# Used only when wsp.skills_eval does not import.


def number_of(text):
    if text is None:
        return None
    text = text.replace(",", "").rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def gsm8k_answer(text):
    """The last "The answer is N", else "#### N", else the last number."""
    number = r"-?[\d,]*\.?\d+"
    found = re.findall(r"answer is[:\s$*\\(]*(" + number + ")", text, flags=re.IGNORECASE)
    if len(found) > 0:
        return number_of(found[-1])
    match = re.search(r"####\s*(" + number + ")", text)
    if match is not None:
        return number_of(match.group(1))
    numbers = re.findall(number, text)
    if len(numbers) == 0:
        return None
    return number_of(numbers[-1])


def mcq_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def local_items(skill):
    from datasets import load_dataset

    items = []
    if skill == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        order = list(range(len(rows)))
        random.Random(0).shuffle(order)
        for index in order[:COUNTS["gsm8k"]]:
            row = rows[index]
            items.append({"id": f"gsm8k:{index}", "question": row["question"],
                          "gold": gsm8k_answer(row["answer"].split("####")[-1])})
    elif skill == "mbpp":
        rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for index in range(len(rows)):
            row = rows[index]
            items.append({"id": f"mbpp:{row['task_id']}", "text": row["prompt"], "test_list": list(row["test_list"]),
                          "test_imports": list(row["test_imports"] or [])})
    elif skill == "mmlu":
        rows = load_dataset("cais/mmlu", "all", split="test")
        order = list(range(len(rows)))
        random.Random(0).shuffle(order)
        for index in order[:COUNTS["mmlu"]]:
            row = rows[index]
            items.append({"id": f"mmlu_dev:{index}", "prompt": mcq_prompt(row["question"], row["choices"]),
                          "gold": int(row["answer"])})
    elif skill == "medmcqa":
        rows = load_dataset("openlifescienceai/medmcqa", split="validation")
        order = list(range(len(rows)))
        random.Random(7).shuffle(order)
        for index in order[:COUNTS["medmcqa"]]:
            row = rows[index]
            choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
            items.append({"id": f"medmcqa:{index}", "prompt": mcq_prompt(row["question"], choices),
                          "gold": int(row["cop"])})
    return items


def local_user_text(skill, item):
    if skill == "gsm8k":
        return item["question"] + "\n" + GSM8K_INSTRUCTION
    return (item["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(item["test_list"])
            + "\n\n" + CODE_INSTRUCTION)


def local_render(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def local_generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=32):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop = [tokenizer.eos_token_id]
    vocab = tokenizer.get_vocab()
    for token in ["<|im_end|>", "<|endoftext|>"]:
        if token in vocab and vocab[token] not in stop:
            stop.append(vocab[token])
    texts = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop)
        for row in range(len(batch)):
            texts.append(tokenizer.decode(out[row, encoded["input_ids"].shape[1]:], skip_special_tokens=True))
    return texts


def code_block(text):
    blocks = re.findall(r"```[A-Za-z0-9_+-]*\n(.*?)```", text, flags=re.DOTALL)
    for block in blocks:
        if "def " in block:
            return block
    if len(blocks) > 0:
        return blocks[0]
    if "```" in text:
        return text.split("```", 1)[1].split("\n", 1)[-1]
    return text


def run_program(program, timeout=10.0):
    """Run one program in a fresh process group; on timeout kill the whole group."""
    import signal

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program.py")
        with open(path, "w") as f:
            f.write(program)
        process = subprocess.Popen([sys.executable, path], cwd=tmp, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return False
        return code == 0


def local_score_choices(model, tokenizer, items, batch_size=16):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = []
    for letter in LETTERS:
        choice_ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            last = model(**encoded).logits[:, -1, :].float()
        for row, item in enumerate(batch):
            pred = int(last[row, choice_ids].argmax())
            records.append({"id": item["id"], "gold": item["gold"], "pred": pred, "correct": pred == item["gold"]})
    return records


def local_evaluate(model, tokenizer, items):
    scores = {}
    results = {}
    for skill in SKILLS:
        start = time.time()
        records = []
        if skill in ["gsm8k", "mbpp"]:
            prompts = []
            for item in items[skill]:
                prompts.append(local_render(tokenizer, local_user_text(skill, item)))
            outputs = local_generate(model, tokenizer, prompts)
            for item, output in zip(items[skill], outputs):
                if skill == "gsm8k":
                    pred = gsm8k_answer(output)
                    correct = pred is not None and item["gold"] is not None and abs(pred - item["gold"]) < 1e-6
                else:
                    program = "\n".join(item["test_imports"]) + "\n" + code_block(output) + "\n\n"
                    program += "\n".join(item["test_list"]) + "\n"
                    correct = run_program(program)
                    pred = None
                records.append({"id": item["id"], "pred": pred, "correct": correct, "output": output[:600]})
        else:
            records = local_score_choices(model, tokenizer, items[skill])
        correct = 0
        for record in records:
            if record["correct"]:
                correct += 1
        scores[skill] = correct / max(1, len(records))
        results[skill] = {"score": scores[skill], "correct": correct, "n": len(records),
                          "seconds": round(time.time() - start, 1), "records": records}
        print(f"{skill}: {scores[skill]:.4f} ({correct}/{len(records)})")
    return {"scores": scores, "suite": sum(scores.values()) / len(scores), "skills": results}


def local_decode_speed(model, tokenizer, new_tokens=128):
    import torch

    ids = tokenizer([local_render(tokenizer, "Explain how a binary search works, step by step.")],
                    return_tensors="pt", add_special_tokens=False).to(model.device)
    best = 0.0
    for run in range(3):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        if run > 0:
            best = max(best, (out.shape[1] - ids["input_ids"].shape[1]) / (time.time() - start))
    return best


# ---------------------------------------------------------------- eval (shared)


def skills_eval():
    """Return wsp.skills_eval when it imports cleanly, else None."""
    try:
        from wsp import skills_eval as module

        return module
    except Exception as error:
        print("wsp.skills_eval does not import, using the local copy:", repr(error)[:300])
        return None


def eval_items(ev):
    if ev is None:
        items = {}
        for skill in SKILLS:
            items[skill] = local_items(skill)
    else:
        items = {
            "gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0),
            "mbpp": ev.mbpp_items(),
            "mmlu": ev.mmlu_items(count=COUNTS["mmlu"]),
            "medmcqa": ev.medmcqa_items(count=COUNTS["medmcqa"]),
        }
    # MBPP: a seeded 200 of the 257 sanitized test items.
    order = list(range(len(items["mbpp"])))
    random.Random(0).shuffle(order)
    picked = []
    for index in sorted(order[:COUNTS["mbpp"]]):
        picked.append(items["mbpp"][index])
    items["mbpp"] = picked
    return items


def evaluate(model, tokenizer, ev, items):
    if ev is None:
        return local_evaluate(model, tokenizer, items)
    return ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items)


def decode_speed(model, tokenizer, ev):
    if ev is None:
        return local_decode_speed(model, tokenizer)
    return ev.decode_speed(model, tokenizer)


def prompt_ids(tokenizer, ev, skill, item):
    """The token ids a skill feeds the model for one item (prompt only)."""
    if skill in ["mmlu", "medmcqa"]:
        return tokenizer(item["prompt"], return_tensors="pt").input_ids
    if ev is None:
        text = local_render(tokenizer, local_user_text(skill, item))
    else:
        text = ev.render_prompt(tokenizer, ev.user_text(skill, item))
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def routing_by_skill(model, tokenizer, ev, items):
    """Routing shares per expert per skill on the prompt tokens of ROUTING_ITEMS items."""
    import torch

    from wsp.skill_moe import routing_shares, track_routing

    device = next(model.parameters()).device
    out = {}
    for skill in SKILLS:
        track_routing(model)
        for item in items[skill][:ROUTING_ITEMS]:
            with torch.no_grad():
                model(prompt_ids(tokenizer, ev, skill, item).to(device))
        shares = routing_shares(model)
        track_routing(model, on=False)
        mean = {}
        mean_prob = {}
        for index, name in enumerate(EXPERT_NAMES):
            mean[name] = round(shares["mean"][index], 4)
            mean_prob[name] = round(shares["mean_prob"][index], 4)
        per_layer = []
        for row in shares["per_layer"]:
            values = []
            for value in row:
                values.append(round(value, 4))
            per_layer.append(values)
        out[skill] = {"share": mean, "mean_prob": mean_prob, "per_layer": per_layer}
    return out


# ---------------------------------------------------------------- heal data


def chat_text(tokenizer, user, assistant):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def gsm8k_solution(answer):
    """GSM8K train answer without calculator marks, ending 'The answer is N.'."""
    text = re.sub(r"<<[^>]*>>", "", answer)
    parts = text.split("####")
    final = ""
    if len(parts) > 1:
        final = parts[1].strip().replace(",", "")
    return parts[0].strip() + "\nThe answer is " + final + "."


def math_pool(tokenizer, count):
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    texts = []
    for index in order[:count]:
        row = rows[index]
        texts.append(chat_text(tokenizer, row["question"] + "\n" + GSM8K_INSTRUCTION, gsm8k_solution(row["answer"])))
    return texts


def code_pool(tokenizer, count):
    from datasets import load_dataset

    texts = []
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    for row in mbpp:
        user = (row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"]) + "\n\n"
                + CODE_INSTRUCTION)
        texts.append(chat_text(tokenizer, user, "```python\n" + row["code"].strip() + "\n```"))
    magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    order = list(range(len(magic)))
    random.Random(0).shuffle(order)
    for index in order:
        if len(texts) >= count:
            break
        row = magic[index]
        if row["lang"] != "python":
            continue
        texts.append(chat_text(tokenizer, row["problem"], row["solution"]))
    return texts[:count]


def general_pool(tokenizer, count):
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    rows = []
    for row in stream:
        messages = row["messages"]
        if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
            rows.append((messages[0]["content"], messages[1]["content"]))
        if len(rows) >= 3 * count:
            break
    random.Random(0).shuffle(rows)
    texts = []
    for user, assistant in rows[:count]:
        texts.append(chat_text(tokenizer, user, assistant))
    return texts


def heal_texts(tokenizer):
    total = HEAL["steps"] * HEAL["batch_size"]
    per_pool = (total + 2) // 3
    texts = math_pool(tokenizer, per_pool) + code_pool(tokenizer, per_pool) + general_pool(tokenizer, per_pool)
    random.Random(0).shuffle(texts)
    return texts


# ---------------------------------------------------------------- heal


def train(model, tokenizer, texts, groups, seed, balance_weight=0.0):
    """Train the param groups on texts (one pass in a seeded order). Return a loss log."""
    import torch

    from wsp.skill_moe import load_balance_loss

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    params = []
    for group in groups:
        params += group["params"]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    losses = []
    aux_losses = []
    model.train()
    cursor = 0
    for step in range(HEAL["steps"]):
        batch = []
        while len(batch) < HEAL["batch_size"]:
            if cursor >= len(order):
                cursor = 0
            batch.append(texts[order[cursor]])
            cursor += 1
        encoded = tokenizer(batch, truncation=True, max_length=HEAL["seq_len"], padding=True, return_tensors="pt",
                            add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        aux = None
        if balance_weight > 0:
            aux = load_balance_loss(model)
        if aux is not None:
            loss = loss + balance_weight * aux.to(loss.device)
            aux_losses.append(float(aux.detach()))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(out.loss.detach()))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{HEAL['steps']} loss {losses[-1]:.4f}")
    model.eval()
    log = {"loss_first20": sum(losses[:20]) / min(20, len(losses)),
           "loss_last20": sum(losses[-20:]) / min(20, len(losses))}
    if len(aux_losses) > 0:
        log["aux_first20"] = sum(aux_losses[:20]) / min(20, len(aux_losses))
        log["aux_last20"] = sum(aux_losses[-20:]) / min(20, len(aux_losses))
    return log


# ---------------------------------------------------------------- TIES


def trim_top(tensor, keep):
    """Keep the top keep fraction of entries by magnitude; zero the rest."""
    import torch

    count = tensor.numel()
    k = int(count * keep)
    if k <= 0:
        return torch.zeros_like(tensor)
    if k >= count:
        return tensor
    threshold = torch.kthvalue(tensor.abs().reshape(-1), count - k + 1).values
    return tensor * (tensor.abs() >= threshold).to(tensor.dtype)


def ties_merge_into(target, pretrained, finetuned):
    """TIES (Yadav et al. 2023): write pretrained + merged task vectors into target, tensor by tensor."""
    import torch

    pre = dict(pretrained.named_parameters())
    others = []
    for model in finetuned:
        others.append(dict(model.named_parameters()))
    skipped = []
    with torch.no_grad():
        for name, param in target.named_parameters():
            ok = name in pre and pre[name].shape == param.shape
            for state in others:
                if name not in state or state[name].shape != param.shape:
                    ok = False
            if not ok:
                skipped.append(name)
                continue
            base = pre[name].to("cuda", torch.float32)
            trimmed = []
            for state in others:
                trimmed.append(trim_top(state[name].to("cuda", torch.float32) - base, TIES_KEEP))
            stacked = torch.stack(trimmed)
            sign = torch.sign(stacked.sum(dim=0))
            agree = (torch.sign(stacked) == sign) & (stacked != 0)
            total = (stacked * agree.to(stacked.dtype)).sum(dim=0)
            count = agree.sum(dim=0).clamp(min=1).to(stacked.dtype)
            param.copy_((base + TIES_SCALE * total / count).to(param.device, param.dtype))
    return skipped


# ---------------------------------------------------------------- cells


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)


def plain_params(model):
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return total


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=6 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from wsp.skill_moe import ROUTER_ADAM_EPS, build_skill_moe, count_params, moe_layers, router_parameters
    from wsp.skill_moe import lora_parameters, save_skill_moe

    start = time.time()
    torch.manual_seed(seed)
    ev = skills_eval()
    row = {"arm": arm, "seed": seed, "base": BASE, "experts": [CODER, MATH],
           "eval": "wsp.skills_eval" if ev is not None else "local copy"}
    tokenizer_name = BASE
    if arm in ALONE:
        tokenizer_name = ALONE[arm]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    moe = False

    if arm in ALONE:
        model = load_model(ALONE[arm]).to("cuda").eval()
    elif arm == "ties":
        model = load_model(BASE)
        pretrained = load_model(PRETRAINED)
        coder = load_model(CODER)
        math = load_model(MATH)
        row["ties_skipped"] = ties_merge_into(model, pretrained, [model, coder, math])
        del pretrained, coder, math
        model = model.to("cuda").eval()
    elif arm in MOE_TOP_K:
        moe = True
        model = load_model(BASE)
        lora_rank = None
        if arm == "moe_top2_lora":
            lora_rank = LORA_RANK
        model = build_skill_moe(model, [load_model(CODER), load_model(MATH)], top_k=MOE_TOP_K[arm],
                                load_balance=True, lora_rank=lora_rank, base_id=BASE, expert_ids=[CODER, MATH])
        model = model.to("cuda")
        groups = [{"params": router_parameters(model), "lr": ROUTER_LR, "eps": ROUTER_ADAM_EPS}]
        if lora_rank is not None:
            groups.append({"params": lora_parameters(model), "lr": LORA_LR})
        row["heal"] = train(model, tokenizer, heal_texts(tokenizer), groups, seed, BALANCE_WEIGHT)
        bias = []
        for layer in moe_layers(model):
            values = []
            for value in layer.router_bias.detach().cpu().tolist():
                values.append(round(value, 3))
            bias.append(values)
        row["router_bias"] = bias
        out_dir = f"/vol/written/round16b_{arm}_s{seed}"
        save_skill_moe(model, out_dir, tokenizer)
        volume.commit()
        row["saved"] = out_dir
    elif arm == "lora_only":
        from wsp.lora import apply_lora

        model = load_model(BASE).to("cuda")
        for param in model.parameters():
            param.requires_grad_(False)
        params = apply_lora(model, LORA_RANK)
        row["heal"] = train(model, tokenizer, heal_texts(tokenizer), [{"params": params, "lr": LORA_LR}], seed)
    else:
        raise ValueError(f"unknown arm {arm}")

    if moe:
        counts = count_params(model)
        row["params_total"] = counts["total"]
        row["params_active"] = counts["active"]
        row["params_router"] = counts["router"]
    else:
        row["params_total"] = plain_params(model)
        row["params_active"] = row["params_total"]
    if arm == "lora_only":
        row["params_lora"] = 0
        for param in model.parameters():
            if param.requires_grad:
                row["params_lora"] += param.numel()

    items = eval_items(ev)
    # The rendered GSM8K prompt, so a difference in chat templates between arms is visible.
    first = items["gsm8k"][0]
    if ev is None:
        row["gsm8k_prompt_sample"] = local_render(tokenizer, local_user_text("gsm8k", first))
    else:
        row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", first))
    if moe:
        row["routing"] = routing_by_skill(model, tokenizer, ev, items)
    result = evaluate(model, tokenizer, ev, items)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["decode_tok_s"] = round(decode_speed(model, tokenizer, ev), 1)
    row["seconds"] = round(time.time() - start, 1)

    os.makedirs("/vol/results/round16b", exist_ok=True)
    full = dict(row)
    full["skills"] = result["skills"]
    with open(f"/vol/results/round16b/{arm}_s{seed}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["arm", "seed", "scores", "macro", "params_total", "params_active",
                                                  "decode_tok_s"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round16b_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
