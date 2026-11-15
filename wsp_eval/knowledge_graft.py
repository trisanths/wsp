"""Knowledge graft: put the donor layers that hold MMLU answers inside a small host.

Why this test: on GSM8K a plain LoRA already reaches the donor (0.455 vs
0.45), so a graft has no headroom. On MMLU fine-tuning adds about 3 points,
while round 7's tap showed Qwen2.5-7B's layer-20 states carry about 20 more
points of answers, and that nothing readable exists before layer 16. So graft
the layers where the answer forms, not a whole donor run beside the host.

Host Qwen/Qwen2.5-0.5B-Instruct (24 layers, MMLU 48.0 in round 8). Donor
Qwen/Qwen2.5-7B-Instruct (28 layers, 71.5). Same tokenizer, so the input
stitch trains on the donor's own next-token loss (donor_lm), which cut the
stitch penalty to under 0.1 nats in the stitch study.

Arms (seed 0), MMLU dev items (round 6's 1000, log-prob of " A".." D"):
  host         untrained
  graft        donor layers 12..20 at host layer 10, linear donor_lm stitch
               (1000 steps on 1200 MMLU-aux texts), rank-64 w_out; then the
               glue trains 1000 steps on 8000 mmlu_aux pairs (answer-letter CE)
  early_block  control: donor layers 2..10 (no overlap, same length), the same
               stitch, glue and training
  lora         control: host LoRA with the graft's trainable count, the same
               pairs and steps (lr 1e-4, the round 6 LoRA setting)
  trunk        donor layers 0..20 at host layer 1: the donor's own path from its
               input to the answer layer, as one graft (same stitch, glue, training)
  trunk_shuffled  control: the trunk with every donor weight tensor scrambled, then
               the same glue training. Same compute, no donor knowledge.
  trunk_full   the trunk with a full-width w_out (3584 -> 896) instead of rank 64

  trunk_at10   the trunk read and written at host layer 10 (where the tap writes)

Run 2: trunk 0.557 (+7.7 over host, +7.9 over the matched LoRA); stitch penalty
0.30 -> -0.26 nats at donor layer 0.
Run 3: trunk_shuffled 0.246 (training diverged on the scrambled outputs); trunk_full
0.561 with 6.4M trainable, so the rank-64 exit is not the bottleneck.
Run 4: trunk_at10 0.487 (stitch penalty 0.06, training loss as low as the trunk's),
so writing at host layer 10 is worse than layer 1, not better.

Run 1 (seed 0): host 0.480, graft 0.462, early_block 0.491, lora 0.478. The
stitch into donor layer 12 only fell from 0.84 to 0.57 nats, against -0.03
for donor layer 2: a 0.5B host cannot produce the 7B's layer-12 input, which
already holds 11 layers of the donor's own work. Hence the trunk arm.
Reference from earlier rounds: Qwen-0.5B + Qwen-7B tap 69.3, donor 71.5.
Dev items are a development check, not a held-out result.
"""
import os
import random

HOST = "Qwen/Qwen2.5-0.5B-Instruct"
DONOR = "Qwen/Qwen2.5-7B-Instruct"
BLOCKS = {
    "graft": {"id": "q7b_block_12_20", "type": "block", "model": DONOR, "layers": [12, 20]},
    "early_block": {"id": "q7b_block_2_10", "type": "block", "model": DONOR, "layers": [2, 10]},
    "trunk": {"id": "q7b_block_0_20", "type": "block", "model": DONOR, "layers": [0, 20]},
}
AT = 10
# The trunk starts at donor layer 0, so it reads the host right after its first layer.
AT_BY_ARM = {"trunk": 1, "trunk_full": 1, "trunk_at10": 10}
# trunk_at10: the same trunk read and written at host layer 10, where the round 8 tap writes.
BLOCKS["trunk_at10"] = BLOCKS["trunk"]
# trunk_full: the same trunk with a full-width w_out (glue_rank None) instead of rank 64.
BLOCKS["trunk_full"] = BLOCKS["trunk"]
GLUE_RANK_BY_ARM = {"trunk_full": None}
# trunk_shuffled reuses the compiled trunk folder, then scrambles the donor weights.
FOLDER_BY_ARM = {"trunk_shuffled": "trunk"}
GLUE_RANK = 64
STITCH = {"kind": "linear", "objective": "donor_lm", "steps": 1000, "lr": 1e-3, "batch_size": 8,
          "seq_len": 256, "seed": 0}
STEPS = 1000
BATCH = 8
GLUE_LR = 3e-4
LORA_LR = 1e-4
OUT_ROOT = "/vol/compiled/knowledge_graft"
LETTERS = ["A", "B", "C", "D"]


def mmlu_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def dev_items(seed=0, count=1000):
    """Round 6's dev items: the first 1000 of MMLU test in a seeded order."""
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(seed).shuffle(order)
    items = []
    for index in order[:count]:
        row = test[index]
        items.append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))
    return items


def train_pairs_and_align_texts(tokenizer):
    """8000 mmlu_aux training pairs, and 1200 further aux texts for the stitch."""
    from wsp.pairs import pairs_for

    pairs = pairs_for("mmlu_aux", tokenizer, count=9200)
    align = []
    for prompt, completion in pairs[8000:]:
        align.append(prompt + completion)
    return pairs[:8000], align


def letter_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    return ids


def score(model, tokenizer, items, batch_size=32):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = letter_ids(tokenizer)
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for prompt, answer in batch:
            prompts.append(prompt)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**encoded, use_cache=False).logits[:, -1, :].float()
        for row in range(len(batch)):
            if int(logits[row, choice_ids].argmax()) == batch[row][1]:
                correct += 1
    return correct / len(items)


def train_answers(model, tokenizer, pairs, params, lr, seed=0):
    """Cross-entropy on the answer letter at the last prompt position. Return the losses."""
    import torch
    import torch.nn.functional as F

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for param in model.parameters():
        param.requires_grad_(False)
    for param in params:
        param.data = param.data.float()
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    targets = {}
    for index, letter in enumerate(LETTERS):
        targets[" " + letter] = letter_ids(tokenizer)[index]
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = []
    losses = []
    model.train()
    for step in range(STEPS):
        batch = []
        while len(batch) < BATCH:
            if len(order) == 0:
                order = torch.randperm(len(pairs), generator=generator).tolist()
            batch.append(pairs[order.pop()])
        prompts = []
        labels = []
        for prompt, completion in batch:
            prompts.append(prompt)
            labels.append(targets[completion])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**encoded, use_cache=False).logits[:, -1, :]
        loss = F.cross_entropy(logits.float(), torch.tensor(labels, device="cuda"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print("[knowledge_graft] step", step + 1, "loss", round(losses[-1], 4), flush=True)
    model.eval()
    return losses


def compile_block(arm, align_texts):
    """Compile the block graft with its trained stitch and no heal. Return the folder."""
    from wsp import Catalog, Program

    out_dir = os.path.join(OUT_ROOT, arm + "_s0")
    if os.path.exists(os.path.join(out_dir, "compile_report.json")):
        return out_dir
    record = BLOCKS[arm]
    at = AT_BY_ARM.get(arm, AT)
    glue_rank = GLUE_RANK_BY_ARM.get(arm, GLUE_RANK)
    program = Program(host=HOST, seed=0).graft(record["id"], at=at, glue_rank=glue_rank, stitch=dict(STITCH))
    program.compile(out_dir, Catalog([record], root="/"), align_texts=align_texts, device="cuda",
                    cache_dir="/vol/tensor_cache")
    return out_dir


def compile_with_retry(arm, align_texts, attempts=5):
    """compile_block, retried when the Hugging Face range fetch drops the connection.

    Fetched tensors are cached in /vol/tensor_cache, so a retry resumes where it stopped.
    """
    import time

    import requests

    for attempt in range(attempts):
        try:
            return compile_block(arm, align_texts)
        except requests.exceptions.ConnectionError as error:
            if attempt == attempts - 1:
                raise
            print("[knowledge_graft]", arm, "fetch failed, retrying:", repr(error)[:200], flush=True)
            time.sleep(30 * (attempt + 1))


def shuffle_part_weights(model, seed=0):
    """Scramble every donor weight tensor inside the grafts (a random permutation of its values).

    Each tensor keeps its shape, dtype and value distribution, so the graft runs the same
    amount of compute at the same scale, but whatever the donor learned is destroyed.
    The glue (w_in, w_out) is left as it is.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    count = 0
    with torch.no_grad():
        for graft in model.wsp_grafts:
            for param in graft.part.parameters():
                flat = param.data.reshape(-1)
                order = torch.randperm(flat.numel(), generator=generator).to(flat.device)
                param.data.copy_(flat[order].reshape(param.shape))
                count += param.numel()
    return count


def glue_params(model):
    params = []
    for graft in model.wsp_grafts:
        for param in graft.w_in.parameters():
            params.append(param)
        for param in graft.w_out.parameters():
            params.append(param)
    return params


def run_arm(arm, budget=None):
    import json

    import torch
    import transformers

    from wsp import load
    from wsp.lora import apply_lora, rank_for_budget

    tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    items = dev_items()
    pairs, align_texts = train_pairs_and_align_texts(tokenizer)
    row = {"arm": arm, "seed": 0, "items": len(items), "train_pairs": len(pairs)}

    if arm == "host":
        model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
        row["mmlu_dev"] = score(model, tokenizer, items)
        return row

    if arm == "lora":
        model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda")
        rank = rank_for_budget(model, budget)
        params = apply_lora(model, rank)
        for param in params:
            param.data = param.data.to("cuda")
        row["lora_rank"] = rank
        lr = LORA_LR
    else:
        folder = compile_with_retry(FOLDER_BY_ARM.get(arm, arm), align_texts)
        with open(os.path.join(folder, "compile_report.json")) as f:
            graft = json.load(f)["grafts"][0]
        row["layers"] = graft.get("layers")
        stitch = graft.get("stitch") or {}
        row["stitch_before"] = stitch.get("before")
        row["stitch_after"] = stitch.get("after")
        model, tokenizer = load(folder, device="cuda")
        if arm == "trunk_shuffled":
            row["shuffled_params"] = shuffle_part_weights(model)
        row["mmlu_dev_before_training"] = score(model, tokenizer, items)
        params = glue_params(model)
        lr = GLUE_LR

    count = 0
    for param in params:
        count += param.numel()
    row["trainable"] = count
    losses = train_answers(model, tokenizer, pairs, params, lr)
    row["loss_first10"] = sum(losses[:10]) / 10
    row["loss_last10"] = sum(losses[-10:]) / 10
    row["mmlu_dev"] = score(model, tokenizer, items)
    return row
