"""Train the way out of a donor block: distill the donor's own predictions through the composite.

Follow-up to wsp_eval/stitch_graft.py. There the input stitch removed 3/4 of
the donor penalty, but the graft scored 0.38 against a matched LoRA's 0.455.
The suspicion: the host's later layers cannot use the donor block's output
through a rank-64 w_out trained 300 steps on next-token loss.

Here every arm starts from its model compiled in that run and trains its
1.53M glue (graft: w_in stitch + w_out; LoRA: a fresh LoRA of the same
budget) for 1000 more steps with one objective: match Qwen2.5-Math-1.5B's
own next-token distribution (full-vocab KL, every non-pad position) on GSM8K
heal texts. Host and donor share a tokenizer, so the logits line up. The LoRA
gets the same donor signal, so a graft win cannot come from seeing the donor's
outputs alone.

Arms (seed 0), scored with eval() on the GSM8K search split (interim):
  graft    block_stitch_s0 (donor layers 8..19 at host layer 7)
  random   random_stitch_s0 (it drew donor layers 9..20: a weak control)
  lora     lora_s0 plus a fresh 1.53M LoRA
"""
import json
import os
import shutil

from wsp_eval import harness, splits

DONOR = "Qwen/Qwen2.5-Math-1.5B"
SOURCE_ROOT = "/vol/compiled/stitch_graft"
OUT_ROOT = "/vol/compiled/exit_stitch"
SOURCES = {"graft": "block_stitch_s0", "random": "random_stitch_s0", "lora": "lora_s0"}
BUDGET = 1531904
STEPS = 1000
LR = 3e-4
BATCH = 8
SEQ_LEN = 256
TEXT_OFFSET = 1200  # skip the texts the input stitch trained on


def arm_root(arm):
    return os.path.join("/vol/eval_exit", arm)


def glue_params(model):
    params = []
    for graft in model.wsp_grafts:
        for param in graft.w_in.parameters():
            params.append(param)
        for param in graft.w_out.parameters():
            params.append(param)
    return params


def distill(model, tokenizer, donor, texts, params, seed=0):
    """Train params so the model's next-token distribution matches the donor's. Return the losses."""
    import torch
    import torch.nn.functional as F

    for param in model.parameters():
        param.requires_grad_(False)
    for param in params:
        param.data = param.data.float()
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = []
    losses = []
    model.train()
    for step in range(STEPS):
        batch = []
        while len(batch) < BATCH:
            if len(order) == 0:
                order = torch.randperm(len(texts), generator=generator).tolist()
            batch.append(texts[order.pop()])
        enc = tokenizer(batch, truncation=True, max_length=SEQ_LEN, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            target = donor(**enc).logits.float()
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        logits = logits.float()
        vocab = min(logits.shape[-1], target.shape[-1])
        mask = enc["attention_mask"].bool()
        student = F.log_softmax(logits[..., :vocab][mask], dim=-1)
        teacher = F.log_softmax(target[..., :vocab][mask], dim=-1)
        loss = F.kl_div(student, teacher, log_target=True, reduction="batchmean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print("[exit_stitch] step", step + 1, "kl", round(losses[-1], 4), flush=True)
    model.eval()
    return losses


def save_graft_copy(model, source_dir, out_dir):
    """Copy the compiled graft folder and overwrite its graft weights with the trained ones."""
    import torch
    from safetensors.torch import load_file, save_file

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir)
    path = os.path.join(out_dir, "graft_weights.safetensors")
    original = load_file(path)
    state = model.wsp_grafts.state_dict()
    tensors = {}
    for key in original:
        tensors[key] = state[key].detach().to(original[key].dtype).contiguous().cpu()
    save_file(tensors, path)
    del original
    torch.cuda.empty_cache()


def run_arm(arm, seed=0):
    """Distill one arm, save it, and score it. Call with WSP_ROOT=arm_root(arm)."""
    import torch
    import transformers

    from wsp import load
    from wsp.lora import apply_lora, merge_lora, rank_for_budget

    source_dir = os.path.join(SOURCE_ROOT, SOURCES[arm])
    out_dir = os.path.join(OUT_ROOT, arm + "_s" + str(seed))
    texts = splits.heal_texts("gsm8k")[TEXT_OFFSET:]
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()

    model, tokenizer = load(source_dir, device="cuda")
    row = {"arm": arm, "seed": seed, "source": source_dir, "model": out_dir}
    if arm == "lora":
        rank = rank_for_budget(model, BUDGET)
        params = apply_lora(model, rank)
        for param in params:
            param.data = param.data.to("cuda")
        row["lora_rank"] = rank
    else:
        params = glue_params(model)
    count = 0
    for param in params:
        count += param.numel()
    row["trainable"] = count

    losses = distill(model, tokenizer, donor, texts, params, seed)
    row["kl_first10"] = sum(losses[:10]) / 10
    row["kl_last10"] = sum(losses[-10:]) / 10
    del donor
    torch.cuda.empty_cache()

    if arm == "lora":
        merge_lora(model)
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    else:
        save_graft_copy(model, source_dir, out_dir)
    with open(os.path.join(out_dir, "exit_stitch.json"), "w") as f:
        json.dump({"source": source_dir, "steps": STEPS, "lr": LR, "batch_size": BATCH, "seq_len": SEQ_LEN,
                   "objective": "KL(donor || model), full vocab, every non-pad position", "donor": DONOR,
                   "trainable": count, "kl_first10": row["kl_first10"], "kl_last10": row["kl_last10"]}, f, indent=2)
    del model
    torch.cuda.empty_cache()

    result = harness.evaluate(out_dir, "search", ["gsm8k"], namespace="exit_stitch", backend="hf",
                              note="exit stitch " + arm + " seed " + str(seed))
    row["gsm8k"] = result["tasks"]["gsm8k"]["score"]
    row["n"] = result["tasks"]["gsm8k"]["n"]
    row["label"] = result["label"]
    return row
