"""Extraction: find donor parts with a known function and write them as index files.

Interp is now the extraction step (brief update, 2026-11-14). For a donor
model we find, per layer:
  mlp_cluster  MLP neurons that fire much more on skill text than on control text
  attn_heads   query heads whose output is much larger on skill text
  expert       MoE experts the router picks much more often on skill text

Method: activation contrast. Run the donor on skill text (GSM8K train, heal
portion, never the search or held-out items) and on control text (WikiText-2).
Score each unit by an effect size:
  neurons, heads   Cohen's d of |activation| (neurons) or per-head output norm
  experts          log2 ratio of how often the router picks the expert
Pick the top n units per layer and the best layers. Each pick is written as a
.pt index file {kind, model, layer, indices, signature}, together with random
same-size controls from the same layer (the main control, one per seed).

Validation ("known function"): `validate` knocks the units out of the donor
and measures the drop on the task's search split, next to knocking out the
random control. A part has a measured function only if its drop beats the
random drop by more than the noise.

Records for Person A (who stores components) go to <out_dir>/records.jsonl.
"""
import datetime
import json
import math
import os

from wsp_eval import config, harness, splits, tasks as tasks_module, units

DEFAULT_UNITS = {"mlp_cluster": 64, "attn_heads": 2, "expert": 4}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(hypothesis, action, result, log_path=None):
    if log_path is None:
        log_path = config.interp_log_path()
    record = {"time": _now(), "hypothesis": hypothesis, "action": action, "result": result}
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print("[extract]", action.get("kind", ""), action.get("layer", ""), json.dumps(result)[:160])


# ---------------------------------------------------------------------------
# Text sources
# ---------------------------------------------------------------------------

def skill_texts(task_name="gsm8k", count=256):
    """Skill text from the task's heal (training) split. Never search or held-out items."""
    texts = splits.heal_texts(task_name)
    return texts[:count]


def control_texts(count=256, min_chars=300):
    """Generic English prose (WikiText-2 train; small download)."""
    data = tasks_module._load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = []
    for row in data:
        text = row["text"].strip()
        if len(text) >= min_chars and not text.startswith("="):
            texts.append(text)
        if len(texts) >= count:
            break
    return texts


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

class RunningStats:
    """Per-unit count, sum, and sum of squares."""

    def __init__(self):
        self.n = 0
        self.sum = None
        self.sumsq = None

    def add(self, values):
        # values: [tokens, units] float tensor
        import torch
        values = values.double()
        total = values.sum(dim=0).cpu()
        total_sq = (values * values).sum(dim=0).cpu()
        if self.sum is None:
            self.sum = torch.zeros_like(total)
            self.sumsq = torch.zeros_like(total_sq)
        self.sum += total
        self.sumsq += total_sq
        self.n += values.shape[0]

    def mean_var(self):
        mean = self.sum / max(self.n, 1)
        var = self.sumsq / max(self.n, 1) - mean * mean
        return mean, var.clamp(min=0)


class RouterCounts:
    def __init__(self):
        self.tokens = 0
        self.counts = None
        self.top_k = None

    def add(self, chosen, num_experts):
        import torch
        if self.counts is None:
            self.counts = torch.zeros(num_experts, dtype=torch.float64)
        flat = chosen.reshape(-1).cpu()
        self.counts += torch.bincount(flat, minlength=num_experts).double()
        self.tokens += chosen.shape[0]
        self.top_k = chosen.shape[-1]


def load_donor(model_id, device=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, tokenizer, device


def collect_stats(model, tokenizer, device, texts, kinds, layers, max_tokens=512, batch_size=8):
    """Run texts through the model and gather per-unit statistics for each (kind, layer)."""
    import torch

    all_layers = units.decoder_layers(model)
    size = units.head_dim(model)
    stats = {}
    current = {"mask": None}
    handles = []

    def valid_rows(tensor):
        # tensor [B, T, ...] -> [valid tokens, ...]
        mask = current["mask"]
        if tensor.dim() >= 3 and mask is not None and tensor.shape[0] == mask.shape[0] and tensor.shape[1] == mask.shape[1]:
            return tensor[mask]
        return tensor.reshape(-1, tensor.shape[-1])

    for layer_index in layers:
        layer = all_layers[layer_index]
        if "mlp_cluster" in kinds:
            try:
                module = units.mlp_down_proj(layer)
                key = ("mlp_cluster", layer_index)
                stats[key] = RunningStats()

                def neuron_hook(m, args, key=key):
                    stats[key].add(valid_rows(args[0].detach()).abs().float())
                handles.append(module.register_forward_pre_hook(neuron_hook))
            except ValueError:
                pass
        if "attn_heads" in kinds:
            module = units.attn_o_proj(layer)
            key = ("attn_heads", layer_index)
            stats[key] = RunningStats()

            def head_hook(m, args, key=key):
                rows = valid_rows(args[0].detach()).float()
                heads = rows.reshape(rows.shape[0], -1, size)
                stats[key].add(heads.norm(dim=-1))
            handles.append(module.register_forward_pre_hook(head_hook))
        if "expert" in kinds:
            try:
                module = units.moe_router(layer)
                key = ("expert", layer_index)
                stats[key] = RouterCounts()
                top_k = getattr(model.config, "num_experts_per_tok", None) or getattr(model.config, "top_k", 2)

                def router_hook(m, args, output, key=key, top_k=top_k):
                    logits = output[0] if isinstance(output, tuple) else output
                    logits = logits.detach().reshape(-1, logits.shape[-1])
                    mask = current["mask"]
                    if mask is not None and mask.numel() == logits.shape[0]:
                        logits = logits[mask.reshape(-1)]
                    chosen = logits.topk(top_k, dim=-1).indices
                    stats[key].add(chosen, logits.shape[-1])
                handles.append(module.register_forward_hook(router_hook))
            except ValueError:
                pass

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
            encoded = encoded.to(device)
            current["mask"] = encoded["attention_mask"].bool()
            with torch.no_grad():
                model(**encoded)
    finally:
        for handle in handles:
            handle.remove()
    return stats


def unit_scores(kind, skill_stat, control_stat):
    """Effect size per unit: positive = more active on skill text."""
    import torch
    if kind == "expert":
        skill_freq = skill_stat.counts / max(skill_stat.tokens * skill_stat.top_k, 1)
        control_freq = control_stat.counts / max(control_stat.tokens * control_stat.top_k, 1)
        eps = 1e-4
        return torch.log2((skill_freq + eps) / (control_freq + eps))
    skill_mean, skill_var = skill_stat.mean_var()
    control_mean, control_var = control_stat.mean_var()
    pooled = torch.sqrt((skill_var + control_var) / 2 + 1e-8)
    return (skill_mean - control_mean) / pooled


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def short_name(model_id):
    name = model_id.split("/")[-1].lower()
    for char in ".-":
        name = name.replace(char, "")
    return name


def extract(model_id, task_name="gsm8k", kinds=None, layers=None, n_units=None, top_layers=3,
            n_texts=256, out_dir=None, control_seeds=(0, 1, 2), model=None, tokenizer=None, device=None,
            skill=None, control=None):
    """Find skill units in a donor and write index files, controls, and records.jsonl.

    Returns the list of catalog records written (for Person A).
    """
    import random
    import torch

    if kinds is None:
        kinds = ["mlp_cluster", "attn_heads", "expert"]
    unit_sizes = dict(DEFAULT_UNITS)
    if n_units is not None:
        for key, value in n_units.items():
            unit_sizes[key] = value
    if out_dir is None:
        out_dir = os.path.join("outputs", "extracted", short_name(model_id))
    os.makedirs(out_dir, exist_ok=True)

    if model is None:
        model, tokenizer, device = load_donor(model_id)
    model_config = model.config.to_dict()
    num_layers = model_config["num_hidden_layers"]
    if layers is None:
        layers = list(range(num_layers))
    if skill is None:
        skill = skill_texts(task_name, n_texts)
    if control is None:
        control = control_texts(n_texts)

    log("units in " + model_id + " respond to " + task_name + " text more than to generic text",
        {"kind": "collect_stats", "model": model_id, "layers": [layers[0], layers[-1]], "kinds": kinds},
        {"skill_texts": len(skill), "control_texts": len(control)})
    skill_stats = collect_stats(model, tokenizer, device, skill, kinds, layers)
    control_stats = collect_stats(model, tokenizer, device, control, kinds, layers)

    # Score every (kind, layer): the top-n units and the mean effect of a random same-size set.
    candidates = {}
    for key in skill_stats:
        kind, layer = key
        scores = unit_scores(kind, skill_stats[key], control_stats[key])
        n = min(unit_sizes[kind], scores.numel() // 2)
        if n <= 0:
            continue
        top = torch.topk(scores, n)
        rng = random.Random(str(config.SEED) + ":" + str(key))
        random_units = rng.sample(range(scores.numel()), n)
        random_mean = float(scores[random_units].mean())
        entry = {"kind": kind, "layer": layer, "indices": top.indices.tolist(),
                 "unit_effects": [round(float(x), 4) for x in top.values],
                 "mean_effect": float(top.values.mean()), "random_mean_effect": random_mean,
                 "n_units_in_layer": int(scores.numel())}
        if kind not in candidates:
            candidates[kind] = []
        candidates[kind].append(entry)

    if kinds and "expert" in kinds and "expert" not in candidates:
        log("the donor has MoE experts", {"kind": "skip", "model": model_id}, {"reason": "no MoE router found"})

    records = []
    for kind, entries in candidates.items():
        ordered = sorted(entries, key=lambda e: e["mean_effect"], reverse=True)
        rank = 0
        for entry in ordered[:top_layers]:
            rank += 1
            layer = entry["layer"]
            component_id = kind + "_" + short_name(model_id) + "_l" + str(layer) + "_" + task_name
            signature = {
                "function": "more active on " + task_name + " text than on generic text",
                "method": ("activation contrast: Cohen's d of |activation|" if kind == "mlp_cluster" else
                           "activation contrast: Cohen's d of head output norm" if kind == "attn_heads" else
                           "router contrast: log2 ratio of expert pick rates"),
                "skill_data": task_name + " heal split (train), " + str(len(skill)) + " texts",
                "control_data": "WikiText-2 train, " + str(len(control)) + " texts",
                "n_units": len(entry["indices"]),
                "n_units_in_layer": entry["n_units_in_layer"],
                "unit_effects": entry["unit_effects"],
                "mean_effect": round(entry["mean_effect"], 4),
                "random_mean_effect": round(entry["random_mean_effect"], 4),
                "layer_rank": rank,
                "extracted": _now(),
            }
            path = os.path.join(out_dir, component_id + ".pt")
            units.save_index_file(path, kind, model_id, layer, entry["indices"], signature=signature)

            control_paths = []
            for seed in control_seeds:
                control_path = os.path.join(out_dir, component_id + "_control_seed" + str(seed) + ".pt")
                units.random_control(path, control_path, model_config, seed)
                control_paths.append(control_path)

            record = {"id": component_id, "type": kind, "model": model_id, "family": model_config.get("model_type"),
                      "layer": layer, "units": path, "trained_on": task_name, "controls": control_paths,
                      "signature": signature}
            records.append(record)
            log("layer " + str(layer) + " of " + model_id + " holds " + task_name + "-specific " + kind,
                {"kind": "write_index", "model": model_id, "layer": layer, "unit_kind": kind},
                {"path": path, "mean_effect": signature["mean_effect"],
                 "random_mean_effect": signature["random_mean_effect"]})

    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return records


# ---------------------------------------------------------------------------
# Validation by ablation in the donor
# ---------------------------------------------------------------------------

def validate(index_path, control_path, tasks=None, split="search", namespace="interp", noise_multiplier=2.0):
    """Knock the units out of the donor; compare with knocking out the random control.

    Returns a dict that is also merged into the index file's signature.
    Costs three eval() calls (intact, ablated, control-ablated) in `namespace`.
    """
    import torch
    data = units.load_index_file(index_path)
    if tasks is None:
        tasks = ["gsm8k"]
    model_id = data["model"]

    intact = harness.evaluate(model_id, split, tasks, namespace=namespace, note="donor intact")
    ablated = harness.evaluate(model_id, split, tasks, namespace=namespace,
                               ablation=units.ablation_from_file(index_path), note="ablate " + index_path)
    random_ablated = harness.evaluate(model_id, split, tasks, namespace=namespace,
                                      ablation=units.ablation_from_file(control_path), note="ablate " + control_path)

    result = {"split": split, "label": intact["label"], "tasks": {}}
    for task in tasks:
        base = intact["tasks"][task]["score"]
        n = intact["tasks"][task]["n"]
        drop = base - ablated["tasks"][task]["score"]
        random_drop = base - random_ablated["tasks"][task]["score"]
        noise = noise_multiplier * math.sqrt(max(base * (1 - base), 1e-6) / n) * math.sqrt(2)
        result["tasks"][task] = {
            "intact": base, "ablated": ablated["tasks"][task]["score"],
            "random_ablated": random_ablated["tasks"][task]["score"],
            "drop": drop, "random_drop": random_drop, "noise": noise, "n": n,
            "specific": drop - random_drop > noise,
        }

    signature = data.get("signature", {})
    signature["ablation"] = result
    data["signature"] = signature
    torch.save(data, index_path)

    first = result["tasks"][tasks[0]]
    log("knocking out " + os.path.basename(index_path) + " hurts " + tasks[0] + " more than a random set",
        {"kind": "validate", "model": model_id, "layer": data["layer"], "unit_kind": data["kind"]},
        {"label": result["label"], "drop": first["drop"], "random_drop": first["random_drop"],
         "noise": first["noise"], "specific": first["specific"]})
    return result
