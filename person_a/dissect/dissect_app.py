# Modal GPU side of the dissection pipeline (first pair: donor Qwen2.5-Math-1.5B, task GSM8K).
#
# Deploy once (and after any change to this file or dissect/data/tasks.json):
#   cd ~/swarmy && .venv/bin/modal deploy dissect/dissect_app.py
# Local code then calls it through dissect/client.py.
#
# A "part" is a plain dict:
#   {"kind": "mlp_cluster", "layer": 14, "indices": [5, 99, ...]}   neurons of one MLP (input of down_proj)
#   {"kind": "attn_heads",  "layer": 3,  "indices": [0, 7]}         query heads (slice of o_proj input)
#   {"kind": "block",       "layers": [10, 11]}                     whole decoder layers
# Ablation is done with forward hooks only. The model weights are never changed.
import json
import os
import random
import time

import modal

APP_NAME = "swarmy-dissect"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
GPU = "L4"
SEED = 0

LOCAL_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tasks.json")
REMOTE_TASKS = "/root/data/tasks.json"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("swarmy-models")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "transformers==4.57.6", "accelerate", "safetensors", "huggingface_hub")
    .env({"HF_HOME": "/models/hf_cache"})
    .add_local_file(LOCAL_TASKS, REMOTE_TASKS)
)

# Task names the scorer understands
#   gsm8k          200 test problems, prob of the digit tokens in the gold worked solution (2-shot)
#   gsm8k_heldout  100 other test problems, same metric (only used by validation)
#   trivia         100 TriviaQA questions, prob of the gold answer tokens (3-shot)
#   wiki           100 WikiText paragraphs, prob of every token
#   wiki_digits    same paragraphs, prob of digit tokens only (numbers without arithmetic)
#   gsm8k_gen          60 of the gsm8k problems, greedy generation, exact-match accuracy (the gated target)
#   gsm8k_gen_heldout  60 of the held-out problems, same (only used by validation)
LP_TASKS = ["gsm8k", "gsm8k_heldout", "trivia", "wiki", "wiki_digits"]


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=3600,
    scaledown_window=240,
)
class Dissector:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(SEED)
        start = time.time()
        token = os.environ.get("HF_TOKEN")
        self.tok = AutoTokenizer.from_pretrained(DONOR, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            DONOR, torch_dtype=torch.bfloat16, device_map="cuda", token=token
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            volume.commit()  # keep the HF cache for the next container
        except Exception as e:
            print("volume commit skipped:", e)

        cfg = self.model.config
        self.layers = self.model.model.layers
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.mlp_width = cfg.intermediate_size
        self.d_model = cfg.hidden_size

        with open(REMOTE_TASKS) as f:
            self.tasks = json.load(f)

        # Which vocabulary entries contain a digit (for the digit-only metrics)
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        is_digit = torch.zeros(vocab_size, dtype=torch.bool)
        for i in range(len(self.tok)):
            text = self.tok.decode([i])
            for ch in text:
                if ch.isdigit():
                    is_digit[i] = True
                    break
        self.is_digit = is_digit.cuda()

        self.items = self._build_items()
        self.means = self._compute_means()
        self.base_cache = {}
        print("loaded in", round(time.time() - start, 1), "s")

    # ---------- task construction ----------

    def _encode(self, context, continuation, score_mode):
        ctx_ids = []
        if context:
            ctx_ids = self.tok(context, add_special_tokens=False)["input_ids"]
        cont_ids = self.tok(continuation, add_special_tokens=False)["input_ids"]
        return {"ids": ctx_ids + cont_ids, "start": max(len(ctx_ids), 1), "mode": score_mode}

    def _gsm8k_prefix(self):
        text = ""
        for ex in self.tasks["gsm8k_fewshot"]:
            text = text + "Question: " + ex["question"] + "\nAnswer: " + ex["solution"] + "\n\n"
        return text

    def _trivia_prefix(self):
        text = ""
        for ex in self.tasks["trivia_fewshot"]:
            text = text + "Question: " + ex["question"] + "\nAnswer: " + ex["answer"] + "\n\n"
        return text

    def _build_items(self):
        items = {}
        gsm_prefix = self._gsm8k_prefix()
        trivia_prefix = self._trivia_prefix()

        for split, task in [("gsm8k_score", "gsm8k"), ("gsm8k_heldout", "gsm8k_heldout")]:
            rows = []
            for ex in self.tasks[split]:
                context = gsm_prefix + "Question: " + ex["question"] + "\nAnswer:"
                rows.append(self._encode(context, " " + ex["solution"], "digits"))
            items[task] = rows

        rows = []
        for ex in self.tasks["trivia_score"]:
            context = trivia_prefix + "Question: " + ex["question"] + "\nAnswer:"
            rows.append(self._encode(context, " " + ex["answer"], "all"))
        items["trivia"] = rows

        rows = []
        for text in self.tasks["wiki_score"]:
            rows.append(self._encode("", text, "all"))
        items["wiki"] = rows  # wiki_digits is read from the same pass

        # Proposal sets (no few-shot, shorter): math text vs control text
        rows = []
        for ex in self.tasks["gsm8k_propose"]:
            rows.append(self._encode("Question: " + ex["question"] + "\nAnswer:", " " + ex["solution"], "digits"))
        items["propose_math"] = rows
        rows = []
        for ex in self.tasks["trivia_propose"]:
            rows.append(self._encode("Question: " + ex["question"] + "\nAnswer:", " " + ex["answer"], "all"))
        for text in self.tasks["wiki_propose"]:
            rows.append(self._encode("", text, "all"))
        items["propose_control"] = rows

        # Mean reference: a mix of all three kinds of text
        rows = []
        for ex in self.tasks["gsm8k_meanref"]:
            rows.append(self._encode("Question: " + ex["question"] + "\nAnswer:", " " + ex["solution"], "all"))
        for ex in self.tasks["trivia_meanref"]:
            rows.append(self._encode("Question: " + ex["question"] + "\nAnswer:", " " + ex["answer"], "all"))
        for text in self.tasks["wiki_meanref"]:
            rows.append(self._encode("", text, "all"))
        items["meanref"] = rows
        return items

    def _batches(self, rows, batch_size):
        # Sort by length so padding is small; yields (batch_rows, input_ids, attention_mask)
        import torch

        order = list(range(len(rows)))
        order.sort(key=lambda i: len(rows[i]["ids"]))
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        for b in range(0, len(order), batch_size):
            batch_rows = []
            for i in order[b:b + batch_size]:
                batch_rows.append(rows[i])
            max_len = 0
            for r in batch_rows:
                if len(r["ids"]) > max_len:
                    max_len = len(r["ids"])
            ids = torch.full((len(batch_rows), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch_rows), max_len), dtype=torch.long)
            for j in range(len(batch_rows)):
                n = len(batch_rows[j]["ids"])
                ids[j, :n] = torch.tensor(batch_rows[j]["ids"])
                mask[j, :n] = 1
            yield batch_rows, ids.cuda(), mask.cuda()

    def _score_mask(self, batch_rows, ids, mask, use_digits):
        # Which target positions count. Target at position t is token t, predicted from t-1.
        import torch

        score = torch.zeros_like(mask, dtype=torch.bool)
        for j in range(len(batch_rows)):
            n = len(batch_rows[j]["ids"])
            score[j, batch_rows[j]["start"]:n] = True
        if use_digits:
            score = score & self.is_digit[ids]
        return score

    # ---------- means for mean-ablation ----------

    def _compute_means(self):
        import torch

        L = self.n_layers
        sums = {
            "mlp": torch.zeros(L, self.mlp_width, device="cuda"),
            "attn": torch.zeros(L, self.d_model, device="cuda"),
            "delta": torch.zeros(L, self.d_model, device="cuda"),
        }
        state = {"mask": None, "count": 0.0}
        handles = []

        def make_mlp_hook(layer):
            def hook(module, args):
                x = args[0].float()
                m = state["mask"].unsqueeze(-1).float()
                sums["mlp"][layer] += (x * m).sum(dim=(0, 1))
            return hook

        def make_attn_hook(layer):
            def hook(module, args):
                x = args[0].float()
                m = state["mask"].unsqueeze(-1).float()
                sums["attn"][layer] += (x * m).sum(dim=(0, 1))
            return hook

        def make_layer_hook(layer):
            def hook(module, args, kwargs, output):
                hs_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
                out = output[0] if isinstance(output, tuple) else output
                m = state["mask"].unsqueeze(-1).float()
                sums["delta"][layer] += ((out.float() - hs_in.float()) * m).sum(dim=(0, 1))
            return hook

        for l in range(L):
            handles.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(make_mlp_hook(l)))
            handles.append(self.layers[l].self_attn.o_proj.register_forward_pre_hook(make_attn_hook(l)))
            handles.append(self.layers[l].register_forward_hook(make_layer_hook(l), with_kwargs=True))
        try:
            with torch.no_grad():
                for batch_rows, ids, mask in self._batches(self.items["meanref"], 16):
                    state["mask"] = mask
                    state["count"] += float(mask.sum().item())
                    self.model.model(input_ids=ids, attention_mask=mask)
        finally:
            for h in handles:
                h.remove()
        means = {}
        for key in sums:
            means[key] = sums[key] / state["count"]
        return means

    # ---------- ablation hooks ----------

    def _head_columns(self, heads):
        cols = []
        for h in heads:
            for c in range(h * self.head_dim, (h + 1) * self.head_dim):
                cols.append(c)
        return cols

    def _install(self, part, method):
        import torch

        handles = []
        if part is None:
            return handles
        kind = part["kind"]

        if kind == "mlp_cluster" or kind == "attn_heads":
            layer = int(part["layer"])
            if kind == "mlp_cluster":
                cols = []
                for i in part["indices"]:
                    cols.append(int(i))
                mean_vec = self.means["mlp"][layer]
                target = self.layers[layer].mlp.down_proj
            else:
                cols = self._head_columns(part["indices"])
                mean_vec = self.means["attn"][layer]
                target = self.layers[layer].self_attn.o_proj
            if len(cols) == 0:
                return handles
            idx = torch.tensor(cols, dtype=torch.long, device="cuda")
            fill = mean_vec[idx]

            def pre_hook(module, args):
                x = args[0].clone()
                if method == "zero":
                    x[..., idx] = 0
                else:
                    x[..., idx] = fill.to(x.dtype)
                rest = tuple(args[1:])
                return (x,) + rest

            handles.append(target.register_forward_pre_hook(pre_hook))

        elif kind == "block":
            for layer in part["layers"]:
                layer = int(layer)
                delta = self.means["delta"][layer]

                def make_hook(d):
                    def hook(module, args, kwargs, output):
                        hs_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
                        if method == "zero":
                            new = hs_in
                        else:
                            new = hs_in + d.to(hs_in.dtype)
                        if isinstance(output, tuple):
                            return (new,) + tuple(output[1:])
                        return new
                    return hook

                handles.append(self.layers[layer].register_forward_hook(make_hook(delta), with_kwargs=True))
        else:
            raise ValueError("unknown part kind: " + str(kind))
        return handles

    # ---------- scoring ----------

    def _lp_scores(self, task_names):
        # Log-prob metrics. score = exp(mean log-prob of scored tokens), i.e. geometric-mean token prob.
        import torch

        wanted = []
        for t in task_names:
            if t == "wiki_digits":
                if "wiki" not in wanted:
                    wanted.append("wiki")
            elif t in LP_TASKS:
                if t not in wanted:
                    wanted.append(t)

        totals = {}
        lm_head = self.model.get_output_embeddings()
        with torch.no_grad():
            for task in wanted:
                rows = self.items[task]
                use_digits = rows[0]["mode"] == "digits"
                sums = {"main": 0.0, "digits": 0.0}
                counts = {"main": 0, "digits": 0}
                for batch_rows, ids, mask in self._batches(rows, 8):
                    hidden = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
                    score_all = self._score_mask(batch_rows, ids, mask, use_digits)
                    score_dig = self._score_mask(batch_rows, ids, mask, True)
                    # predict token t from position t-1
                    sel = score_all[:, 1:] | score_dig[:, 1:]
                    h = hidden[:, :-1][sel]
                    targets = ids[:, 1:][sel]
                    logp = torch.log_softmax(lm_head(h).float(), dim=-1)
                    tok_lp = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
                    is_main = score_all[:, 1:][sel]
                    is_dig = score_dig[:, 1:][sel]
                    sums["main"] += float(tok_lp[is_main].sum().item())
                    counts["main"] += int(is_main.sum().item())
                    sums["digits"] += float(tok_lp[is_dig].sum().item())
                    counts["digits"] += int(is_dig.sum().item())
                totals[task] = {"lp_sum": sums["main"], "n": counts["main"]}
                if task == "wiki":
                    totals["wiki_digits"] = {"lp_sum": sums["digits"], "n": counts["digits"]}

        out = {}
        for t in task_names:
            if t in totals:
                n = totals[t]["n"]
                mean_lp = totals[t]["lp_sum"] / max(n, 1)
                out[t] = {"score": round(float(torch.tensor(mean_lp).exp().item()), 5),
                          "nll": round(-mean_lp, 5), "n_tokens": n}
        return out

    def _gen_accuracy(self, split="gsm8k_score", n_items=60, max_new_tokens=256):
        import re
        import torch

        prefix = self._gsm8k_prefix()
        prompts = []
        answers = []
        for ex in self.tasks[split][:n_items]:
            prompts.append(prefix + "Question: " + ex["question"] + "\nAnswer:")
            answers.append(ex["answer"])
        self.tok.padding_side = "left"
        correct = 0
        samples = []
        for b in range(0, len(prompts), 30):
            enc = self.tok(prompts[b:b + 30], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          stop_strings=["\nQuestion:"], tokenizer=self.tok,
                                          pad_token_id=self.tok.pad_token_id)
            for j in range(out.shape[0]):
                text = self.tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                text = text.split("\nQuestion:")[0]
                m = re.search(r"The answer is\s*\$?(-?[\d,]*\.?\d+)", text)
                if m:
                    pred = m.group(1)
                else:
                    nums = re.findall(r"-?[\d,]*\.?\d+", text)
                    pred = nums[-1] if nums else ""
                pred = pred.replace(",", "").rstrip(".")
                gold = answers[b + j]
                ok = False
                try:
                    ok = abs(float(pred) - float(gold)) < 1e-6
                except ValueError:
                    ok = False
                if ok:
                    correct += 1
                if len(samples) < 2:
                    samples.append({"pred": pred, "gold": gold, "text": text[:300]})
        self.tok.padding_side = "right"
        return {"score": round(correct / len(prompts), 4), "n": len(prompts), "samples": samples}

    def _score(self, part, method, task_names):
        handles = self._install(part, method)
        try:
            out = self._lp_scores(task_names)
            if "gsm8k_gen" in task_names:
                out["gsm8k_gen"] = self._gen_accuracy("gsm8k_score")
            if "gsm8k_gen_heldout" in task_names:
                out["gsm8k_gen_heldout"] = self._gen_accuracy("gsm8k_heldout")
        finally:
            for h in handles:
                h.remove()
        return out

    def _base(self, task_names):
        need = []
        for t in task_names:
            if t not in self.base_cache:
                need.append(t)
        if len(need) > 0:
            got = self._score(None, "zero", need)
            for t in got:
                self.base_cache[t] = got[t]
        result = {}
        for t in task_names:
            result[t] = self.base_cache[t]
        return result

    def _drops(self, base, ablated):
        drops = {}
        for t in ablated:
            b = base[t]["score"]
            a = ablated[t]["score"]
            drops[t] = round((b - a) / b, 4) if b > 0 else None
        return drops

    # ---------- public methods ----------

    @modal.method()
    def model_info(self):
        return {"model": DONOR, "num_layers": self.n_layers, "num_heads": self.n_heads,
                "num_kv_heads": self.model.config.num_key_value_heads, "head_dim": self.head_dim,
                "mlp_width": self.mlp_width, "d_model": self.d_model,
                "part_kinds": ["mlp_cluster", "attn_heads", "block"]}

    @modal.method()
    def base_scores(self, task_names):
        return self._base(task_names)

    @modal.method()
    def ablate_and_score(self, part, method, task_names):
        base = self._base(task_names)
        ablated = self._score(part, method, task_names)
        return {"base": base, "ablated": ablated, "drop": self._drops(base, ablated)}

    def _random_baseline(self, part, n, method, task_names):
        # n random parts of the same kind, size and layer (blocks: same length, any start)
        rng = random.Random(SEED + 1)
        base = self._base(task_names)
        results = []
        for k in range(n):
            if part["kind"] == "mlp_cluster":
                pool = list(range(self.mlp_width))
                rnd = {"kind": "mlp_cluster", "layer": part["layer"], "indices": rng.sample(pool, len(part["indices"]))}
            elif part["kind"] == "attn_heads":
                pool = list(range(self.n_heads))
                rnd = {"kind": "attn_heads", "layer": part["layer"], "indices": rng.sample(pool, len(part["indices"]))}
            else:
                length = len(part["layers"])
                start = rng.randint(0, self.n_layers - length)
                rnd = {"kind": "block", "layers": list(range(start, start + length))}
            ablated = self._score(rnd, method, task_names)
            results.append(self._drops(base, ablated))
        return {"base": base, "random_drops": results}

    def _activation_profile(self, kind, top_k=20, layers=None):
        # Mean |activation| per unit on math text vs control text. Ranks units that fire more on math.
        import torch

        if layers is None:
            layers = list(range(self.n_layers))
        width = self.mlp_width if kind == "mlp_cluster" else self.n_heads
        stats = {}
        for name in ["propose_math", "propose_control"]:
            sums = torch.zeros(self.n_layers, width, device="cuda")
            state = {"mask": None, "count": 0.0}
            handles = []

            def make_hook(layer):
                def hook(module, args):
                    x = args[0].float()
                    if kind == "attn_heads":
                        b, s, _ = x.shape
                        x = x.view(b, s, self.n_heads, self.head_dim).norm(dim=-1)
                    else:
                        x = x.abs()
                    m = state["mask"].unsqueeze(-1).float()
                    sums[layer] += (x * m).sum(dim=(0, 1))
                return hook

            for l in layers:
                if kind == "mlp_cluster":
                    target = self.layers[l].mlp.down_proj
                else:
                    target = self.layers[l].self_attn.o_proj
                handles.append(target.register_forward_pre_hook(make_hook(l)))
            try:
                with torch.no_grad():
                    for batch_rows, ids, mask in self._batches(self.items[name], 16):
                        state["mask"] = mask
                        state["count"] += float(mask.sum().item())
                        self.model.model(input_ids=ids, attention_mask=mask)
            finally:
                for h in handles:
                    h.remove()
            stats[name] = sums / state["count"]

        math_mean = stats["propose_math"]
        ctrl_mean = stats["propose_control"]
        diff = math_mean - ctrl_mean
        allowed = torch.full_like(diff, float("-inf"))
        for l in layers:
            allowed[l] = diff[l]
        flat = allowed.flatten()
        top = torch.topk(flat, top_k)
        units = []
        for value, pos in zip(top.values.tolist(), top.indices.tolist()):
            layer = pos // width
            unit = pos % width
            units.append({"layer": layer, "unit": unit, "diff": round(value, 4),
                          "math": round(float(math_mean[layer, unit]), 4),
                          "control": round(float(ctrl_mean[layer, unit]), 4)})
        per_layer = []
        k_layer = min(64, width)
        for l in layers:
            per_layer.append({"layer": l, "top_sum_diff": round(float(torch.topk(diff[l], k_layer).values.sum()), 3)})
        return {"kind": kind, "top_units": units, "per_layer": per_layer}

    def _attribution_scan(self, kind, top_k=20, layers=None, rank_by="selective"):
        # Attribution patching: estimated change in mean NLL if a unit is mean-ablated,
        #   effect = sum over tokens of grad . (mean - activation)
        # computed for the math loss (digit tokens of GSM8K solutions) and the control loss.
        import torch

        if layers is None:
            layers = list(range(self.n_layers))
        width = self.mlp_width if kind == "mlp_cluster" else self.n_heads
        effects = {}
        embed = self.model.model.embed_tokens

        for name in ["propose_math", "propose_control"]:
            total = torch.zeros(self.n_layers, width, device="cuda")
            n_tokens = 0
            for batch_rows, ids, mask in self._batches(self.items[name], 4):
                captured = {}

                def embed_hook(module, args, output):
                    return output.detach().requires_grad_(True)

                def make_hook(layer):
                    def hook(module, args):
                        x = args[0]
                        x.retain_grad()
                        captured[layer] = x
                    return hook

                handles = [embed.register_forward_hook(embed_hook)]
                for l in layers:
                    if kind == "mlp_cluster":
                        target = self.layers[l].mlp.down_proj
                    else:
                        target = self.layers[l].self_attn.o_proj
                    handles.append(target.register_forward_pre_hook(make_hook(l)))
                try:
                    with torch.enable_grad():
                        hidden = self.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
                        use_digits = name == "propose_math"
                        score = self._score_mask(batch_rows, ids, mask, use_digits)[:, 1:]
                        h = hidden[:, :-1][score]
                        targets = ids[:, 1:][score]
                        logp = torch.log_softmax(self.model.get_output_embeddings()(h).float(), dim=-1)
                        loss = -logp.gather(1, targets.unsqueeze(1)).sum()
                        loss.backward()
                    n_tokens += int(score.sum().item())
                    with torch.no_grad():
                        for l in layers:
                            x = captured[l]
                            g = x.grad.float()
                            if kind == "mlp_cluster":
                                d = self.means["mlp"][l] - x.float()
                                total[l] += (g * d).sum(dim=(0, 1))
                            else:
                                d = self.means["attn"][l] - x.float()
                                b, s, _ = d.shape
                                prod = (g * d).view(b, s, self.n_heads, self.head_dim).sum(dim=-1)
                                total[l] += prod.sum(dim=(0, 1))
                finally:
                    for hd in handles:
                        hd.remove()
            effects[name] = total / max(n_tokens, 1)

        math_eff = effects["propose_math"]
        ctrl_eff = effects["propose_control"]
        if rank_by == "selective":
            # penalise only harm to the controls; a unit whose removal helps the controls is not "math-selective"
            rank = math_eff - torch.clamp(ctrl_eff, min=0.0)
        else:
            rank = math_eff.clone()
        allowed = torch.full_like(rank, float("-inf"))
        for l in layers:
            allowed[l] = rank[l]
        top = torch.topk(allowed.flatten(), top_k)
        units = []
        for value, pos in zip(top.values.tolist(), top.indices.tolist()):
            layer = pos // width
            unit = pos % width
            units.append({"layer": layer, "unit": unit, "rank_value": round(value, 6),
                          "math_effect": round(float(math_eff[layer, unit]), 6),
                          "control_effect": round(float(ctrl_eff[layer, unit]), 6)})
        per_layer = []
        k_layer = min(64, width)
        for l in layers:
            per_layer.append({"layer": l,
                              "top_sum_rank": round(float(torch.topk(rank[l], k_layer).values.sum()), 5),
                              "top_sum_math": round(float(torch.topk(math_eff[l], k_layer).values.sum()), 5)})
        return {"kind": kind, "rank_by": rank_by, "top_units": units, "per_layer": per_layer,
                "note": "effects are estimated increases in mean NLL per scored token if mean-ablated"}

    @modal.method()
    def random_baseline(self, part, n, method, task_names):
        return self._random_baseline(part, n, method, task_names)

    @modal.method()
    def activation_profile(self, kind, top_k=20, layers=None):
        return self._activation_profile(kind, top_k, layers)

    @modal.method()
    def attribution_scan(self, kind, top_k=20, layers=None, rank_by="selective"):
        return self._attribution_scan(kind, top_k, layers, rank_by)

    @modal.method()
    def top_units_in_layer(self, kind, layer, k, source="attribution", rank_by="selective"):
        # Convenience: the k best units of one layer, ready to form a part.
        if source == "attribution":
            res = self._attribution_scan(kind, top_k=k, layers=[layer], rank_by=rank_by)
        else:
            res = self._activation_profile(kind, top_k=k, layers=[layer])
        indices = []
        for u in res["top_units"]:
            indices.append(u["unit"])
        return {"kind": kind, "layer": layer, "indices": indices, "units": res["top_units"][:10]}

    @modal.method()
    def index_file_bytes(self, part):
        # The part's index file in the team format: torch .pt dict {kind, model, layer, indices: LongTensor}
        import io
        import torch

        if part["kind"] == "block":
            indices = torch.tensor(part["layers"], dtype=torch.long)
            layer = int(part["layers"][0])
        else:
            indices = torch.tensor(part["indices"], dtype=torch.long)
            layer = int(part["layer"])
        buf = io.BytesIO()
        torch.save({"kind": part["kind"], "model": DONOR, "layer": layer, "indices": indices}, buf)
        return buf.getvalue()

    @modal.method()
    def validate(self, part, n_random=20, n_random_gen=10):
        # Everything the validation gate needs, in one call. Gate thresholds live in client.py.
        # Target = generation accuracy (score subset and held-out subset). Log-prob tasks give the controls.
        all_tasks = ["gsm8k", "gsm8k_heldout", "trivia", "wiki", "wiki_digits", "gsm8k_gen", "gsm8k_gen_heldout"]
        base = self._base(all_tasks)
        evidence = {"base": base, "methods": {}}
        for method in ["zero", "mean"]:
            ablated = self._score(part, method, all_tasks)
            evidence["methods"][method] = {"ablated": ablated, "drop": self._drops(base, ablated)}
        rnd = self._random_baseline(part, n_random, "mean", ["gsm8k"])
        random_lp = []
        for d in rnd["random_drops"]:
            random_lp.append(d["gsm8k"])
        evidence["random_gsm8k_drops_mean"] = random_lp
        rnd_gen = self._random_baseline(part, n_random_gen, "mean", ["gsm8k_gen"])
        random_gen = []
        for d in rnd_gen["random_drops"]:
            random_gen.append(d["gsm8k_gen"])
        evidence["random_gen_drops_mean"] = random_gen
        return evidence
