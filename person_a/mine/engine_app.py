# Modal GPU engine for the v2 mining pipeline: one class, parameterised by donor model.
#   .venv/bin/modal deploy mine/engine_app.py      (once, and after any change to mine/adapter.py or this file)
# Local code calls it through mine/client.py, which picks the GPU by model size.
#
# What it does:
#   - loads the full donor once per container (cached on the swarmy-models volume)
#   - scores every active skill in ONE teacher-forced pass (mean log-prob of the answer tokens)
#   - zero-ablates parts with hooks (see mine/adapter.py) and returns relative drops per skill
#   - activation / attribution scans and MoE routing profiles for proposing
#   - an optional greedy-generation math check ("confirmed" tier)
import json
import math
import os
import random
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "swarmy-mine"
VOLUME_NAME = "swarmy-models"
CUSTOM_CODE = ["k2horizon7b", "k2mova36b", "g9v3_39a5b", "mimo2.6_flash", "mimo2.6_pro"]
# FP8 checkpoints to dequantize to bf16 at load time. Backward passes (attribution scans) need bf16
# weights; only list a model here when its bf16 size fits the GPUs of its tier.
DEQUANTIZE = ["glm5.3_flash"]
SEED = 0

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.23.0", "accelerate", "safetensors", "huggingface_hub",
                 "hf_xet", "sentencepiece", "protobuf", "tiktoken", "einops", "flash-linear-attention",
                 "kernels==0.18.2")  # kernels: transformers needs it to run fine-grained FP8 checkpoints
    .env({"HF_HOME": "/models/hf_cache"})
    .add_local_file(os.path.join(HERE, "adapter.py"), "/root/adapter.py")
    .add_local_file(os.path.join(HERE, "data", "builtin_skills.json"), "/root/builtin_skills.json")
)


def rel_drop(base, ablated):
    if base is None or ablated is None or base <= 0:
        return None
    return round((base - ablated) / base, 4)


class EngineBase:
    # Shared implementation. Concrete GPU tiers are declared at the bottom of this file
    # (EngineL4, EngineL40S, EngineH100, EngineH200), each with Modal GPU fallbacks.
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()

    @modal.enter()
    def load(self):
        import sys
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        sys.path.insert(0, "/root")
        import adapter

        self.adapter = adapter
        torch.manual_seed(SEED)
        start = time.time()
        token = os.environ.get("HF_TOKEN")
        remote = self.model_key in CUSTOM_CODE
        kw = {"revision": self.revision, "token": token, "trust_remote_code": remote}
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, **kw)
        self.config = AutoConfig.from_pretrained(self.hf_id, **kw)
        model = None
        errors = []
        load_kw = {"device_map": "cuda"}
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            # Multi-GPU tiers: spread the layers over all GPUs and leave room on each for activations.
            # Inputs go to cuda:0; the scoring code moves logits back to the input device.
            per_gpu = int(torch.cuda.get_device_properties(0).total_memory * 0.85 / 2**30)
            load_kw = {"device_map": "auto", "max_memory": {i: str(per_gpu) + "GiB" for i in range(n_gpu)}}
        if self.model_key in DEQUANTIZE:
            from transformers import FineGrainedFP8Config
            load_kw["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        print("loading on", n_gpu, "GPU(s):", {k: v for k, v in load_kw.items() if k != "quantization_config"},
              "dequantize" if self.model_key in DEQUANTIZE else "")
        for cls_name in ["AutoModelForCausalLM", "AutoModelForImageTextToText"]:
            try:
                cls = getattr(transformers, cls_name)
                model = cls.from_pretrained(self.hf_id, dtype=torch.bfloat16, **load_kw, **kw)
                break
            except ValueError as e:
                errors.append(cls_name + ": " + str(e)[:300])
        if model is None:
            raise RuntimeError("could not load " + self.hf_id + ": " + " | ".join(errors))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.load_seconds = round(time.time() - start, 1)
        try:
            volume.commit()
        except Exception as e:
            print("volume commit skipped:", e)

        self.info, self.layers = adapter.describe(model, self.config)
        self.vocab = model.get_output_embeddings().weight.shape[0]
        is_digit = torch.zeros(self.vocab, dtype=torch.bool)
        tokens = self.tok.convert_ids_to_tokens(list(range(min(len(self.tok), self.vocab))))
        for i in range(len(tokens)):
            t = tokens[i]
            if t is None:
                continue
            for ch in t:
                if ch.isdigit():
                    is_digit[i] = True
                    break
        self.is_digit = is_digit.cuda()
        with open("/root/builtin_skills.json") as f:
            self.builtin = json.load(f)
        self.tok_cache = {}
        self.base_cache = {}
        print("loaded", self.hf_id, "in", self.load_seconds, "s")

    # ---------------- skills and tokenisation ----------------

    def _skill_sets(self, payload):
        # payload: {"skills": {name: {"hash": h, "items": {"search": [...], "heldout": [...]},
        #                              "control": {"search": [...], "heldout": [...]} or None}}}
        # built-in skills are referenced as {"builtin": true}
        sets = {}
        for name in payload["skills"]:
            spec = payload["skills"][name]
            if spec.get("builtin"):
                spec = dict(self.builtin[name])
                spec["hash"] = "builtin-" + name
            for split in ["search", "heldout"]:
                sets[(name, split)] = (spec["hash"] + ":" + split, spec["items"][split])
                if spec.get("control"):
                    sets[(name + "__control", split)] = (spec["hash"] + ":control:" + split, spec["control"][split])
        return sets

    def _tokenized(self, key, items):
        if key in self.tok_cache:
            return self.tok_cache[key]
        rows = []
        for it in items:
            ctx = self.tok(it["context"], add_special_tokens=True)["input_ids"]
            cont = self.tok(it["continuation"], add_special_tokens=False)["input_ids"]
            ids = (ctx + cont)[:1024]
            start = min(len(ctx), len(ids) - 1)
            rows.append({"ids": ids, "start": start, "digits": bool(it.get("digits_only"))})
        self.tok_cache[key] = rows
        return rows

    def _batches(self, rows, max_elems):
        # sort by length; batch so batch * length * vocab stays under max_elems logits
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
        batch = []
        longest = 0
        for i in order:
            n = len(rows[i]["ids"])
            new_longest = max(longest, n)
            if batch and (len(batch) + 1) * new_longest * self.vocab > max_elems:
                yield batch
                batch = []
                new_longest = n
            batch.append(i)
            longest = new_longest
        if batch:
            yield batch

    def _pad(self, rows, batch):
        import torch

        longest = 0
        for i in batch:
            longest = max(longest, len(rows[i]["ids"]))
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for j in range(len(batch)):
            r = rows[batch[j]]["ids"]
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        return ids.cuda(), mask.cuda()

    def _targets(self, rows, batch, ids):
        # (row j, position p) pairs whose next-token prediction is scored
        import torch

        pos_j = []
        pos_p = []
        for j in range(len(batch)):
            r = rows[batch[j]]
            for t in range(r["start"], len(r["ids"])):
                pos_j.append(j)
                pos_p.append(t - 1)
        pj = torch.tensor(pos_j, dtype=torch.long, device="cuda")
        pp = torch.tensor(pos_p, dtype=torch.long, device="cuda")
        tgt = ids[pj, pp + 1]
        keep = torch.ones_like(tgt, dtype=torch.bool)
        digit_rows = torch.tensor([rows[batch[j]]["digits"] for j in range(len(batch))], device="cuda")
        keep = torch.where(digit_rows[pj], self.is_digit[tgt], keep)
        return pj[keep], pp[keep], tgt[keep]

    def _score_rows(self, rows):
        import torch

        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in self._batches(rows, 1.0e9):
                ids, mask = self._pad(rows, batch)
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.to(ids.device)
                pj, pp, tgt = self._targets(rows, batch, ids)
                if len(tgt) == 0:
                    continue
                lp = torch.log_softmax(logits[pj, pp].float(), dim=-1)
                total += float(lp.gather(1, tgt[:, None]).sum())
                count += int(len(tgt))
                del logits, lp
        if count == 0:
            return {"lp": None, "score": None, "n_tokens": 0}
        mean = total / count
        return {"lp": round(mean, 5), "score": round(math.exp(mean), 5), "n_tokens": count}

    def _score(self, payload, part, only=None):
        sets = self._skill_sets(payload)
        handles = []
        out = {}
        try:
            if part is not None:
                err = self.adapter.check_part(self.info, part)
                if err:
                    raise ValueError("bad part: " + err)
                self.adapter.install(self.info, self.layers, part, handles)
            for key in sets:
                if only is not None and key not in only:
                    continue
                cache_key, items = sets[key]
                if part is None and cache_key in self.base_cache:
                    out[key] = self.base_cache[cache_key]
                    continue
                res = self._score_rows(self._tokenized(cache_key, items))
                if part is None:
                    self.base_cache[cache_key] = res
                out[key] = res
        finally:
            self.adapter.remove_all(handles)
        return out

    def _as_dict(self, scores):
        out = {}
        for key in scores:
            out[key[0] + "|" + key[1]] = scores[key]
        return out

    # ---------------- public methods ----------------

    @modal.method()
    def info_summary(self):
        s = self.adapter.summarize(self.info)
        s["hf_id"] = self.hf_id
        s["revision"] = self.revision
        s["load_seconds"] = self.load_seconds
        s["model_class"] = type(self.model).__name__
        s["vocab"] = int(self.vocab)
        s["layers_detail"] = self.info["layers"]
        return s

    @modal.method()
    def base(self, payload):
        # Base scores plus a chance check: the model must prefer each item's own answer over the
        # answer of a different item (shuffled), by a clear log-prob margin.
        start = time.time()
        scores = self._score(payload, None)
        sets = self._skill_sets(payload)
        margins = {}
        for key in sets:
            name, split = key
            if split != "search":
                continue
            cache_key, items = sets[key]
            shuffled = []
            n = len(items)
            for i in range(n):
                other = items[(i + 1) % n]
                shuffled.append({"context": items[i]["context"], "continuation": other["continuation"],
                                 "digits_only": items[i].get("digits_only", False)})
            res = self._score_rows(self._tokenized(cache_key + ":shuffled", shuffled))
            real = scores[key]["lp"]
            margins[name] = None if real is None or res["lp"] is None else round(real - res["lp"], 4)
        return {"scores": self._as_dict(scores), "margin_vs_shuffled": margins, "seconds": round(time.time() - start, 1)}

    @modal.method()
    def ablate(self, payload, part):
        start = time.time()
        base = self._score(payload, None)
        abl = self._score(payload, part)
        drops = {}
        for key in abl:
            drops[key[0] + "|" + key[1]] = rel_drop(base[key]["score"], abl[key]["score"])
        return {"drops": drops, "ablated": self._as_dict(abl), "seconds": round(time.time() - start, 1)}

    @modal.method()
    def skill_maps(self, payload, routing_items=64):
        # Model-level maps for the compiler, measured only on the payload's skill sets:
        # - layer_lp: mean answer log-prob of each skill when one whole layer is skipped
        # - expert_freq: for layers with fused experts, how often the router picks each expert
        #   (picks per scored token) on each skill's text
        import torch

        start = time.time()
        base = self._score(payload, None)
        layer_lp = {}
        for l in range(self.info["n_layers"]):
            abl = self._score(payload, {"kind": "block", "layers": [l]})
            for key in abl:
                layer_lp.setdefault(key[0] + "|" + key[1], []).append(abl[key]["lp"])
        exp_layers = []
        for e in self.info["layers"]:
            if "experts" in e and e["experts"].get("style") != "module_list":
                exp_layers.append(e["layer"])
        expert_freq = None
        if exp_layers:
            expert_freq = {}
            sets = self._skill_sets(payload)
            for key in sets:
                rows = self._tokenized(sets[key][0], sets[key][1])[:routing_items]
                sums = {}
                state = {"mask": None}
                handles = []
                try:
                    for l in exp_layers:
                        handles.append(self._stat_hook("expert", l, sums, state))
                    with torch.no_grad():
                        for batch in self._batches(rows, 4.0e8):
                            ids, mask = self._pad(rows, batch)
                            m = mask.clone()
                            m[:, 0] = 0
                            state["mask"] = m
                            state["count"] = float(m.sum())
                            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                finally:
                    self.adapter.remove_all(handles)
                per_layer = {}
                for l in sums:
                    freq = (sums[l]["sum"] / max(sums[l]["n"], 1.0)).float().cpu().tolist()
                    per_layer[str(l)] = [round(float(x), 6) for x in freq]
                expert_freq[key[0] + "|" + key[1]] = per_layer
        base_out = {}
        for key in base:
            base_out[key[0] + "|" + key[1]] = {"lp": base[key]["lp"], "score": base[key]["score"]}
        return {"base": base_out, "layer_lp": layer_lp, "expert_freq": expert_freq, "info": self.info_summary_local(),
                "seconds": round(time.time() - start, 1)}

    def info_summary_local(self):
        s = self.adapter.summarize(self.info)
        s["model_class"] = type(self.model).__name__
        s["vocab"] = self.vocab
        s["load_seconds"] = self.load_seconds
        return s

    @modal.method()
    def random_drops(self, payload, part, target, n, seed):
        # drops on the target skill (search split) for n random same-size parts (same layer & kind;
        # blocks: same length, different start). The candidate itself is excluded.
        start = time.time()
        rng = random.Random(seed)
        key = (target, "search")
        base = self._score(payload, None, only=[key])[key]["score"]
        results = []
        tries = 0
        while len(results) < n and tries < n * 20:
            tries += 1
            rp = self._random_like(part, rng)
            if rp is None:
                break
            res = self._score(payload, rp, only=[key])[key]["score"]
            results.append(rel_drop(base, res))
        return {"drops": results, "seconds": round(time.time() - start, 1)}

    def _random_like(self, part, rng):
        if part["kind"] == "block":
            length = len(part["layers"])
            starts = []
            for s in range(0, self.info["n_layers"] - length + 1):
                if s != part["layers"][0]:
                    starts.append(s)
            if not starts:
                return None
            s = rng.choice(starts)
            layers = []
            for l in range(s, s + length):
                layers.append(l)
            return {"kind": "block", "layers": layers}
        entry = self.info["layers"][part["layer"]]
        key = {"mlp_cluster": "mlp", "attn_heads": "attn", "expert": "experts"}[part["kind"]]
        size = entry[key][{"mlp": "width", "attn": "n_heads", "experts": "n_experts"}[key]]
        k = len(part["indices"])
        if k >= size:
            return None
        cand = sorted(part["indices"])
        for _ in range(50):
            pick = sorted(rng.sample(range(size), k))
            if pick != cand:
                return {"kind": part["kind"], "layer": part["layer"], "indices": pick}
        return None

    # ---------------- proposing tools ----------------

    def _scan_rows(self, payload, skill, n_items):
        sets = self._skill_sets(payload)
        cache_key, items = sets[(skill, "search")]
        return self._tokenized(cache_key, items)[:n_items]

    @modal.method()
    def activation_scan(self, payload, kind, skills, layers, n_items=32):
        # mean |activation| per unit per skill (MLP columns, heads; for experts: routing frequency),
        # ignoring padding and the first position (attention sink)
        import torch

        start = time.time()
        layer_list = self._layer_list(kind, layers)
        stats = {}
        for skill in skills:
            rows = self._scan_rows(payload, skill, n_items)
            sums = {}
            state = {"mask": None}
            handles = []
            try:
                for l in layer_list:
                    handles.append(self._stat_hook(kind, l, sums, state))
                with torch.no_grad():
                    for batch in self._batches(rows, 4.0e8):
                        ids, mask = self._pad(rows, batch)
                        m = mask.clone()
                        m[:, 0] = 0
                        state["mask"] = m
                        state["count"] = float(m.sum())
                        self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                self.adapter.remove_all(handles)
            per_layer = {}
            for l in sums:
                per_layer[l] = (sums[l]["sum"] / max(sums[l]["n"], 1.0)).float().cpu()
            stats[skill] = per_layer
        return {"stats": self._rank(stats, "ratio"), "seconds": round(time.time() - start, 1)}

    def _layer_list(self, kind, layers):
        key = {"mlp_cluster": "mlp", "attn_heads": "attn", "expert": "experts"}[kind]
        out = []
        for e in self.info["layers"]:
            if key in e and (not layers or e["layer"] in layers):
                out.append(e["layer"])
        return out

    def _stat_hook(self, kind, l, sums, state):
        import torch

        entry = self.info["layers"][l]
        if kind == "expert":
            mod = self.adapter.get_module(self.layers[l], entry["experts"]["path"])
            n_exp = entry["experts"]["n_experts"]

            def pre(module, args, kwargs):
                idx = args[1] if len(args) >= 3 else kwargs["top_k_index"]
                flat_mask = state["mask"].reshape(-1).bool()
                sel = idx[flat_mask] if idx.shape[0] == flat_mask.shape[0] else idx
                counts = torch.bincount(sel.reshape(-1), minlength=n_exp)[:n_exp].float()
                if l not in sums:
                    sums[l] = {"sum": torch.zeros(n_exp, device=counts.device), "n": 0.0}
                sums[l]["sum"] += counts
                sums[l]["n"] += float(sel.shape[0])
            return mod.register_forward_pre_hook(pre, with_kwargs=True)
        if kind == "attn_heads":
            mod = self.adapter.get_module(self.layers[l], entry["attn"]["path"])
            nh = entry["attn"]["n_heads"]
            hd = entry["attn"]["head_dim"]
        else:
            mod = self.adapter.get_module(self.layers[l], entry["mlp"]["path"])

        def pre_cols(module, args):
            x = args[0]
            m = state["mask"]
            if x.dim() == 2:
                xm = x.float() * m.reshape(-1, 1).float()
            else:
                xm = x.float() * m.unsqueeze(-1).float()
            if kind == "attn_heads":
                xm = xm.reshape(xm.shape[:-1] + (nh, hd)).norm(dim=-1)
            else:
                xm = xm.abs()
            total = xm.reshape(-1, xm.shape[-1]).sum(dim=0)
            if l not in sums:
                sums[l] = {"sum": torch.zeros_like(total), "n": 0.0}
            sums[l]["sum"] += total
            sums[l]["n"] += float(m.sum())
        return mod.register_forward_pre_hook(pre_cols)

    def _rank(self, stats, how, top_k=20):
        # per skill: units whose value is highest relative to the average over the OTHER skills
        import torch

        skills = list(stats.keys())
        out = {}
        for s in skills:
            best = []
            for l in stats[s]:
                mine = stats[s][l]
                others = []
                for o in skills:
                    if o != s and l in stats[o]:
                        others.append(stats[o][l])
                if others:
                    ref = torch.stack(others).mean(dim=0)
                else:
                    ref = torch.zeros_like(mine)
                if how == "ratio":
                    # smoothed ratio: a small floor stops near-zero references (rare experts, dead
                    # units) from producing huge, meaningless ratios
                    eps = 0.05 * float(ref.mean()) + 1e-6
                    score = (mine + eps) / (ref + eps)
                else:
                    score = mine - torch.clamp(ref, min=0)
                k = min(top_k, score.numel())
                vals, idx = torch.topk(score, k)
                for v, i in zip(vals.tolist(), idx.tolist()):
                    best.append({"layer": int(l), "unit": int(i), "score": round(float(v), 4),
                                 "value": round(float(mine[i]), 5)})
            best.sort(key=lambda r: -r["score"])
            out[s] = best[:top_k]
        return out

    @modal.method()
    def attribution_scan(self, payload, kind, skills, layers, n_items=24, token_budget=2048):
        if getattr(self.model.config, "quantization_config", None) is not None and self.model_key not in DEQUANTIZE:
            raise RuntimeError("attribution_scan needs bf16 weights, but this model is loaded quantized (FP8); "
                               "use activation_scan instead")
        # Attribution patching for zero ablation: estimated increase in mean NLL of each skill if a
        # unit is zeroed = -(grad . activation), summed over that unit's positions.
        import torch

        start = time.time()
        layer_list = self._layer_list(kind, layers)
        stats = {}
        for skill in skills:
            rows = self._scan_rows(payload, skill, n_items)
            acc = {}
            n_batches = 0
            for batch in self._batches(rows, token_budget * self.vocab):
                ids, mask = self._pad(rows, batch)
                probes = {}
                handles = []
                try:
                    for l in layer_list:
                        handles.append(self._probe_hook(kind, l, probes))
                    with torch.enable_grad():
                        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.to(ids.device)
                        pj, pp, tgt = self._targets(rows, batch, ids)
                        if len(tgt) == 0:
                            continue
                        lp = torch.log_softmax(logits[pj, pp].float(), dim=-1).gather(1, tgt[:, None])
                        loss = -lp.mean()
                        leaves = []
                        for l in probes:
                            leaves.append(probes[l]["probe"])
                        grads = torch.autograd.grad(loss, leaves, allow_unused=True)
                    for (l, g) in zip(list(probes.keys()), grads):
                        if g is None:
                            continue
                        eff = self._effect(kind, l, probes[l], g)
                        if l not in acc:
                            acc[l] = torch.zeros_like(eff)
                        acc[l] += eff
                    n_batches += 1
                finally:
                    self.adapter.remove_all(handles)
                    del probes
            per_layer = {}
            for l in acc:
                per_layer[l] = (acc[l] / max(n_batches, 1)).float().cpu()
            stats[skill] = per_layer
        return {"stats": self._rank(stats, "selective"), "seconds": round(time.time() - start, 1)}

    def _probe_hook(self, kind, l, probes):
        import torch

        entry = self.info["layers"][l]
        if kind == "expert":
            mod = self.adapter.get_module(self.layers[l], entry["experts"]["path"])

            def pre(module, args, kwargs):
                args = list(args)
                if len(args) >= 3:
                    idx, w = args[1], args[2]
                else:
                    idx, w = kwargs["top_k_index"], kwargs["top_k_weights"]
                probe = torch.zeros_like(w, requires_grad=True)
                probes[l] = {"probe": probe, "x": w.detach(), "idx": idx.detach()}
                w2 = w + probe
                if len(args) >= 3:
                    args[2] = w2
                else:
                    kwargs["top_k_weights"] = w2
                return tuple(args), kwargs
            return mod.register_forward_pre_hook(pre, with_kwargs=True)
        if kind == "attn_heads":
            mod = self.adapter.get_module(self.layers[l], entry["attn"]["path"])
        else:
            mod = self.adapter.get_module(self.layers[l], entry["mlp"]["path"])

        def pre_cols(module, args):
            x = args[0]
            probe = torch.zeros_like(x, requires_grad=True)
            probes[l] = {"probe": probe, "x": x.detach()}
            return (x + probe,) + tuple(args[1:])
        return mod.register_forward_pre_hook(pre_cols)

    def _effect(self, kind, l, rec, g):
        import torch

        prod = -(g.float() * rec["x"].float())  # first-order change in loss when the value goes to 0
        if kind == "expert":
            n_exp = self.info["layers"][l]["experts"]["n_experts"]
            out = torch.zeros(n_exp, device=prod.device)
            out.index_add_(0, rec["idx"].reshape(-1), prod.reshape(-1))
            return out
        flat = prod.reshape(-1, prod.shape[-1]).sum(dim=0)
        if kind == "attn_heads":
            entry = self.info["layers"][l]["attn"]
            flat = flat.reshape(entry["n_heads"], entry["head_dim"]).sum(dim=-1)
        return flat

    # ---------------- optional confirmation ----------------

    @modal.method()
    def confirm_math(self, part, n_items=40, max_new_tokens=256):
        import re
        import torch

        start = time.time()
        items = self.builtin["math"]["items"]["search"][:n_items]
        prompts = []
        for it in items:
            prompts.append(it["context"])

        def accuracy():
            correct = 0
            old_side = self.tok.padding_side
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            try:
                for b in range(0, len(prompts), 8):
                    enc = self.tok(prompts[b:b + 8], return_tensors="pt", padding=True).to("cuda")
                    with torch.no_grad():
                        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
                    for j in range(out.shape[0]):
                        text = self.tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        text = text.split("Question:")[0]
                        m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text)
                        nums = re.findall(r"-?[\d,]*\.?\d+", text)
                        got = m.group(1) if m else (nums[-1] if nums else "")
                        got = got.replace(",", "").rstrip(".")
                        if got == items[b + j]["answer"]:
                            correct += 1
            finally:
                self.tok.padding_side = old_side
            return correct / len(prompts)

        if "base_gen" not in self.__dict__:
            self.base_gen = accuracy()  # cached per container
        base = self.base_gen
        handles = []
        try:
            self.adapter.install(self.info, self.layers, part, handles)
            abl = accuracy()
        finally:
            self.adapter.remove_all(handles)
        return {"base_accuracy": round(base, 4), "ablated_accuracy": round(abl, 4),
                "drop": rel_drop(base, abl), "n": len(prompts), "seconds": round(time.time() - start, 1)}


COMMON = {"image": image, "volumes": {"/models": volume}, "secrets": [modal.Secret.from_name("hf-token")],
          "timeout": 3600, "startup_timeout": 3600, "scaledown_window": 600}


@app.cls(gpu=["L4", "A10G", "L40S"], **COMMON)  # L4 capacity runs short
class EngineL4(EngineBase):
    # modal.parameter fields must be declared on each deployed class itself (inherited ones are ignored,
    # which made every container crash at startup with "takes no arguments")
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()



@app.cls(gpu=["L40S", "A100-40GB"], **COMMON)
class EngineL40S(EngineBase):
    # modal.parameter fields must be declared on each deployed class itself (inherited ones are ignored,
    # which made every container crash at startup with "takes no arguments")
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()



@app.cls(gpu=["H100", "H200", "A100-80GB"], max_containers=1, **COMMON)  # cost cap: parallel agents share one big GPU
class EngineH100(EngineBase):
    # modal.parameter fields must be declared on each deployed class itself (inherited ones are ignored,
    # which made every container crash at startup with "takes no arguments")
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()



@app.cls(gpu=["H200", "B200"], max_containers=1, **COMMON)  # no H100: 34-80B bf16 does not fit 80 GB  # cost cap: parallel agents share one big GPU
class EngineH200(EngineBase):
    # modal.parameter fields must be declared on each deployed class itself (inherited ones are ignored,
    # which made every container crash at startup with "takes no arguments")
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()


COMMON_MULTI = dict(COMMON, timeout=3 * 3600)
# Multi-GPU tiers for frontier models. Layers are spread with device_map="auto". Cost cap: one container each.


@app.cls(gpu=["H200:2", "B200:2"], max_containers=1, **COMMON_MULTI)
class EngineH200x2(EngineBase):
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()


@app.cls(gpu=["H200:4", "B200:4"], max_containers=1, **COMMON_MULTI)
class EngineH200x4(EngineBase):
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()


@app.cls(gpu=["H200:8", "B200:8"], max_containers=1, **COMMON_MULTI)
class EngineH200x8(EngineBase):
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()


@app.cls(gpu="B200:8", max_containers=1, **COMMON_MULTI)
class EngineB200x8(EngineBase):
    # 1T-class models (about 1.1 TB of FP8 weights): only 8 x B200 has room
    model_key: str = modal.parameter()
    hf_id: str = modal.parameter()
    revision: str = modal.parameter()
