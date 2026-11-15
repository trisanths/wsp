"""Correctness audit on real models, with known right answers.

1. Fetch: range-fetched donor tensors equal from_pretrained tensors.
2. Parts: a block part built from fetched tensors equals the donor's own
   layers on real text; an mlp_cluster with every neuron equals the donor
   MLP on real states.
3. Stitch calibration. Host == donor (Qwen2.5-Math-1.5B), host layer 7 to
   donor layer 8: the right map is the identity, so R2 should be about 1
   and the stitch penalty about 0. Any big penalty here is a bug.
4. Ridge settings: the current fit scales its regularizer by mean input
   variance, which a few massive-activation dims inflate. Compare the
   current setting, tiny regularization, and standardized inputs, on the
   identity pair, on Qwen2.5-1.5B base to Math (same init), and on
   Qwen2.5-0.5B-Instruct to Math (round 4's pair).

    MODAL_PROFILE=trisanth modal run modal_audit.py
"""

import json

import modal

app = modal.App("wsp-audit")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "transformers==5.23.0", "accelerate", "huggingface_hub", "requests",
                 "safetensors", "zstandard", "numpy", "datasets")
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

DONOR = "Qwen/Qwen2.5-Math-1.5B"


def ridge_variants(x, y):
    """Return {name: (w, r2_dim)} for several ridge settings. w maps x to y."""
    import torch

    from wsp.align import ridge_fit

    out = {}
    w, r2, r2_dim = ridge_fit(x, y)
    out["current_lam1e-2_meanvar"] = (w, r2, r2_dim)
    w, r2, r2_dim = ridge_fit(x, y, lam=1e-6)
    out["lam1e-6"] = (w, r2, r2_dim)
    # Standardize inputs per dim, fit with small ridge, fold the scaling back into w.
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp(min=1e-6)
    xs = (x - mean) / std
    w_s, r2, r2_dim = ridge_fit(xs, y, lam=1e-4)
    w_fold = w_s / std[None, :]
    out["standardized_lam1e-4_nobias"] = (w_fold, r2, r2_dim)
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=98304, timeout=3600)
def audit():
    import torch
    import transformers

    from wsp.align import collect_pairs, layer_input_capture
    from wsp.data import format_gsm8k, gsm8k_splits
    from wsp.fetch import fetch_config, fetch_tensors
    from wsp.parts import (block_tensor_names, build_block_part, build_mlp_cluster, find_decoder_layers,
                           find_layer_modules, mlp_tensor_names)

    results = {}
    search, heal_items = gsm8k_splits(seed=0)
    texts = []
    for item in heal_items[:400]:
        texts.append(format_gsm8k(item))

    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float32).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
    config = fetch_config(DONOR)
    state = donor.state_dict()

    # 1. Fetch equality.
    names = block_tensor_names(config, 8, 19)
    fetched = fetch_tensors(DONOR, names, cache_dir="/vol/tensor_cache")
    worst = 0.0
    for name in names:
        diff = (fetched[name].float().cuda() - state[name].float()).abs().max().item()
        worst = max(worst, diff)
    results["fetch_max_abs_diff_vs_bf16_checkpoint"] = worst

    # 2. Block part and mlp cluster vs the donor's own layers on real text.
    _, layers = find_decoder_layers(donor, 0)
    store = {}

    def grab_in(module, args, kwargs):
        store["in"] = (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach()

    def grab_out(module, args, output):
        store["out"] = (output[0] if isinstance(output, tuple) else output).detach()

    h1 = layers[8].register_forward_pre_hook(grab_in, with_kwargs=True)
    h2 = layers[19].register_forward_hook(grab_out)
    ids = tokenizer(texts[0], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        donor(ids)
    h1.remove()
    h2.remove()
    part = build_block_part(config, 8, 19, fetched).to("cuda").float()
    with torch.no_grad():
        delta = part(store["in"])
    expected = store["out"] - store["in"]
    results["block_part_max_abs_diff"] = (delta - expected).abs().max().item()
    results["block_part_ref_scale"] = expected.abs().max().item()

    mlp_names = mlp_tensor_names(config, 14)
    mlp_tensors = fetch_tensors(DONOR, mlp_names, cache_dir="/vol/tensor_cache")
    cluster = build_mlp_cluster(config, 14, torch.arange(config.intermediate_size), mlp_tensors).to("cuda").float()
    found = find_layer_modules(config, 14)
    norm = donor.get_submodule(found["pre_norm_prefix"])
    mlp = donor.get_submodule(found["mlp_prefix"])
    x = torch.randn(1, 8, config.hidden_size, device="cuda") * 3
    with torch.no_grad():
        results["mlp_cluster_max_abs_diff"] = (cluster(x) - mlp(norm(x))).abs().max().item()

    # 3 and 4. Stitch calibration with several ridge settings.
    def stitch_eval(host_id, k, l, label):
        host = donor if host_id == DONOR else transformers.AutoModelForCausalLM.from_pretrained(
            host_id, dtype=torch.float32).to("cuda").eval()
        _, host_layers = find_decoder_layers(host, 0)
        capture = layer_input_capture(layers[l])
        x, y = collect_pairs(host, tokenizer, k, donor, tokenizer, capture, texts[:250])
        variants = ridge_variants(x, y)

        def donor_loss(w, sink=None):
            total = 0.0
            count = 0
            for text in texts[250:400]:
                ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
                handles = []
                if w is not None:
                    hold = {}

                    def host_hook(module, args, output):
                        hold["h"] = (output[0] if isinstance(output, tuple) else output).detach()

                    hh = host_layers[k].register_forward_hook(host_hook)
                    with torch.no_grad():
                        host(ids)
                    hh.remove()

                    native_in = {}
                    if sink == "oracle":
                        def grab_native(module, args, kwargs):
                            native_in["h"] = (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach()
                        gn = layers[l].register_forward_pre_hook(grab_native, with_kwargs=True)
                        with torch.no_grad():
                            donor(ids)
                        gn.remove()

                    def pre(module, args, kwargs):
                        mapped = hold["h"].float() @ w.T.cuda().float()
                        if sink == "oracle":
                            mapped = mapped.clone()
                            mapped[:, 0] = native_in["h"][:, 0].float()
                        elif sink == "mean":
                            mapped = mapped.clone()
                            mapped[:, 0] = mean_sink
                        if len(args) > 0:
                            return (mapped,) + tuple(args[1:]), kwargs
                        kwargs["hidden_states"] = mapped
                        return args, kwargs

                    handles.append(layers[l].register_forward_pre_hook(pre, with_kwargs=True))
                with torch.no_grad():
                    loss = donor(ids, labels=ids).loss.item()
                for handle in handles:
                    handle.remove()
                total += loss * (ids.shape[1] - 1)
                count += ids.shape[1] - 1
            return total / count

        # Donor's own first-token state (the attention sink), averaged over the fit texts.
        sink_sum = None
        sink_count = 0
        for text in texts[:250]:
            ids0 = tokenizer(text, return_tensors="pt").input_ids.cuda()
            got = {}

            def grab0(module, args, kwargs):
                got["h"] = (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach()

            gh = layers[l].register_forward_pre_hook(grab0, with_kwargs=True)
            with torch.no_grad():
                donor(ids0)
            gh.remove()
            if sink_sum is None:
                sink_sum = got["h"][0, 0].float().clone()
            else:
                sink_sum += got["h"][0, 0].float()
            sink_count += 1
        mean_sink = sink_sum / sink_count
        native = donor_loss(None)
        row = {"pair": label, "host": host_id, "host_layer": k, "donor_layer": l, "pairs": x.shape[0],
               "native_loss": native, "variants": {}}
        if host_id == DONOR and k == l - 1:
            row["variants"]["exact_identity"] = {"stitched_loss": donor_loss(torch.eye(y.shape[1]))}
        for name, (w, r2, r2_dim) in variants.items():
            if name == "standardized_lam1e-4_nobias":
                continue
            for sink in [None, "mean", "oracle"]:
                stitched = donor_loss(w, sink)
                label_v = name if sink is None else f"{name}+sink_{sink}"
                row["variants"][label_v] = {"r2": r2, "r2_dim": r2_dim, "stitched_loss": stitched,
                                            "penalty": stitched - native}
        print(json.dumps(row))
        if host is not donor:
            del host
            torch.cuda.empty_cache()
        return row

    results = {}
    results["stitch"] = [
        stitch_eval(DONOR, 7, 8, "identity: Math L7 -> Math L8"),
        stitch_eval("Qwen/Qwen2.5-1.5B", 7, 8, "same init: base L7 -> Math L8"),
        stitch_eval("Qwen/Qwen2.5-0.5B-Instruct", 7, 8, "round 4 pair: 0.5B L7 -> Math L8"),
    ]
    return results


@app.local_entrypoint()
def main():
    results = audit.remote()
    print(json.dumps(results, indent=2))
    with open("results/audit_rows.json", "w") as f:
        json.dump(results, f, indent=2)
