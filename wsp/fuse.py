"""Skill fusion: write blocks from several specialists into one host, with token gates.

compile_fusion builds a Program's grafts the same way wsp.compiler does
(wsp.assemble.build_graft: ridge fit, sink, trained stitch, part LoRA),
but each graft can have its own align texts, so the math stitch trains on
math text and the code stitch on code text. Then it attaches a token gate
to each graft (wsp.gate), and one heal pass trains the glue, the part
LoRAs, and the gates together on the mixed heal texts. The gates train at
their own learning rate.

The output directory is a normal compiled directory (tokenizer,
program.json, grafts.json, graft_weights.safetensors,
compile_report.json) plus gates.json and gate_weights.safetensors.
wsp.fuse.load reads it back with the gates; wsp.load reads it back
without them.
"""

import json
import os
import time

import torch


def compile_fusion(program, catalog, output_dir, align_texts, heal_texts, gates=True, gate_lr=1e-2, gate_bias=0.0,
                   lora_lr=1e-4,
                   device=None, cache_dir=None, log_every=10):
    """Build, gate, heal, and save the program's grafts. Return the compile report.

    align_texts: one list of texts for every graft, or a list with one list
        (or None) per graft, in the order of program.grafts.
    heal_texts: training texts for the heal pass. program.heal_spec gives
        steps, lr, batch_size, and seq_len; without a heal spec nothing
        trains.
    gates: True puts a wsp.gate.TokenGate on each graft.
    """
    start = time.time()
    model, tokenizer, report, metas = fuse_model(program, catalog, align_texts, heal_texts, gates=gates,
                                                 gate_lr=gate_lr, gate_bias=gate_bias, device=device,
                                                 cache_dir=cache_dir, log_every=log_every, lora_lr=lora_lr)
    report["seconds"] = round(time.time() - start, 1)
    save_fusion(model, tokenizer, program, metas, report, output_dir)
    return report


def fuse_model(program, catalog, align_texts, heal_texts, gates=True, gate_lr=1e-2, gate_bias=0.0, device=None,
               cache_dir=None, log_every=10, lora_lr=1e-4):
    """Build, gate, and heal the program's grafts on its host. Return (model, tokenizer, report, metas)."""
    import transformers

    from wsp.assemble import build_graft
    from wsp.gate import attach_gates, gate_meta, gate_parameters
    from wsp.graft import attach_grafts, freeze_all_but_glue, glue_parameters, part_lora_parameters

    if len(program.grafts) == 0:
        raise ValueError("compile_fusion needs a program with at least one graft")
    if len(program.taps) > 0:
        raise ValueError("compile_fusion takes grafts and an optional host LoRA; compile taps with wsp.compiler")
    if program.masks is not None or len(program.task_vectors) > 0:
        raise ValueError("compile_fusion takes grafts only; masks and task vectors go through wsp.compiler")
    per_graft = texts_per_graft(align_texts, len(program.grafts))
    torch.manual_seed(program.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(program.host, dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
    model.to(device)
    model.eval()
    report = {"host": program.host, "fusion": True, "grafts": [], "heal": None, "gates": None, "lora": None}

    # An optional LoRA on the host lets the base learn to READ the written-in
    # skills (run 14: with only seam glue trained, the base could not use a
    # lossless donor block). It goes on before the grafts, so its targets are
    # the host's own layers, and it trains in the same heal.
    host_lora = []
    if program.lora_spec is not None:
        from wsp.compiler import add_lora

        host_lora = add_lora(program.lora_spec, model, report)

    placements = []
    metas = []
    for number, spec in enumerate(program.grafts):
        graft_start = time.time()
        host_layer, graft, meta = build_graft(spec, catalog, model, tokenizer, per_graft[number], device, cache_dir)
        placements.append((host_layer, graft))
        metas.append(meta)
        # These go in the report only, so grafts.json keeps the compiler's keys.
        report["grafts"].append(dict(meta))
        report["grafts"][-1]["build_seconds"] = round(time.time() - graft_start, 1)
        report["grafts"][-1]["align_texts"] = 0
        if per_graft[number] is not None:
            report["grafts"][-1]["align_texts"] = len(per_graft[number])
    attach_grafts(model, placements)
    model.wsp_grafts.to(device)

    if gates:
        attach_gates(model, init_bias=gate_bias)

    if program.heal_spec is not None:
        freeze_all_but_glue(model)
        glue = glue_parameters(model)
        for param in part_lora_parameters(model):
            glue.append(param)
        gate_params = gate_parameters(model)
        for param in gate_params:
            param.requires_grad_(True)
        spec = program.heal_spec
        groups = [{"params": glue, "lr": spec.lr}]
        if len(gate_params) > 0:
            groups.append({"params": gate_params, "lr": gate_lr})
        if len(host_lora) > 0:
            for param in host_lora:
                param.requires_grad_(True)
            groups.append({"params": host_lora, "lr": lora_lr})
        heal_start = time.time()
        losses = heal_groups(model, tokenizer, heal_texts, groups, steps=spec.steps, batch_size=spec.batch_size,
                             seq_len=spec.seq_len, seed=program.seed, log_every=log_every)
        trainable = 0
        for group in groups:
            for param in group["params"]:
                trainable += param.numel()
        part_lora = 0
        for param in part_lora_parameters(model):
            part_lora += param.numel()
        gate_count = 0
        for param in gate_params:
            gate_count += param.numel()
        report["heal"] = {"spec": spec.to_dict(), "losses": losses, "seconds": round(time.time() - heal_start, 1),
                          "trainable_params": trainable, "part_lora_params": part_lora, "gate_params": gate_count,
                          "gate_lr": gate_lr, "texts": len(heal_texts)}

    for param in model.parameters():
        param.requires_grad_(False)
    if gates:
        report["gates"] = gate_meta(model)
    report["params_total"] = param_count(model)
    return model, tokenizer, report, metas


def save_fusion(model, tokenizer, program, metas, report, output_dir):
    """Save a fused model: tokenizer, program.json, grafts, gates (when on), and compile_report.json."""
    from wsp.compiler import remove_stale_outputs, save_grafts
    from wsp.gate import save_gates

    os.makedirs(output_dir, exist_ok=True)
    remove_stale_outputs(output_dir)
    for name in ["gates.json", "gate_weights.safetensors"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
    tokenizer.save_pretrained(output_dir)
    save_grafts(model, output_dir, metas)
    if report.get("lora") is not None:
        from wsp.compiler import save_lora

        save_lora(model, output_dir, report)
    if getattr(model, "wsp_gates", None) is not None:
        save_gates(model, output_dir)
    with open(os.path.join(output_dir, "program.json"), "w") as f:
        json.dump(program.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def texts_per_graft(align_texts, count):
    """Return a list with one align text list (or None) per graft."""
    if align_texts is None:
        out = []
        for _ in range(count):
            out.append(None)
        return out
    nested = len(align_texts) > 0
    for entry in align_texts:
        if entry is not None and not isinstance(entry, (list, tuple)):
            nested = False
    if not nested:
        out = []
        for _ in range(count):
            out.append(list(align_texts))
        return out
    if len(align_texts) != count:
        raise ValueError(f"align_texts has {len(align_texts)} lists for {count} grafts")
    out = []
    for entry in align_texts:
        if entry is None:
            out.append(None)
        else:
            out.append(list(entry))
    return out


def heal_groups(model, tokenizer, texts, groups, steps=300, batch_size=8, seq_len=384, seed=0, log_every=10):
    """The wsp.heal loop with optimizer groups. Return the per-step losses.

    groups: AdamW parameter groups, each a dict with "params" and "lr".
    With one group this is the same loop as wsp.heal.heal: the same batch
    order for a seed, the full-sequence LM loss, gradient clipping at 1.0
    over every trained parameter, and bf16 autocast on CUDA.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    every = []
    for group in groups:
        for param in group["params"]:
            every.append(param)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    model.train()
    losses = []
    order = []
    for step in range(steps):
        batch = []
        while len(batch) < batch_size:
            if len(order) == 0:
                order = torch.randperm(len(texts), generator=generator).tolist()
            batch.append(texts[order.pop()])

        enc = tokenizer(batch, truncation=True, max_length=seq_len, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(every, 1.0)
        optimizer.step()

        loss = float(out.loss.detach())
        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"fusion heal step {step + 1}/{steps} loss {loss:.4f}")

    model.eval()
    return losses


def param_count(model):
    """Count every distinct parameter of a model once (tied weights count once)."""
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total


def load(path, device="cpu", dtype=torch.bfloat16):
    """Load a fused model: wsp.load, then the gates when the directory has them. Return (model, tokenizer)."""
    import wsp
    from wsp.gate import load_gates

    model, tokenizer = wsp.load(path, device=device, dtype=dtype, gates=False)
    load_gates(model, path)
    model.eval()
    return model, tokenizer
