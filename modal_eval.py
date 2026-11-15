"""Run the eval harness on Modal with vLLM (Person B).

    MODAL_PROFILE=trisanth modal run modal_eval.py::first_numbers \
        --parents "mask_mmlupro_biology_llama1b mask_mmlupro_chemistry_llama1b" \
        --programs "mask_mmlupro_biology_llama1b | mask_mmlupro_chemistry_llama1b"

    MODAL_PROFILE=trisanth modal run modal_eval.py::eval_program \
        --program-path program.json --split search --tasks "mmlupro:biology mmlupro:chemistry" --namespace swarm

    MODAL_PROFILE=trisanth modal run modal_eval.py::interp --max-evals 80

    MODAL_PROFILE=trisanth modal run modal_eval.py::gate --seeds "0 1 2"

    MODAL_PROFILE=trisanth modal run modal_eval.py::extract --model Qwen/Qwen2.5-Math-1.5B --validate 2
    MODAL_PROFILE=trisanth modal run modal_eval.py::extract --model Qwen/Qwen3-30B-A3B --gpu H100 --kinds expert

Budget: $250 of Modal per person. An L4 is about $0.80/hour, an H100 about
$4/hour. Extraction on a 1.5B donor takes minutes; a 30B-class donor fits on
one H100 in bf16 (about 60 GB) for forward passes.

Shared state (generation cache, budget counter, held-out ledger,
experiments.jsonl, interp log, compiled probe models) lives on the
`wsp-models` volume under /vol/eval, so every caller shares one budget and
one cache. Results are also copied to outputs/ locally.
"""
import json
import os

import modal

app = modal.App("wsp-eval")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.11.0",
        "transformers==4.57.1",
        "accelerate",
        "huggingface_hub",
        "requests",
        "datasets",
        "matplotlib",
        "safetensors",
        "zstandard",
        "numpy",
    )
    .env({"HF_HOME": "/vol/hf", "WSP_ROOT": "/vol/eval", "WSP_CATALOG": "/vol/eval/catalog.jsonl",
          "WSP_BACKEND": "vllm", "WSP_HF_BATCH_SIZE": "64"})
    .add_local_python_source("wsp", "wsp_eval")
    .add_local_file("catalog.jsonl", "/root/catalog.jsonl")
    .add_local_dir("scripts", "/root/scripts")
)


def _prepare_catalog():
    """Keep a catalog copy on the volume so interp signatures persist; refresh other fields from the repo."""
    import shutil
    os.makedirs("/vol/eval", exist_ok=True)
    if not os.path.exists("/vol/eval/catalog.jsonl"):
        shutil.copy("/root/catalog.jsonl", "/vol/eval/catalog.jsonl")
        return
    signatures = {}
    with open("/vol/eval/catalog.jsonl") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if "signature" in record:
                    signatures[record["id"]] = record["signature"]
    lines = []
    with open("/root/catalog.jsonl") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record["id"] in signatures and not record.get("signature"):
                    record["signature"] = signatures[record["id"]]
                lines.append(json.dumps(record))
    with open("/vol/eval/catalog.jsonl", "w") as f:
        f.write("\n".join(lines) + "\n")


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=3600)
def first_numbers_remote(parents, tasks, programs):
    import subprocess
    import sys
    _prepare_catalog()
    command = [sys.executable, "/root/scripts/eval_first_numbers.py", "--parents"] + parents
    if tasks:
        command += ["--tasks"] + tasks
    if programs:
        command += ["--programs"] + programs
    completed = subprocess.run(command, capture_output=True, text=True, cwd="/vol/eval")
    volume.commit()
    with open("/vol/eval/work/reference_search.json") as f:
        reference = json.load(f)
    return {"stdout": completed.stdout, "stderr": completed.stderr[-4000:], "returncode": completed.returncode,
            "reference": reference}


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=3600)
def eval_remote(program_dict, split, tasks, namespace):
    from wsp import Program
    from wsp_eval import components, harness
    _prepare_catalog()
    program = Program.from_dict(program_dict)
    model_dir = components.build_model(program)
    result = harness.evaluate(model_dir, split, tasks, namespace=namespace, note=repr(program))
    volume.commit()
    return result


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=6 * 3600)
def interp_remote(max_evals, band_size, ids, own_task_only=True, localize=False):
    from wsp_eval.interp_agent import InterpAgent
    _prepare_catalog()
    os.environ["WSP_DELETE_PROBE_MODELS"] = "1"
    agent = InterpAgent(max_evals=max_evals, band_size=band_size, ids=ids, own_task_only=own_task_only,
                        localize=localize)
    summary = agent.run()
    volume.commit()
    with open("/vol/eval/catalog.jsonl") as f:
        catalog_text = f.read()
    return {"summary": summary, "catalog": catalog_text}


def _extract_body(model, task, kinds, validate, top_layers):
    import subprocess
    import sys
    out_dir = "/vol/eval/extracted/" + model.replace("/", "__")
    command = [sys.executable, "/root/scripts/extract_donor.py", "--model", model, "--task", task,
               "--validate", str(validate), "--top-layers", str(top_layers), "--out", out_dir]
    if kinds:
        command += ["--kinds"] + kinds
    completed = subprocess.run(command, capture_output=True, text=True, cwd="/vol/eval")
    volume.commit()
    files = {}
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            with open(os.path.join(out_dir, name), "rb") as f:
                files[name] = f.read()
    return {"stdout": completed.stdout, "stderr": completed.stderr[-4000:], "returncode": completed.returncode,
            "files": files}


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=32768, timeout=3 * 3600)
def extract_l4(model, task, kinds, validate, top_layers):
    return _extract_body(model, task, kinds, validate, top_layers)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=3 * 3600)
def extract_h100(model, task, kinds, validate, top_layers):
    return _extract_body(model, task, kinds, validate, top_layers)


@app.function(image=image, volumes={"/vol": volume}, gpu="L40S", memory=65536, timeout=4 * 3600)
def gate_remote(seeds, donor_layer, neurons, heal_steps):
    from wsp_eval.gate import run_gate
    _prepare_catalog()
    summary = run_gate(seeds=tuple(seeds), donor_layer=donor_layer, neurons=neurons, heal_steps=heal_steps)
    volume.commit()
    return json.loads(json.dumps(summary, default=str))


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=163840, timeout=8 * 3600)
def part_gate_remote(donor, part_id, seeds, heal_steps):
    from wsp_eval.gate import run_part_gate
    _prepare_catalog()
    summary = run_part_gate(donor, part_id=part_id or None, seeds=tuple(seeds), heal_steps=heal_steps)
    volume.commit()
    return summary


@app.function(image=image, volumes={"/vol": volume}, gpu="L40S", memory=65536, timeout=5 * 3600)
def final_arm_remote(donor, part_id, arm, seed):
    from wsp_eval import final
    root = final.arm_root(donor, arm, seed)
    os.makedirs(root, exist_ok=True)
    os.environ["WSP_ROOT"] = root
    output = final.run_arm(donor, part_id, arm, seed)
    volume.commit()
    return output


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=8192, timeout=600)
def final_merge_remote(donor):
    from wsp_eval import final
    volume.reload()
    added = final.merge_ledgers(donor)
    volume.commit()
    return added


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def stitch_graft_remote(arm, seed, budget=None):
    from wsp_eval import stitch_graft
    root = stitch_graft.arm_root(arm)
    os.makedirs(root, exist_ok=True)
    os.environ["WSP_ROOT"] = root
    row = stitch_graft.run_arm(arm, seed, budget=budget)
    volume.commit()
    return row


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def exit_stitch_remote(arm, seed):
    from wsp_eval import exit_stitch
    root = exit_stitch.arm_root(arm)
    os.makedirs(root, exist_ok=True)
    os.environ["WSP_ROOT"] = root
    row = exit_stitch.run_arm(arm, seed)
    volume.commit()
    return row


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=3600)
def mask_signatures_remote():
    from wsp_eval.signatures import structural_signatures
    _prepare_catalog()
    count = structural_signatures("/vol/eval/catalog.jsonl")
    volume.commit()
    with open("/vol/eval/catalog.jsonl") as f:
        return {"count": count, "catalog": f.read()}


@app.function(image=image, volumes={"/vol": volume}, gpu="L40S", memory=65536, timeout=4 * 3600)
def base_scores_remote(host, tasks, split="search"):
    from wsp_eval import harness
    _prepare_catalog()
    result = harness.evaluate(host, split, tasks, namespace="reference", backend="hf",
                              note="host base scores (" + split + ")")
    volume.commit()
    return result


def _save(name, data):
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", path)


@app.local_entrypoint()
def first_numbers(parents: str, programs: str = "", tasks: str = ""):
    program_list = []
    for text in programs.split(";"):
        if text.strip():
            program_list.append(text.strip())
    result = first_numbers_remote.remote(parents.split(), tasks.split(), program_list)
    print(result["stdout"])
    if result["returncode"] != 0:
        print(result["stderr"])
    _save("first_numbers.json", result)


@app.local_entrypoint()
def eval_program(program_path: str, split: str = "search", tasks: str = "", namespace: str = "default"):
    with open(program_path) as f:
        program_dict = json.load(f)
    result = eval_remote.remote(program_dict, split, tasks.split(), namespace)
    print(json.dumps(result, indent=2))
    _save("eval_" + os.path.basename(program_path), result)


@app.local_entrypoint()
def interp(max_evals: int = 80, band_size: int = 4, ids: str = "", localize: bool = False):
    id_list = None
    if ids.strip():
        id_list = ids.split()
    result = interp_remote.remote(max_evals, band_size, id_list, True, localize)
    print(result["summary"])
    with open("outputs/catalog_with_signatures.jsonl", "w") as f:
        f.write(result["catalog"])
    print("signatures written to outputs/catalog_with_signatures.jsonl (Person A merges into catalog.jsonl)")


@app.local_entrypoint()
def extract(model: str, task: str = "gsm8k", kinds: str = "", validate: int = 0, top_layers: int = 3,
            gpu: str = "L4"):
    function = extract_l4
    if gpu.upper() == "H100":
        function = extract_h100
    result = function.remote(model, task, kinds.split(), validate, top_layers)
    print(result["stdout"])
    if result["returncode"] != 0:
        print(result["stderr"])
    out_dir = os.path.join("outputs", "extracted", model.replace("/", "__"))
    os.makedirs(out_dir, exist_ok=True)
    for name, content in result["files"].items():
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(content)
    print("index files and records.jsonl copied to", out_dir)


@app.local_entrypoint()
def gate(seeds: str = "0 1 2", donor_layer: int = 10, neurons: int = 512, heal_steps: int = 300):
    """Hour-8 gate: host, donor, graft, random-part control, matched LoRA, scored by eval()."""
    seed_list = []
    for text in seeds.split():
        seed_list.append(int(text))
    summary = gate_remote.remote(seed_list, donor_layer, neurons, heal_steps)
    for row in summary["rows"]:
        print("%-7s seed %-4s score %.3f  fit %s  [%s]" % (row["arm"], row["seed"], row["score"],
                                                          row["fit_r2"], row["label"]))
    from wsp_eval import claims
    print(claims.format_verdict(summary["verdict"]))
    _save("gate_summary.json", summary)


@app.local_entrypoint()
def mask_signatures():
    """Structural signatures (counts, per-layer deletions, overlaps) for every mask. CPU only."""
    import json as json_module
    result = mask_signatures_remote.remote()
    signatures = {}
    for line in result["catalog"].splitlines():
        if line.strip():
            record = json_module.loads(line)
            if "signature" in record:
                signatures[record["id"]] = record["signature"]
    # Merge only the signature field into the local catalog; every other field stays as Person A wrote it.
    lines = []
    with open("catalog.jsonl") as f:
        for line in f:
            if line.strip():
                record = json_module.loads(line)
                if record["id"] in signatures:
                    record["signature"] = signatures[record["id"]]
                lines.append(json_module.dumps(record))
    with open("catalog.jsonl", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("signatures merged into catalog.jsonl for", result["count"], "masks")


@app.local_entrypoint()
def base_scores(host: str = "unsloth/Llama-3.2-1B-Instruct", tasks: str = "gsm8k if_guard", split: str = "search"):
    """Host scores in the form wsp/swarm config base_scores expects.

    split="heldout" runs once per host and task (the held-out ledger refuses a rerun).
    The host never changes, so its held-out score can be taken before the finalists.
    """
    result = base_scores_remote.remote(host, tasks.split(), split)
    table = {split: {}}
    for task, entry in result["tasks"].items():
        table[split][task] = entry["score"]
        print("%-10s %.3f  (n=%d) [%s]" % (task, entry["score"], entry["n"], result["label"]))
    print('config "base_scores":', json.dumps(table))
    _save("base_scores_" + split + ".json", table)


@app.local_entrypoint()
def part_gate(donor: str, part_id: str = "", seeds: str = "0 1 2", heal_steps: int = 300):
    """Hour-16 control: an extracted part of `donor` against a random same-size part, 3 seeds (search split)."""
    seed_list = []
    for value in seeds.split():
        seed_list.append(int(value))
    summary = part_gate_remote.remote(donor, part_id, seed_list, heal_steps)
    for row in summary["rows"]:
        print(row["arm"], "seed", row["seed"], "score", row["score"])
    _save("part_gate_" + donor.replace("/", "__") + ".json", summary)


@app.local_entrypoint()
def final(donor: str = "microsoft/phi-4", part_id: str = "mlp_cluster_phi4_l16_gsm8k", seeds: str = "0 1 2"):
    """End-of-run HELD-OUT table: graft, random-part control, matched LoRA, 3 seeds. Each model runs once."""
    from wsp_eval import final as final_module
    jobs = []
    for value in seeds.split():
        for arm in final_module.ARMS:
            jobs.append((donor, part_id, arm, int(value)))
    outputs = []
    for output in final_arm_remote.starmap(jobs, return_exceptions=True):
        if isinstance(output, Exception):
            print("arm failed:", repr(output)[:500])
            continue
        print(output["row"])
        outputs.append(output)
    print("ledger entries merged:", final_merge_remote.remote(donor))
    with open(os.path.join("results", "base_scores_heldout.json")) as f:
        host = json.load(f)["heldout"]
    summary = final_module.summarize(donor, part_id, outputs, host)
    print(summary["verdict_text"])
    _save("final_heldout_" + donor.replace("/", "__") + ".json", summary)


@app.local_entrypoint()
def stitch_graft(seed: int = 0):
    """Trained-stitch block graft vs ridge graft, random block and matched LoRA (one seed, search split)."""
    rows = []
    first = [("host", seed), ("block_ridge", seed), ("block_stitch", seed), ("random_stitch", seed)]
    for row in stitch_graft_remote.starmap(first, return_exceptions=True):
        if isinstance(row, Exception):
            print("arm failed:", repr(row)[:600])
            continue
        print(json.dumps(row))
        rows.append(row)
    budget = None
    for row in rows:
        if row["arm"] == "block_stitch":
            budget = row.get("budget")
    if budget is None:
        print("arm skipped: lora (block_stitch did not finish, so there is no budget to match)")
        _save("stitch_graft_s" + str(seed) + ".json", rows)
        return
    try:
        row = stitch_graft_remote.remote("lora", seed, budget)
        print(json.dumps(row))
        rows.append(row)
    except Exception as error:
        print("arm failed: lora", repr(error)[:600])
    _save("stitch_graft_s" + str(seed) + ".json", rows)


@app.local_entrypoint()
def exit_stitch(seed: int = 0):
    """Distill the donor's predictions through each stitch_graft arm's glue (or a LoRA), then score (search split)."""
    jobs = [("graft", seed), ("random", seed), ("lora", seed)]
    rows = []
    for row in exit_stitch_remote.starmap(jobs, return_exceptions=True):
        if isinstance(row, Exception):
            print("arm failed:", repr(row)[:600])
            continue
        print(json.dumps(row))
        rows.append(row)
    _save("exit_stitch_s" + str(seed) + ".json", rows)
