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
def interp_remote(max_evals, band_size, ids):
    from wsp_eval.interp_agent import InterpAgent
    _prepare_catalog()
    agent = InterpAgent(max_evals=max_evals, band_size=band_size, ids=ids)
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
def interp(max_evals: int = 80, band_size: int = 4, ids: str = ""):
    id_list = None
    if ids.strip():
        id_list = ids.split()
    result = interp_remote.remote(max_evals, band_size, id_list)
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
