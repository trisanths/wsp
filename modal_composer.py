"""The Modal side of the composer: compile and practice, the LoRA comparison, and held-out scoring.

The composer loop (composer/agent.py) runs on the local machine and only
calls the OpenAI API and these functions. Compiles and scoring run here,
on one H200 each.

Launch one request:

    MODAL_PROFILE=trisanth .venv/bin/modal run modal_composer.py --request composer/requests/math_plus.json

That runs the loop to finish, compiles the comparison arms (base + LoRA,
base + distillation from each source with the host's tokenizer, and the
program with random components), scores the final program, the base, each
source model, the arms, and the best existing model at the size limit
(when the limit is far above the base) on the held-out sets in parallel,
and writes results/composer_<request>.json with a verdict for each arm.
The loop's archive is in composer/runs/<request>/<timestamp>/.

Functions:
- compile_and_practice(program_dict, payload): compile with wsp (stitches
  and a heal are the only training) into /vol/composer/<request>/<hash>,
  then score the practice sets (train splits only). The request's limits
  are in the program's budget, so the compile checks them too.
- compile_arm(arm_program, payload): compile one comparison arm.
- score_heldout(model_ref, payload): the held-out sets, fixed settings.
"""

import json
import os
import time
import traceback

import modal

app = modal.App("wsp-composer")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp", "composer")
    .add_local_dir("parts_extracted", "/root/parts_root/parts_extracted")
    .add_local_dir("person_a/parts", "/root/parts_root/person_a/parts")
)

PARTS_ROOT = "/root/parts_root"
OUT_ROOT = "/vol/composer"


def short_error(error):
    text = f"{type(error).__name__}: {error}\n" + traceback.format_exc()
    return text[-3000:]


def heal_summary(report):
    heal = report.get("heal") or {}
    losses = heal.get("losses") or []
    out = {"trainable_params": heal.get("trainable_params"), "seconds": heal.get("seconds"), "steps": len(losses)}
    if len(losses) > 0:
        first = losses[:20]
        last = losses[-20:]
        out["loss_first20"] = round(sum(first) / len(first), 4)
        out["loss_last20"] = round(sum(last) / len(last), 4)
    return out


def compile_summary(report):
    from composer.sources import sanitize

    summary = {"heal": heal_summary(report)}
    for key in ["grafts", "gates", "experts", "inserts", "budget", "lora", "carve", "splice", "mask",
                "task_vectors", "seconds", "params_after", "layers_after"]:
        if key in report and report[key] is not None:
            summary[key] = sanitize(report[key])
    return summary


def compile_program_dict(program_dict, catalog_records, out_dir):
    """Compile one program with wsp. Return the compile report."""
    import transformers

    from composer.evalsets import heal_texts_for, register_heal_data
    from wsp import Catalog, Program

    register_heal_data()
    program = Program.from_dict(program_dict)
    catalog = Catalog(catalog_records, root=PARTS_ROOT)
    texts = None
    if program.heal_spec is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
        texts = heal_texts_for(program.heal_spec.data, tokenizer, seed=program.seed)
    align = None
    if texts is not None and len(program.grafts) > 0:
        align = texts[:200]
    options = {}
    if program.carve_spec is not None and program.carve_spec.domain_data is not None:
        # Seed 0 data, as for heal (see heal_texts_for); the compiler would use the program seed.
        tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
        options["domain_texts"] = heal_texts_for(program.carve_spec.domain_data, tokenizer)
    program.compile(out_dir, catalog, align_texts=align, heal_texts=texts, device="cuda",
                    cache_dir="/vol/tensor_cache", **options)
    with open(os.path.join(out_dir, "compile_report.json")) as f:
        return json.load(f)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=4 * 3600)
def compile_and_practice(program_dict, payload):
    import torch

    from composer import sizes
    from composer.evalsets import count_params, decode_tok_s, score_set
    from wsp import Program, load

    out = {"ok": False, "errors": [], "practice_sets": {}, "practice_counts": {}}
    out_dir = f"{OUT_ROOT}/{payload['request']}/{payload['program_hash']}"
    out["model_dir"] = out_dir
    start = time.time()

    # The same step cap the local check uses, checked again here.
    cap = (payload.get("constraints") or {}).get("heal_steps_cap")
    if cap is not None:
        for what, steps in sizes.training_steps(Program.from_dict(program_dict)):
            if steps is not None and steps > cap:
                out["errors"].append(f"{what} has {steps} steps, over the cap of {cap}")
                return out

    try:
        torch.manual_seed(program_dict.get("seed", 0))
        report = compile_program_dict(program_dict, payload["catalog"], out_dir)
        out["compile_summary"] = compile_summary(report)
    except Exception as error:
        out["errors"].append("compile failed: " + short_error(error))
        out["compile_seconds"] = round(time.time() - start, 1)
        return out
    out["compile_seconds"] = round(time.time() - start, 1)
    volume.commit()

    step = time.time()
    try:
        model, tokenizer = load(out_dir, device="cuda")
    except Exception as error:
        out["errors"].append("load failed: " + short_error(error))
        return out
    # The compile report holds the measured numbers: params_active leaves out the input
    # embedding table and the experts that top_k leaves out.
    total = report.get("params_total")
    active = report.get("params_active")
    if total is None:
        out["params_total_b"] = round(count_params(model), 4)
        out["params_run_b"] = out["params_total_b"]
    else:
        out["params_total_b"] = round(total / 1e9, 4)
        out["params_run_b"] = round((active if active is not None else total) / 1e9, 4)
    out["heal_seconds"] = report.get("heal_seconds")
    out["heal_steps_run"] = len((report.get("heal") or {}).get("losses") or [])
    out["heal_stopped_at_limit"] = bool((report.get("heal") or {}).get("stopped_early", False))
    try:
        out["decode_tok_s"] = decode_tok_s(model, tokenizer)
    except Exception as error:
        out["errors"].append("decode speed failed: " + short_error(error))

    for name in payload["practice_sets"]:
        try:
            value, count, seconds = score_set(model, tokenizer, name, payload.get("practice_items"))
            out["practice_sets"][name] = round(value, 4)
            out["practice_counts"][name] = count
        except Exception as error:
            out["errors"].append(f"practice set {name} failed: " + short_error(error))
    out["practice_seconds"] = round(time.time() - step, 1)
    out["ok"] = len(out["practice_sets"]) > 0
    with open(os.path.join(out_dir, "practice.json"), "w") as f:
        json.dump(out, f, indent=2)
    volume.commit()
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=4 * 3600)
def compile_arm(arm_program, payload):
    """Compile one comparison arm (composer/arms.py): a LoRA or distillation program on the host, or the
    final program with random components. The program is built locally, so it carries its own heal."""
    start = time.time()
    out_dir = f"{OUT_ROOT}/{payload['request']}/arm_{payload['program_hash']}_{payload['arm_hash']}"
    try:
        report = compile_program_dict(arm_program, payload.get("catalog") or [], out_dir)
    except Exception as error:
        return {"ok": False, "model_dir": None, "error": short_error(error)}
    volume.commit()
    heal = report.get("heal") or {}
    return {"ok": True, "model_dir": out_dir, "seconds": round(time.time() - start, 1),
            "heal_seconds": report.get("heal_seconds"), "heal_steps_run": len(heal.get("losses") or []),
            "params_total_b": round((report.get("params_total") or 0) / 1e9, 4)}


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600)
def score_heldout(model_ref, payload):
    """Score one model on the request's held-out sets. model_ref is a /vol path or a Hugging Face id."""
    import torch
    import transformers

    from composer.evalsets import count_params, decode_tok_s, score_set
    from composer.skills import skill_scores
    from wsp import load

    start = time.time()
    out = {"sets": {}, "errors": []}
    if model_ref.startswith("/vol/"):
        volume.reload()
        model, tokenizer = load(model_ref, device="cuda")
    else:
        kwargs = {"dtype": torch.bfloat16}
        if "gemma" in model_ref.lower():
            kwargs["attn_implementation"] = "eager"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, **kwargs).to("cuda").eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref)
    out["params_total_b"] = round(count_params(model), 4)
    try:
        out["decode_tok_s"] = decode_tok_s(model, tokenizer)
    except Exception as error:
        out["errors"].append("decode speed failed: " + short_error(error))
    values = {}
    for name in payload["heldout_sets"]:
        try:
            value, count, seconds = score_set(model, tokenizer, name, payload.get("heldout_items"))
            values[name] = value
            out["sets"][name] = {"score": round(value, 4), "items": count, "seconds": seconds}
        except Exception as error:
            out["errors"].append(f"held-out set {name} failed: " + short_error(error))
    out["skills"] = skill_scores(values, payload["skills"], "heldout")
    out["seconds"] = round(time.time() - start, 1)
    return out


@app.local_entrypoint()
def main(request: str, skip_heldout: bool = False):
    from composer.agent import REPO_ROOT, load_request, run_request
    from composer.comparison import run_comparison
    from composer.report import build_results
    from composer.sources import Sources

    def practice_fn(program_dict, payload):
        return compile_and_practice.remote(program_dict, payload)

    def compile_many(calls):
        return list(compile_arm.starmap(calls, return_exceptions=True))

    def score_many(calls):
        return list(score_heldout.starmap(calls, return_exceptions=True))

    spec = load_request(request)
    summary = run_request(request, practice_fn)
    final = summary["final"] or {}
    print(f"loop stopped: {summary['stop_reason']}; compiles {summary['compile_budget_used']} of "
          f"{summary['compile_budget']}; archive {summary['archive_dir']}")
    sources = Sources(REPO_ROOT)
    if final.get("no_model") or skip_heldout:
        out = build_results(spec, summary, [], None, sources.records_by_id())
    else:
        # A failed arm compile or score must not lose the held-out scores of the other models:
        # run_comparison keeps each error in its row.
        out = run_comparison(spec, summary, sources, compile_many, score_many)
    os.makedirs("results", exist_ok=True)
    path = f"results/composer_{spec['name']}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    if out.get("final") and out["final"].get("skills"):
        print("final held-out skills:", json.dumps(out["final"]["skills"]))
    for item in out.get("verdicts", []):
        print(item["text"])
