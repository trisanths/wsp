"""Run a Sapien request on Modal: plan, make, comparison arms, and the reported scores, in one call.

    modal run modal_sapien.py --request sapien_requests/qwen15_code_seam.json
    modal run modal_sapien.py --request sapien_requests/llama8b_math_medicine.json --plan-only

--plan-only prints the plan on this machine (it reads the Hub by range reads, uses no GPU, and starts nothing
on Modal). A full run writes results/sapien_<name>.json here, and the compiled directories stay on the volume
wsp-models under /vol/sapien/<name>.

Order of a run: make (wsp.auto.make: stages, probe report, sapien_report.json), then the comparison arms of the
plan (base, base + LoRA, base + distillation, best existing model), then score_reported once for the result and
each arm. Nothing scores on the reported benchmarks before that last call.

Unverified: this runner and the methods behind it have not run on a GPU on real weights.
"""

import json
import os
import time

import modal

app = modal.App("wsp-sapien")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)


def safe_reload():
    """Reload the volume to see other shards' files. Open files (HF logs, caches) block a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf", "pyyaml",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
# The planner reads mined data from the repo next to the wsp package (REPO/atlas, REPO/library, catalog.jsonl).
# The container plans nothing (the local plan is passed in), but the stages read library and atlas component rows.
HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in ("atlas", "library"):
    if os.path.isdir(os.path.join(HERE, _folder)):
        image = image.add_local_dir(os.path.join(HERE, _folder), "/root/" + _folder)
# The pgraft method builds its data and eval sets with the round modules of modal_wi_xfam.py when the volume has
# no cached pool (wsp/xfam.py reads them lazily).
for _name in ("modal_graft", "modal_round5", "modal_round6", "modal_round7", "modal_round22pg", "modal_round23pg"):
    if os.path.exists(os.path.join(HERE, _name + ".py")):
        image = image.add_local_file(os.path.join(HERE, _name + ".py"), "/root/%s.py" % _name)
if os.path.exists(os.path.join(HERE, "catalog.jsonl")):
    image = image.add_local_file(os.path.join(HERE, "catalog.jsonl"), "/root/catalog.jsonl")


def _modal_backend_class():
    from wsp import auto

    class ModalBackend(auto.HubBackend):
        """HubBackend that commits the volume when a stage asks (the pgraft report and artifact)."""

        def checkpoint(self):
            try:
                volume.commit()
            except Exception as error:
                print("commit skipped:", repr(error)[:200])

    return ModalBackend


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=393216, timeout=20 * 3600)
def run_request(request, plan=None, atlas_entries=None, limit=None, settings=None):
    """Make the request on one H200, build the comparison arms, and score once. Returns the full result dict.

    plan is the plan the caller checked and approved; it runs as given. A failure after the build still returns
    the report, with the error recorded.
    """
    import gc

    import torch

    from wsp import auto

    safe_reload()
    started = time.time()
    backend = _modal_backend_class()(device="cuda", dtype=torch.bfloat16, cache_dir="/vol/hf/hub")
    out_dir = f"/vol/sapien/{request['name']}"
    result = auto.make(request, out_dir, backend, plan_=plan, atlas_entries=atlas_entries, settings=settings)
    report = result.report
    try:
        volume.commit()
    except Exception as error:
        report["commit_error"] = repr(error)[:300]
    plan = report["plan"]
    if [st["method"] for st in plan["stages"]] == ["pgraft"]:
        # The write-in scored itself on held-out items inside make (base, reloaded network, controls); there are no
        # generative arms to build.
        report["total_seconds"] = round(time.time() - started, 1)
        with open(os.path.join(out_dir, "sapien_report.json"), "w") as handle:
            json.dump(auto._json_safe(report), handle, indent=1)
        try:
            volume.commit()
        except Exception as error:
            report["commit_error"] = repr(error)[:300]
        return auto._json_safe(report)
    # The result is scored first and then freed: the holder gives up its only reference when the arm loads.
    holder = {"pair": (result.model, result.tokenizer)}
    result.model = None
    arms = auto.build_arms(request, plan, backend, result_loader=lambda: holder.pop("pair"))
    ordered = {"result": arms.pop("result")}
    ordered.update(arms)
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    if "base_distill" in ordered and 2 * plan["base"]["params"] * 2 > 0.7 * gpu_bytes:
        del ordered["base_distill"]
        report["base_distill_skipped"] = "the base and its teacher do not fit in 70 percent of GPU memory"
    gc.collect()
    try:
        scored = auto.score_reported(request, ordered, limit=limit, keep_going=True)
        report["scored_on_reported"] = True
        report["reported_scores"] = scored
    except Exception as error:
        report["reported_scores_error"] = repr(error)[:500]
    report["total_seconds"] = round(time.time() - started, 1)
    try:
        with open(os.path.join(out_dir, "sapien_report.json"), "w") as handle:
            json.dump(auto._json_safe(report), handle, indent=1)
        volume.commit()
    except Exception as error:
        report["write_error"] = repr(error)[:300]
    return auto._json_safe(report)


@app.local_entrypoint()
def main(request: str, plan_only: bool = False, limit: int = 0, smoke: bool = False, settings: str = ""):
    from wsp import auto

    spec = auto.load_request(request)
    overrides = json.loads(settings) if settings else None
    if smoke:
        # A cheap check of a pgraft request on the real models: 3 steps, 32 items per eval set, 64 pool items.
        spec["name"] = spec["name"] + "_smoke"
        overrides = dict(overrides or {}, pgraft_spec={"steps": 3, "align_steps": 3}, pgraft_eval_limit=32,
                         pgraft_pool_limit=64)
    if plan_only:
        plan = auto.plan(spec)
        print(auto.format_plan(plan))
        return
    plan = auto.plan(spec)
    if not plan["feasible"]:
        print(auto.format_plan(plan))
        raise SystemExit("the plan cannot run; see the refusals above")
    result = run_request.remote(spec, plan, auto.load_atlas(), limit or None, overrides)
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"sapien_{spec['name']}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    if "pgraft" in result:
        print("scores:", json.dumps(result["reported_scores"]["result"]))
        print("base:", json.dumps(result["reported_scores"]["base"]))
    print("verdict:", result["verdict"]["verdict"])
    for reason in result["verdict"]["reasons"]:
        print("  -", reason)
    print("wrote", path)
