"""Round 23: which mined parts keep their skill after a compile into another model.

No benchmark data trains anything.

DO NOT RUN YET. docs/partcheck_review.md lists defects that make its verdicts
invalid (held-out items that repeat search items, group-level gains copied to
every part, undertrained heals, a wrong OLMo-2 norm). main() refuses to start.

Question. A miner finds a part (MLP neuron cluster, attention heads) in a
donor model and freezes a skill file for it: items, matched controls, and a
search half and a held-out half. The compiler (wsp/synth.py) puts the part
into a different base model with learned translation layers at the seams.
Per part: does the base model, after the compile, show the part's skill on the
HELD-OUT items, more than the same compile with a random same-size part of the
same donor layer, and more than the same compile with the part's weights
scrambled? Three controls of the answer are reported: recovered share of the
donor-over-base gap, contribution (compiled minus the same network with only
this part's gate closed), and content_z against several control seeds.

Run --smoke first and wait for its row. Smoke runs ONE donor with at most 3
parts and one control seed, so a first row appears in minutes. A larger run
starts only after that row printed: main refuses more than 3 cells unless
results/round23_rows.json has a row (or --force is given), and refuses a tf5 run
until a tf5 smoke has printed a row. (LESSONS.md: four rounds died on setup on
2026-11-14.) Smoke the tf5 image once before the tf5 cells, and read the drift
numbers it prints (drift_ref, the control median) before trusting any "harmful".

At most --max-containers cells run at once (default 4; the workspace cap is about
10). Bases shared by two cells are downloaded once, first (warm_models). A donor
with more than 24 parts is cut to 24 (large parts first). Stage 1 gets 100 steps
per part. Each cell saves its partial report after every compile and stops
starting compiles after --wall-hours (default 5.2).

    modal run modal_round23.py --smoke
    modal run modal_round23.py --only-donors qwenmath1.5b,granite8b --seeds 0,1,2
    modal run modal_round23.py --only-donors some/New-Donor-7B --base some/New-Small-1B
    modal run modal_round23.py --tf5 --only-donors qwen3.8_27b     # donors that need the 5.23 image
    modal run modal_round23.py --list                              # print the table, run nothing

No reported benchmark is read anywhere. Training uses the miners' own synthetic
skill items (the "search" half of each skill file) and general text from the
wsp.healdata pools (healdata.check_sources runs on every dataset id).
Scoring uses only the "heldout" half of each skill file, which the mining agent
never saw, plus its matched control items. No dataset is loaded by this file.

Parts come from library/catalog.jsonl and the per-donor files
person_a/catalog/*.jsonl, deduped by part id, tier "screened" only
(wsp.partcheck.load_parts). Parts with 64 units or more come first when
--max-parts cuts the list.

Donor to base pairs (Hub API checked 2026-11-14: model ids, config, tokenizer file hash; lineage by
wsp.lineage.relation):
  Qwen2.5-Math-1.5B       into Qwen2.5-1.5B-Instruct       diverged_same_shape, same tokenizer
  Qwen2.5-Math-7B-Instruct into Qwen2.5-7B-Instruct        diverged_same_shape, same tokenizer
  Qwen2.5-Coder-7B-Inst.  into Qwen2.5-7B-Instruct         diverged_same_shape, same tokenizer
  Qwen2.5-Coder-7B-Inst.  into Qwen2.5-1.5B-Instruct       same_family_other_size (vocab 152064 vs 151936)
  granite-3.3-8b          into granite-3.3-2b-instruct     same_family_other_size, same tokenizer
  Nanbeige4.3-33B         into Nanbeige4.1-3B              same_family_other_size (lineage reads the tokenizer as same)
  OLMo-2-1124-7B-Instruct into OLMo-2-0425-1B-Instruct     same_family_other_size, same tokenizer
  Apollo2-7B              into Apollo2-1.5B                same_family_other_size, same tokenizer
  (Apollo2-7B into Qwen2.5-1.5B-Instruct is same family but logit KD is not usable, so it is not chosen)
  K2-Horizon-7B           into K2-Horizon-3.7B             same_family_other_size; needs the tf5 image
Donors that need transformers 5.x (not run unless --tf5, never silently skipped; their bases are NOT lineage
checked and are a first guess from tokenizer files):
  Qwen3.6-35B-A3B -> Qwen3.5-4B (same tokenizer file), Qwen3.8-27B -> Qwen3.5-9B (same vocabulary, 248044
  tokens; the tokenizer files differ, vocabularies equal, so logit KD applies), gemma-4-31B-it -> gemma-4-E4B-it
  (same model_type gemma4, same vocabulary of 262144 tokens, so logit KD applies), K2-Horizon-7B -> K2-Horizon-3.7B.
  (Vocabularies compared from the Hub tokenizer.json files, 2026-11-14; lineage.relation not run.)
OLMo-2 cells are refused: wsp.parts.find_layer_modules reads its norms wrongly (partcheck.UNSUPPORTED_MODEL_TYPES).
"""

import json
import os
import re
import time

import modal

app = modal.App("wsp-round23")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

HERE = os.path.dirname(os.path.abspath(__file__))


def safe_reload():
    """Reload the volume to see other shards' files. Open files (HF logs, caches) block a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200])


def build_image(transformers_version):
    """The round 20 pip list (zstandard included). tiktoken is added for the tf5 donors, as in modal_atlas.py."""
    pip = ["torch==2.7.1", "transformers==" + transformers_version, "accelerate", "huggingface_hub", "requests",
           "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf"]
    if transformers_version.startswith("5"):
        pip.append("tiktoken")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(*pip)
        .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
        .add_local_python_source("wsp")
    )
    for folder in ("library", "person_a/catalog", "person_a/mine/skills", "person_a/mine/data", "person_a/parts"):
        if os.path.isdir(os.path.join(HERE, folder)):
            image = image.add_local_dir(os.path.join(HERE, folder), "/root/" + folder)
    return image


image = build_image("4.57.1")
image_tf5 = build_image("5.23.0")

# key: (catalog model id, base id, needs the 5.23 image, relation known from the Hub check, note)
DONORS = {
    "qwenmath1.5b": ("Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", False, "diverged_same_shape", ""),
    "qwenmath7b": ("Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", False, "diverged_same_shape",
                   "63 parts in person_a/catalog, 43 with 64 units or more"),
    "qwencoder7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", False, "diverged_same_shape", ""),
    "qwencoder7b_small": ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", False,
                          "same_family_other_size", "vocab 152064 vs 151936: logit KD may fall back to LM loss"),
    "granite8b": ("ibm-granite/granite-3.3-8b-instruct", "ibm-granite/granite-3.3-2b-instruct", False,
                  "same_family_other_size", ""),
    "nanbeige33b": ("Nanbeige/Nanbeige4.3-33B", "Nanbeige/Nanbeige4.1-3B", False, "same_family_other_size",
                    "vocab 166912 vs 166144"),
    "olmo2_7b": ("allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-0425-1B-Instruct", False,
                 "same_family_other_size", "refused: find_layer_modules misreads OLMo-2 norms"),
    "apollo2_7b": ("FreedomIntelligence/Apollo2-7B", "FreedomIntelligence/Apollo2-1.5B", False,
                   "same_family_other_size", "Apollo2-7B is Qwen2-7B based"),
    "k2horizon7b": ("IFM/K2-Horizon-7B", "IFM/K2-Horizon-3.7B", True, "same_family_other_size",
                    "custom code, config transformers_version 5.4.0"),
    "qwen3.6_35b_a3b": ("Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-4B", True, None, "base not lineage checked"),
    "qwen3.8_27b": ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-9B", True, None, "base not lineage checked; same vocabulary"),
    "gemma4_31b": ("google/gemma-4-31B-it", "google/gemma-4-E4B-it", True, None, "base not lineage checked; same vocabulary"),
}

# Cells that run_donor would refuse after paying for a GPU container. main drops them before launch.
REFUSED = {"olmo2_7b": "wsp.parts.find_layer_modules misreads the OLMo-2 norms; fix it and test a tiny Olmo2 compile first"}

NEEDS_TF5 = [k for k in DONORS if DONORS[k][2]]

# Default heal budget. Stage 1 only unless --stage2. seq_len 512: items longer are dropped, never cut.
HEAL = {"stage1_steps": 300, "stage2_steps": 300, "batch_size": 4, "seq_len": 512, "lora_rank": 16,
        "steps_in": 100, "steps_lm": 60, "batch_in": 4, "lr_lora": 5e-5}
SMOKE_HEAL = dict(HEAL, stage1_steps=100, steps_in=40, steps_lm=20)
GENERAL_POOLS = [("general", 150), ("prose", 150)]
DEFAULT_SEEDS = [0, 1, 2]
CONFIRM_SEEDS = [3, 4]  # run only when a part would survive with fewer than 5 seeds
MAX_PARTS_DEFAULT = 24  # a donor with more parts is cut to this many, large parts first
MAX_CONTAINERS = 4
WALL_HOURS = 5.2        # no compile starts after this; the Modal timeout is 6 hours

# Rough GPU-hours per cell (H200), set from the parameter counts, not measured: see docs/partcheck.md.
GPU_HOURS = {"qwenmath1.5b": 1.5, "qwenmath7b": 4.0, "qwencoder7b": 3.0, "qwencoder7b_small": 2.5, "granite8b": 3.0, "nanbeige33b": 5.0,
             "olmo2_7b": 2.5, "apollo2_7b": 2.0, "k2horizon7b": 3.0, "qwen3.6_35b_a3b": 5.0, "qwen3.8_27b": 5.0,
             "gemma4_31b": 5.0}


def key_of(model_id):
    """Donor key for a catalog model id. Unknown donors get a file-safe slug of the id."""
    for key in DONORS:
        if DONORS[key][0] == model_id and not key.endswith("_small"):
            return key
    return re.sub(r"[^a-z0-9.]+", "_", model_id.lower()).strip("_")


def plan(only, base, tf5, smoke):
    """Return the list of (key, donor id, base id, relation, note, needs tf5) to run."""
    wanted = [w for w in only.split(",") if w]
    out = []
    if not wanted:
        for key in DONORS:
            if DONORS[key][2] and not tf5:
                continue
            if tf5 and not DONORS[key][2]:
                continue
            out.append(key)
        if smoke:
            out = out[:1]
    else:
        out = wanted
    cells = []
    for item in out:
        if item in DONORS:
            donor, default_base, needs, relation, note = DONORS[item]
            if needs and not tf5:
                print("needs the 5.23 image, add --tf5:", item)
                continue
            if tf5 and not needs:
                print("--tf5 given, running", item, "on the 5.23 image")
            cells.append((item, donor, base or default_base, relation, note, needs or tf5))
        else:
            if not base:
                raise SystemExit("donor " + item + " is not in DONORS: pass --base")
            cells.append((key_of(item), item, base, None, "new donor", tf5))
    return cells


# ------------------------------------------------------------------ the run


def load_pair(donor_id, base_id):
    """Load donor and base on the GPU. Return ((donor, donor tokenizer), (base, base tokenizer))."""
    import torch
    import transformers

    out = []
    for name in (donor_id, base_id):
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, trust_remote_code=True)
        except ValueError:
            model = transformers.AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16,
                                                                             trust_remote_code=True)
        model = model.to("cuda").eval()
        tok = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        out.append((model, tok))
    return out[0], out[1]


def run_donor(key, donor_id, base_id, relation_hint, opts):
    """Compile one donor's parts into one base and return the partcheck report."""
    import transformers
    from wsp import healdata
    from wsp import partcheck

    start = time.time()
    safe_reload()
    groups, missing = partcheck.load_parts("/root/library/catalog.jsonl", tiers=("screened",), models=[donor_id])
    parts = groups.get(donor_id, [])
    print("round23 cell=%s donor=%s parts=%d missing=%d base=%s" % (key, donor_id, len(parts), len(missing), base_id))
    if len(parts) == 0:
        return {"donor": donor_id, "base": base_id, "key": key, "parts": [], "missing": missing, "skipped": [],
                "group": None, "relation": None, "seconds": 0.0}
    for model_id in (donor_id, base_id):
        model_type = None
        try:
            model_type = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True).model_type
        except Exception as error:
            print("round23 cell=%s config read failed for %s: %s" % (key, model_id, repr(error)[:200]))
        reason = partcheck.unsupported_reason(model_type)
        if reason:
            print("round23 cell=%s refused: %s is %s: %s" % (key, model_id, model_type, reason))
            return {"donor": donor_id, "base": base_id, "key": key, "parts": [], "missing": missing, "group": None,
                    "relation": None, "seconds": 0.0, "refused": reason,
                    "skipped": [{"id": p["id"], "reason": "refused: " + reason} for p in parts]}
    relation = partcheck.relation_of(base_id, donor_id)
    if relation.get("kind") == "unknown" and relation_hint:
        relation = {"kind": relation_hint, "base": base_id, "donor": donor_id, "note": "Hub check, 2026-11-14"}
    donor, base = load_pair(donor_id, base_id)
    base_tok = base[1]
    general = []
    for pool, count in GENERAL_POOLS:
        healdata.check_sources([s[0] for s in healdata.SOURCES[pool]])
        general += healdata.pool(pool, count, tokenizer=base_tok, seed=3)
    smoke = opts["smoke"]
    budget = dict(SMOKE_HEAL if smoke else HEAL)
    cap = opts["max_parts"]
    if not cap and len(parts) > MAX_PARTS_DEFAULT:
        cap = MAX_PARTS_DEFAULT
        print("round23 cell=%s has %d parts, cut to %d (large parts first)" % (key, len(parts), cap))
    cell = "%s|%s" % (key, base_id)
    hash_bad = partcheck.hash_mismatches(groups)

    def save_partial(partial):
        partial = dict(partial, key=key, missing=missing, hash_mismatch=hash_bad, parts_total=len(parts))
        save_report(partial, key)

    report = partcheck.check_parts(
        base, donor, parts, general, base_id=base_id, relation=relation, stage2=opts["stage2"],
        seeds=opts["seeds"], per_part=opts["per_part"], budget=budget, max_parts=cap, cell=cell,
        confirm_seeds=opts["confirm_seeds"], on_progress=save_partial, deadline=start + 3600 * opts["wall_hours"])
    report["key"] = key
    report["missing"] = missing
    report["parts_total"] = len(parts)
    report["hash_mismatch"] = hash_bad
    report["wall_seconds"] = round(time.time() - start, 1)
    return report


def save_report(report, key):
    """Write the report to the volume. Called after every compile with a partial report, then with the final one."""
    os.makedirs("/vol/results/round23", exist_ok=True)
    path = "/vol/results/round23/%s__%s.json" % (key, re.sub(r"[^A-Za-z0-9.]+", "_", str(report.get("base"))))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, default=str)
    os.replace(tmp, path)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, timeout=3600)
def warm_models(model_ids):
    """Download model files into /vol/hf once, so cells that share a model do not download it at the same time."""
    from huggingface_hub import snapshot_download

    safe_reload()
    for model_id in model_ids:
        snapshot_download(model_id, ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx", "*.h5", "original/*", "*.msgpack"])
        print("round23 warmed", model_id)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=6 * 3600)
def donor_cell(key, donor_id, base_id, relation_hint, opts):
    report = run_donor(key, donor_id, base_id, relation_hint, opts)
    save_report(report, key)
    return report


@app.function(image=image_tf5, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=6 * 3600)
def donor_cell_tf5(key, donor_id, base_id, relation_hint, opts):
    report = run_donor(key, donor_id, base_id, relation_hint, opts)
    save_report(report, key)
    return report


ROWS_PATH = os.path.join("results", "round23_rows.json")


def smoke_marker(tf5):
    return os.path.join("results", "round23_smoke_%s.ok" % ("tf5" if tf5 else "tf4"))


def count_rows(path=ROWS_PATH):
    """Number of rows in the rows file, 0 when it does not exist or cannot be read."""
    try:
        with open(path) as f:
            return len(json.load(f))
    except Exception:
        return 0


def gate(cells, smoke, tf5, force, rows_path=ROWS_PATH, marker_exists=os.path.exists):
    """Return a refusal text, or None when the run may start.

    More than 3 cells need a row in the rows file (an earlier smoke printed one). A tf5 run that is not itself a
    smoke needs a tf5 smoke marker, since that image path has never run.
    """
    if smoke or force:
        return None
    if len(cells) > 3 and count_rows(rows_path) == 0:
        return "more than 3 cells and %s has no row: run --smoke first, or pass --force" % rows_path
    if tf5 and not marker_exists(smoke_marker(True)):
        return "no tf5 smoke has printed a row: run --tf5 --smoke first, or pass --force"
    return None


def shared_models(cells):
    """Model ids that two or more cells load (donor or base), in first-seen order."""
    seen = {}
    for key, donor, base_id, relation, note, needs in cells:
        for model_id in (donor, base_id):
            seen.setdefault(model_id, set()).add(key)
    return [m for m in seen if len(seen[m]) > 1]


def collect(call, report_key, partcheck):
    """Write one finished cell's report and rows. Return the report."""
    report = call.get()
    path = os.path.join("results", "round23_%s.json" % report_key)
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    reports = [r for r in old if r.get("base") != report.get("base")] + [report]
    with open(path, "w") as f:
        json.dump(partcheck.json_safe(reports), f, indent=2)
    rows = partcheck.rows_from_report(report)
    partcheck.write_rows(rows, ROWS_PATH)
    for r in rows:
        print("round23 cell=%s donor=%s part=%s base=%s verdict=%s n_seeds=%s recovered=%s contribution=%s content_z=%s" % (
            report_key, report.get("donor"), r["id"], r["base"], r["verdict"], r["n_seeds"], r["recovered"],
            r["contribution"], r["content_z"]))
    print("round23 cell=%s donor=%s group=%s seconds=%s" % (report_key, report.get("donor"), json.dumps(report.get("group")),
                                                         report.get("wall_seconds")))
    return report


@app.local_entrypoint()
def main(only_donors: str = "", base: str = "", seeds: str = "", max_parts: int = 0, smoke: bool = False,
         tf5: bool = False, stage2: bool = False, per_part: bool = False, list: bool = False,
         max_containers: int = MAX_CONTAINERS, force: bool = False, confirm_seeds: str = "3,4",
         wall_hours: float = WALL_HOURS):
    raise SystemExit("round 23 is not ready: fix the defects in docs/partcheck_review.md first")
    from wsp import partcheck

    cells = plan(only_donors, base, tf5, smoke)
    kept = []
    for c in cells:
        if c[0] in REFUSED and not list:
            print("round23 cell=%s dropped: %s" % (c[0], REFUSED[c[0]]))
        else:
            kept.append(c)
    cells = kept
    if list:
        for key, donor, base_id, relation, note, needs in cells:
            print("%-18s %s -> %s relation=%s tf5=%s hours~%s %s" % (key, donor, base_id, relation, needs,
                                                                     GPU_HOURS.get(key, "?"), note))
        return
    seed_list = [int(s) for s in seeds.split(",") if s != ""] or DEFAULT_SEEDS
    cap = max_parts or None
    confirm = [int(s) for s in confirm_seeds.split(",") if s != ""]
    if smoke:
        seed_list = seed_list[:1]
        cap = min(cap or 3, 3)
        cells = cells[:1]
        confirm = []
    refusal = gate(cells, smoke, tf5, force)
    if refusal:
        raise SystemExit("round23 refused: " + refusal)
    opts = {"smoke": smoke, "seeds": seed_list, "max_parts": cap, "stage2": stage2, "per_part": per_part,
            "confirm_seeds": confirm, "wall_hours": wall_hours}
    print("round23 cells:", [(c[0], c[2]) for c in cells], "opts:", opts, "max_containers:", max_containers)
    shared = shared_models(cells)
    if shared:
        print("round23 warming shared models:", shared)
        warm_models.remote(shared)
    pending = [(cells[i][0], cells[i][5], (cells[i][0], cells[i][1], cells[i][2], cells[i][3], opts))
               for i in range(len(cells))]
    active = []
    os.makedirs("results", exist_ok=True)
    got_row = False
    while pending or active:
        while pending and len(active) < max(1, max_containers):
            key, needs, args = pending.pop(0)
            fn = donor_cell_tf5 if needs else donor_cell
            active.append((key, fn.spawn(*args)))
            print("round23 cell=%s started (%d running)" % (key, len(active)))
        still = []
        for key, call in active:
            try:
                report = collect_wait(call, key, partcheck)
            except Exception as error:
                print("round23 cell=%s failed: %s (a partial report may be in /vol/results/round23)" % (key, repr(error)[:600]))
                continue
            if report is None:
                still.append((key, call))
            elif report.get("parts"):
                got_row = True
        active = still
    if smoke and got_row:
        with open(smoke_marker(tf5), "w") as f:
            f.write("smoke row printed\n")


def collect_wait(call, key, partcheck):
    """Return the cell's report when it is done, None when it is still running (waits up to 20 seconds)."""
    try:
        call.get(timeout=20)
    except Exception as error:
        if "imeout" in type(error).__name__ or isinstance(error, TimeoutError):
            return None
        raise
    return collect(call, key, partcheck)
