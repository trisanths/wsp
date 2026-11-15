"""Round 25: minimal layer set per skill, then compose two skills (wsp/layerset.py).

Experiment 1 of docs/research/2026-11-15-transfer-sweep.md. A true fine-tune
may keep its skill in a few layers. For each member this round finds the
smallest set S of layers whose difference (attention and MLP together) keeps
90 percent of the member's probe transfer, then writes two members, each on
its own S, into one host-sized network.

Rule: no benchmark data trains anything, and nothing here selects on a
benchmark either. Selection reads only the probe texts of wsp.healdata pools
(the caches of round 21, same key). Benchmarks are scored once, at the end.

Family qwen7b: host Qwen/Qwen2.5-7B-Instruct, members OpenThinker-7B (math
and reasoning) and OpenCodeReasoning-Nemotron-7B (code), both gated in round
21. The downloadable reference is Qwen/Qwen2.5-Coder-7B-Instruct.

Probe splits (round 21's): a selects (ranking, bisection, greedy check,
composition), b checks the chosen set, c is never used to select and gives
every reported number.

Stages. Each writes its row to the volume the moment it ends, so a lost
client loses nothing, and a rerun skips a stage whose row exists (--redo
repeats it). The path is /vol/results/round25/v<FIT_VERSION>/<cell>/ with
one <stage>.json per stage and rows.jsonl. Bump FIT_VERSION when the fit
code changes, so an old row is never read as a new one.
  caches            round 21's teacher caches (built only when a file is missing)
  gates             the weave gate numbers of each member and the host calibration
  importance_<m>    one layer at a time: both, attention only, MLP only
  minimal_<m>       bisection over the ranking, then a greedy check, reported on a, b, c
  compose           both members on their sets; overlaps by larger norm, or by trying both
  controls_<m>      random sets of the same size (--seeds of them), the set chosen by
                    difference norm only, the scrambled difference on S, and the
                    cross-skill swap (this member's layers on the other member's S)
  bench_<arm>       benchmark scores, once per arm
  summary           the table, from the saved rows

Benchmarks: MBPP 257, HumanEval 164, GSM8K 300, MMLU 1000. Arms: host, each
member alone, each member on its minimal set, the random set of equal size
(seed 1), the composed model, and Qwen2.5-Coder-7B-Instruct. All arms use the
host's tokenizer and chat template and the same long token limit (2048 new
tokens), because the reasoning members write long chains. Two raw generations
per benchmark per arm go to the log.

Go for a member: |S| at most 50 percent of the layers, S keeps at least 90
percent of the transfer on split c, the random set keeps less than 50
percent, and the set is within 5 points of the member on its benchmark.
Stop: no S under 75 percent of the layers holds 90 percent.

    modal run --detach modal_round25.py --family qwen7b --smoke
    modal run --detach modal_round25.py --family qwen7b --seeds 5
    modal run --detach modal_round25.py --family qwen7b --only bench,summary
    modal run modal_round25.py --family qwen7b --plan-only
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modal

import modal_round21 as r21

app = modal.App("wsp-round25")
volume = r21.volume
image = r21.image.add_local_python_source("modal_round21")

FIT_VERSION = 1
OUT_ROOT = f"/vol/results/round25/v{FIT_VERSION}"
PROBE_SPLITS = r21.PROBE_SPLITS
SELECT_SPLIT = "a"
CHECK_SPLIT = "b"
REPORT_SPLIT = "c"
TARGET_SHARE = 0.9
DEFAULT_SEEDS = 5
SCRAMBLE_SEEDS = [1, 2, 3]
RANDOM_BENCH_SEED = 1
SIZE_GO = 0.5
SIZE_STOP = 0.75
RANDOM_KEEP_MAX = 0.5
BENCH_POINTS = 0.05
COMPOSE_KEEP = 0.95

MAX_NEW_LONG = {"gsm8k": 2048, "mbpp": 2048, "humaneval": 2048}
GEN_BATCH = 64
BENCH_SKILLS = ["mbpp", "humaneval", "gsm8k", "mmlu"]
SMOKE_SKILLS = ["mbpp"]
FINAL_COUNTS = {"mbpp": 257, "humaneval": 164, "gsm8k": 300, "mmlu": 1000}
RAW_SHOWN = 2
RAW_CHARS = 1200

LAYERS = {"qwen7b": 28}
FAMILIES = {
    "qwen7b": {
        "base": "qwen7b", "reference": "Qwen/Qwen2.5-Coder-7B-Instruct", "smoke_members": ["ocr_nemotron"],
        "go_skill": {"openthinker": "gsm8k", "ocr_nemotron": "mbpp"},
    },
}

STAGE_GROUPS = ("caches", "gates", "importance", "minimal", "compose", "controls", "bench", "summary")

# Seconds for the plan. Every number is a model, not a measurement.
APPLY_LAYER_SECONDS = 0.5
APPLY_SET_SECONDS = 4.0
GEN_STEP_FACTOR = 1.15
ARM_CAP_SHARE = {"host": 0.25, "reference": 0.25, "random": 0.5, "member": 1.0, "minimal": 1.0, "composed": 1.0}


# ---------------------------------------------------------------- cells, arms, plan (no torch)


def cell_family(name, smoke=False):
    """The round 21 family spec for this cell: all members, or only the smoke member."""
    cfg = FAMILIES[name]
    fam = dict(r21.FAMILIES[cfg["base"]])
    if smoke:
        fam["members"] = [m for m in fam["members"] if m["name"] in cfg["smoke_members"]]
    return fam


def cell_name(name, smoke=False):
    return name + ("-smoke" if smoke else "")


def arms_for(name, smoke=False):
    fam = cell_family(name, smoke)
    arms = ["host"]
    arms += ["member:" + m["name"] for m in fam["members"]]
    arms += ["minimal:" + m["name"] for m in fam["members"]]
    arms += ["random:" + m["name"] for m in fam["members"]]
    if not smoke:
        arms += ["composed", "reference"]
    return arms


def arm_kind(arm):
    return arm.split(":")[0]


def generation_seconds(params_b, skills, share):
    """Seconds of greedy generation for the skills when the share of batches that run to the cap is `share`."""
    step = r21.step_seconds(params_b) * GEN_STEP_FACTOR
    total = 0.0
    for skill in skills:
        if skill in MAX_NEW_LONG:
            batches = math.ceil(FINAL_COUNTS[skill] / GEN_BATCH)
            total += batches * MAX_NEW_LONG[skill] * step * share
        else:
            total += FINAL_COUNTS[skill] * 0.012 * max(1.0, params_b / 7.0)
    return total


def make_plan(name, smoke=False, seeds=DEFAULT_SEEDS, caches_present=True):
    """Expected GPU time by part, in minutes, from a model of forward and generation cost. Not measured."""
    fam = cell_family(name, smoke)
    params_b = fam["params_b"]
    members = len(fam["members"])
    layers = LAYERS[name]
    skills = SMOKE_SKILLS if smoke else BENCH_SKILLS
    pass_s = r21.forward_seconds(params_b, r21.PROBE_TEXTS)
    one = 3.0 * pass_s
    joint = (members + 2.0) * pass_s
    parts = {}
    parts["load"] = 600.0 + 900.0 * members * params_b / 8.0 / 4.0
    parts["caches"] = 0.0 if caches_present else (members + 1) * 3 * 2 * pass_s + (members + 1) * params_b * 2.0 / r21.DISK_GBPS
    parts["calibrate"] = len(PROBE_SPLITS) * (members + 2.0) * pass_s
    parts["importance"] = members * (1 + 3 * layers) * (one + APPLY_LAYER_SECONDS)
    parts["minimal"] = members * (31 * (one + APPLY_SET_SECONDS))
    parts["compose"] = 0.0 if smoke else 24 * (joint + APPLY_SET_SECONDS)
    parts["controls"] = members * (seeds + 1 + len(SCRAMBLE_SEEDS) + 2) * (one + APPLY_SET_SECONDS)
    arms = arms_for(name, smoke)
    bench = 0.0
    for arm in arms:
        bench += generation_seconds(params_b, [s for s in skills if s in MAX_NEW_LONG], ARM_CAP_SHARE[arm_kind(arm)])
        bench += generation_seconds(params_b, [s for s in skills if s not in MAX_NEW_LONG], 1.0)
    parts["bench"] = bench
    total = sum(parts.values())
    return {"family": name, "cell": cell_name(name, smoke), "smoke": smoke, "arms": arms, "benchmarks": skills,
            "seeds": seeds, "minutes": {k: round(v / 60.0, 1) for k, v in parts.items()},
            "expected_minutes": round(total / 60.0, 1), "gpu_hours": round(total / 3600.0, 2),
            "gpu_hours_high": round(1.5 * total / 3600.0, 2), "caches_present": caches_present,
            "max_new_tokens": dict(MAX_NEW_LONG)}


def print_plan(plan):
    cell = plan["cell"]
    print(f"[{cell}] plan: {plan['expected_minutes']} min expected ({plan['gpu_hours']} H200 hours, up to "
          f"{plan['gpu_hours_high']} if the model is low), caches present {plan['caches_present']}", flush=True)
    print(f"[{cell}] plan minutes by part: {plan['minutes']}", flush=True)
    print(f"[{cell}] plan arms: {plan['arms']}; benchmarks: {plan['benchmarks']}", flush=True)


# ---------------------------------------------------------------- go and stop rules (no torch)


def member_go(size_share, share_c, random_share, member_score, set_score, skill="mbpp"):
    """The go rule for one member. Shares are fractions of the member's probe transfer on split c."""
    checks = {
        "size_at_most_half": bool(size_share is not None and size_share <= SIZE_GO),
        "keeps_90_percent_on_c": bool(share_c is not None and share_c >= TARGET_SHARE),
        "random_keeps_under_half": bool(random_share is not None and random_share < RANDOM_KEEP_MAX),
        "within_5_points_on_" + skill: bool(member_score is not None and set_score is not None
                                            and set_score >= member_score - BENCH_POINTS),
    }
    return {"checks": checks, "go": bool(all(checks.values()))}


def member_stop(size_share):
    """Stop rule: no set under 75 percent of the layers holds 90 percent of the transfer."""
    return {"size_share": size_share, "stop": bool(size_share is None or size_share > SIZE_STOP)}


# ---------------------------------------------------------------- rows on the volume


def emit(cell, message):
    print(f"[{cell}] {message}", flush=True)


def cell_dir(cell):
    return f"{OUT_ROOT}/{cell}"


def stage_file(cell, stage):
    return f"{cell_dir(cell)}/{stage.replace(':', '_')}.json"


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w") as handle:
        json.dump(payload, handle, default=str)
    os.replace(temp, path)


def save_stage(cell, stage, payload):
    """Write one stage row to the volume and commit, so a lost client loses nothing."""
    row = dict(payload)
    row.update({"cell": cell, "stage": stage, "fit_version": FIT_VERSION, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(stage_file(cell, stage), row)
    with open(f"{cell_dir(cell)}/rows.jsonl", "a") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
    try:
        volume.commit()
    except Exception as error:
        emit(cell, f"volume commit failed: {repr(error)[:200]}")
    emit(cell, f"stage {stage} saved to {stage_file(cell, stage)}")
    return row


def load_stage(cell, stage):
    path = stage_file(cell, stage)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def safe_reload(cell=""):
    """Reload the volume to see files other containers wrote. An open file blocks a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print(f"[{cell}] volume reload skipped: {repr(error)[:200]}", flush=True)


class Stages:
    """Run a stage unless its row exists (and --redo is off). A stage outside --only must already have a row."""

    def __init__(self, cell, only, redo):
        self.cell = cell
        self.only = only
        self.redo = redo

    def selected(self, stage):
        group = stage.split("_")[0].split(":")[0]
        return len(self.only) == 0 or group in self.only

    def get(self, stage, runner):
        saved = load_stage(self.cell, stage)
        if saved is not None and not (self.redo and self.selected(stage)):
            emit(self.cell, f"stage {stage}: using the saved row")
            return saved
        if not self.selected(stage):
            raise RuntimeError(f"stage {stage} is not selected and has no saved row in {cell_dir(self.cell)}")
        started = time.time()
        emit(self.cell, f"stage {stage}: start")
        result = runner()
        result["seconds"] = round(time.time() - started, 1)
        return save_stage(self.cell, stage, result)


# ---------------------------------------------------------------- scoring (run on Modal)


def check_probe_sources(fam, data):
    """Check that the pools the probe texts came from hold no reported benchmark."""
    from wsp import healdata

    used = set(["general"])
    for spec in fam["members"]:
        used.update(data["choices"][spec["name"]]["pools"])
    healdata.check_sources([source[0] for pool in sorted(used) for source in healdata._usable(pool)[0]])


def final_items(ev, skills, cell):
    """The benchmark items. They are read for scoring only; nothing is selected or trained on them."""
    items = {}
    for skill in skills:
        if skill == "mbpp":
            items[skill] = ev.mbpp_items()
        elif skill == "humaneval":
            items[skill] = ev.humaneval_items()
        elif skill == "gsm8k":
            items[skill] = ev.gsm8k_items(count=FINAL_COUNTS["gsm8k"], seed=0)
        elif skill == "mmlu":
            items[skill] = ev.mmlu_items(count=FINAL_COUNTS["mmlu"])
        else:
            raise ValueError(skill)
        emit(cell, f"items {skill}: {len(items[skill])}")
    if "mbpp" in items:
        assert len(items["mbpp"]) == FINAL_COUNTS["mbpp"], len(items["mbpp"])
    if "humaneval" in items:
        assert len(items["humaneval"]) == FINAL_COUNTS["humaneval"], len(items["humaneval"])
    return items


def score_skill(ev, model, tokenizer, skill, items, cell, arm):
    """Score one skill. Generative skills use the long token limit; two raw generations go to the log."""
    started = time.time()
    if skill not in MAX_NEW_LONG:
        result = ev.evaluate_skill(model, tokenizer, skill, items=items)
        emit(cell, f"score {arm} {skill}: {result['score']:.4f} ({result['correct']}/{result['n']}) "
                   f"in {round(time.time() - started, 1)} s")
        return {"score": result["score"], "correct": result["correct"], "n": result["n"]}
    cap = MAX_NEW_LONG[skill]
    prompts = [ev.render_prompt(tokenizer, ev.user_text(skill, item)) for item in items]
    outputs = ev.generate_texts(model, tokenizer, prompts, max_new_tokens=cap, batch_size=GEN_BATCH)
    records = ev.grade_outputs(skill, items, outputs)
    correct = sum(1 for record in records if record["correct"])
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in outputs]
    cut = sum(1 for n in lengths if n >= cap - 1)
    for index in range(min(RAW_SHOWN, len(outputs))):
        text = outputs[index]
        shown = text if len(text) <= RAW_CHARS else text[:RAW_CHARS // 2] + " ...[cut]... " + text[-RAW_CHARS // 2:]
        emit(cell, f"raw {arm} {skill} #{index} correct={records[index]['correct']} tokens={lengths[index]}: {shown!r}")
    emit(cell, f"score {arm} {skill}: {correct / len(records):.4f} ({correct}/{len(records)}) cap {cap}, "
               f"{cut} reached the cap, mean {sum(lengths) / len(lengths):.0f} tokens, "
               f"in {round(time.time() - started, 1)} s")
    return {"score": correct / len(records), "correct": correct, "n": len(records), "reached_cap": cut,
            "mean_new_tokens": sum(lengths) / len(lengths), "cap": cap}


def score_arm(ev, model, tokenizer, items, skills, cell, arm):
    scores = {}
    for skill in skills:
        scores[skill] = score_skill(ev, model, tokenizer, skill, items[skill], cell, arm)
    return scores


# ---------------------------------------------------------------- the cell (run on Modal)


def report_set(scorers, name, layers):
    """The set's probe numbers on every split: transfer, share of the full member, harm."""
    out = {}
    for split, scorer in scorers.items():
        full = scorer.full(name)
        result = scorer.score({name: layers})
        out[split] = {"transfer": result["transfer"][name], "share": scorer.share(result, name, full),
                      "harm": result["harm"], "full_transfer": full["transfer"][name], "full_harm": full["harm"]}
    return out


def run_cell(name, smoke=False, only="", seeds=DEFAULT_SEEDS, redo=False):
    import torch
    import transformers

    from wsp import layerset, probe, weave
    from wsp import skills_eval as ev

    started = time.time()
    cell = cell_name(name, smoke)
    fam = cell_family(name, smoke)
    cfg = FAMILIES[name]
    wanted = set(s for s in only.split(",") if s)
    for stage in wanted:
        if stage not in STAGE_GROUPS:
            raise ValueError(f"unknown stage {stage}; the stages are {', '.join(STAGE_GROUPS)}")
    stages = Stages(cell, wanted, redo)
    plan = make_plan(name, smoke, seeds)
    print_plan(plan)
    safe_reload(cell)
    os.makedirs(cell_dir(cell), exist_ok=True)
    seed_list = list(range(1, int(seeds) + 1))

    host_dir = r21.download(fam["host"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_dir)
    assert tokenizer.chat_template, f"{fam['host']}: no chat template; prompts would use a plain format"
    revisions = {repo: r21.revision_of(repo) for repo in [fam["host"]] + [m["id"] for m in fam["members"]]
                 + [cfg["reference"]]}

    # Teacher caches: round 21's files, same key. A missing file is built, then the model is freed.
    data = r21.probe_texts_for(tokenizer, fam)
    check_probe_sources(fam, data)
    caches = r21.build_probe_caches(fam, tokenizer, revisions, data)
    weave.memory_report(f"{cell} caches")
    stages.get("caches", lambda: {"revisions": revisions, "pools": data["choices"], "ranges": data["ranges"],
                                  "cache_dir": r21.CACHE_DIR, "texts_per_split": r21.PROBE_TEXTS,
                                  "plan": plan, "seeds": seed_list, "smoke": smoke})

    model = transformers.AutoModelForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    host_config = model.config
    members = []
    gates = {}
    for spec in fam["members"]:
        directory = r21.download(spec["id"])
        config = transformers.AutoConfig.from_pretrained(directory)
        mlp_only = weave.positional_differs(host_config, config)
        member = weave.member_from_directory(spec["name"], spec["skill"], directory, model, ref=None, ref_name="host",
                                             mlp_only=mlp_only, groups=("attn", "mlp"),
                                             meta={"id": spec["id"], "revision": revisions[spec["id"]]})
        gate = weave.gate_member(member, model, max_rel=r21.GATE_MAX_REL, min_cos=r21.GATE_MIN_COS)
        gate["mlp_only"] = mlp_only
        gates[spec["name"]] = gate
        emit(cell, f"member {spec['name']}: gate {gate}")
        if not gate["passed"]:
            raise RuntimeError(f"{cell}: member {spec['name']} fails the weave gate: {gate}")
        members.append(member)
    names = [m.name for m in members]
    weaver = layerset.layer_weaver(model, members, density_step=r21.DENSITY_STEP)
    n_layers = weaver.n_layers
    assert n_layers == LAYERS[name], n_layers
    weave.memory_report(f"{cell} weaver")

    # One probe objective per split (members and general text), calibrated on the host. Forward passes only.
    joint = {}
    baseline = {}
    for split in PROBE_SPLITS:
        member_caches = {n: caches["members"][n][split] for n in names}
        joint[split] = weave.ProbeObjective(names, member_caches, caches["host"][split]["general"],
                                            r21.DRIFT_TOLERANCE, r21.DRIFT_WEIGHT, r21.PROBE_BATCH)
        baseline[split] = joint[split].calibrate(model)
        emit(cell, f"calibrated split {split}: baseline KL {baseline[split]['baseline_kl']}, "
                   f"host nll {baseline[split]['host_nll']:.4f}")
    stages.get("gates", lambda: {"gate": gates, "baseline": baseline, "n_layers": n_layers,
                                 "members": names, "mlp_only": {m.name: m.mlp_only for m in members}})
    scorers = {split: {} for split in PROBE_SPLITS}
    for split in PROBE_SPLITS:
        for n in names:
            scorers[split][n] = layerset.LayerScorer(weaver, model, layerset.restrict_objective(joint[split], [n]))
        scorers[split]["joint"] = layerset.LayerScorer(weaver, model, joint[split])
    logger = lambda message: emit(cell, message)
    for split in PROBE_SPLITS:
        for scorer in scorers[split].values():
            scorer.log = logger if split == SELECT_SPLIT else None

    def by_split(who):
        return {split: scorers[split][who] for split in PROBE_SPLITS}

    # importance and minimal set, per member
    importance = {}
    minimal = {}
    for n in names:
        importance[n] = stages.get(
            f"importance_{n}", lambda n=n: layerset.layer_importance(scorers[SELECT_SPLIT][n], n))
        emit(cell, f"importance {n}: ranking {importance[n]['ranking']}")

        def run_minimal(n=n):
            found = layerset.minimal_set(scorers[SELECT_SPLIT][n], n, importance[n]["ranking"], TARGET_SHARE)
            found["report"] = report_set(by_split(n), n, found["layers"])
            found["stop"] = member_stop(found["size_share"])
            return found

        minimal[n] = stages.get(f"minimal_{n}", run_minimal)
        emit(cell, f"minimal {n}: set {minimal[n]['layers']} ({minimal[n]['size']} of {n_layers}), share on a "
                   f"{minimal[n]['share']}, on c {minimal[n]['report'][REPORT_SPLIT]['share']}, "
                   f"stop {minimal[n]['stop']['stop']}")
    norms = {n: importance[n]["norms"] for n in names}
    sets = {n: minimal[n]["layers"] for n in names}

    # composition (two members)
    composed = None
    if not smoke and len(names) > 1:
        def run_compose():
            tries = {}
            for resolve in ("norm", "try_both"):
                result = layerset.compose(scorers[SELECT_SPLIT]["joint"], sets, resolve=resolve, norms=norms)
                result["report"] = {}
                for split in PROBE_SPLITS:
                    probe_result = scorers[split]["joint"].score({n: result["assignment"][n] for n in names})
                    shares = {n: scorers[split][n].share(probe_result, n) for n in names}
                    result["report"][split] = {"transfer": probe_result["transfer"], "share": shares,
                                               "harm": probe_result["harm"], "value": probe_result["value"]}
                tries[resolve] = result
            chosen = max(tries, key=lambda r: (tries[r]["value"], r == "norm"))
            return {"tries": tries, "chosen": chosen, "assignment": tries[chosen]["assignment"],
                    "chosen_on": "probe value on split a"}

        composed = stages.get("compose", run_compose)
        emit(cell, f"compose: chosen {composed['chosen']}, assignment {composed['assignment']}, "
                   f"c {composed['tries'][composed['chosen']]['report'][REPORT_SPLIT]}")

    # controls, per member, on split c (never used to select)
    controls = {}
    sc = scorers[REPORT_SPLIT]
    for n in names:
        def run_controls(n=n):
            layers = sets[n]
            out = {"size": len(layers), "layers": layers, "seeds": seed_list}
            out["random"] = layerset.random_set_control(sc[n], n, len(layers), seed_list, reference=layers)
            out["norm_only"] = layerset.norm_set_control(sc[n], n, len(layers), norms=norms[n], reference=layers)
            out["scrambled"] = layerset.scrambled_control(sc[n], n, layers, SCRAMBLE_SEEDS)
            real = minimal[n]["report"][REPORT_SPLIT]
            out["real"] = real
            out["content_z"] = probe.content_z(real["transfer"], [r["transfer"] for r in out["random"]["runs"]])
            others = [m for m in names if m != n]
            if len(others) > 0:
                out["cross_skill"] = {m: layerset.cross_skill_control(sc[n], n, sets[m]) for m in others}
            return out

        controls[n] = stages.get(f"controls_{n}", run_controls)
        emit(cell, f"controls {n}: real share on c {controls[n]['real']['share']}, random mean "
                   f"{controls[n]['random']['share']['mean']}, norm only {controls[n]['norm_only']['share']}, "
                   f"scrambled {controls[n]['scrambled']['share']['mean']}, content z {controls[n]['content_z']}")
    assert layerset.host_matches(weaver, model), "the host drifted during selection"

    # benchmarks, once per arm
    skills = SMOKE_SKILLS if smoke else BENCH_SKILLS
    if len(wanted) == 0 or "bench" in wanted:
        arms = arms_for(name, smoke)
        todo = [arm for arm in arms if redo or load_stage(cell, "bench_" + arm) is None]
        if len(todo) > 0:
            items = final_items(ev, skills, cell)
            emit(cell, f"prompt sample: {ev.render_prompt(tokenizer, ev.user_text('mbpp', items['mbpp'][0]))!r}"[:700])
        for arm in todo:
            kind = arm_kind(arm)
            if kind == "reference":
                continue
            member = arm.split(":")[1] if ":" in arm else None
            index = names.index(member) if member is not None else None
            layers = None
            if kind == "member":
                weaver.load_member(model, index)
            elif kind == "minimal":
                layers = sets[member]
                weaver.apply(model, layerset.layer_genome(len(names), n_layers, {index: layers}))
            elif kind == "random":
                layers = layerset.random_layer_sets(n_layers, len(sets[member]), [RANDOM_BENCH_SEED])[0]
                weaver.apply(model, layerset.layer_genome(len(names), n_layers, {index: layers}))
            elif kind == "composed":
                assignment = {names.index(m): ls for m, ls in composed["assignment"].items()}
                layers = composed["assignment"]
                weaver.apply(model, layerset.layer_genome(len(names), n_layers, assignment))
            elif kind != "host":
                raise ValueError(arm)
            try:
                scores = score_arm(ev, model, tokenizer, items, skills, cell, arm)
            finally:
                weaver.restore(model)
            assert layerset.host_matches(weaver, model), f"the host drifted after arm {arm}"
            save_stage(cell, "bench_" + arm, {"arm": arm, "scores": scores, "layers": layers})
        if "reference" in todo:
            reference = transformers.AutoModelForCausalLM.from_pretrained(r21.download(cfg["reference"]),
                                                                         dtype=torch.bfloat16, device_map="cuda")
            reference.eval()
            try:
                scores = score_arm(ev, reference, tokenizer, items, skills, cell, "reference")
            finally:
                del reference
                weave.free_gpu()
            save_stage(cell, "bench_reference", {"arm": "reference", "repo": cfg["reference"],
                                                 "revision": revisions[cfg["reference"]], "scores": scores})

    summary = summarize(cell, name, smoke, names, minimal, controls, composed, skills)
    row = save_stage(cell, "summary", summary)
    row["seconds"] = round(time.time() - started, 1)
    return row


def summarize(cell, name, smoke, names, minimal, controls, composed, skills):
    """The table, from saved rows. A missing arm shows as None."""
    cfg = FAMILIES[name]
    arms = {}
    for arm in arms_for(name, smoke):
        saved = load_stage(cell, "bench_" + arm)
        arms[arm] = None if saved is None else {s: v["score"] for s, v in saved["scores"].items()}
    out = {"arms": arms, "members": {}}
    for n in names:
        skill = cfg["go_skill"][n]
        member_score = None if arms.get("member:" + n) is None else arms["member:" + n].get(skill)
        set_score = None if arms.get("minimal:" + n) is None else arms["minimal:" + n].get(skill)
        report_c = minimal[n]["report"][REPORT_SPLIT]
        out["members"][n] = {
            "set": minimal[n]["layers"], "size": minimal[n]["size"], "size_share": minimal[n]["size_share"],
            "share_c": report_c["share"], "harm_c": report_c["harm"],
            "random_share_c_mean": controls[n]["random"]["share"]["mean"], "content_z": controls[n]["content_z"],
            "skill": skill, "member_score": member_score, "set_score": set_score,
            "random_score": None if arms.get("random:" + n) is None else arms["random:" + n].get(skill),
            "probe_c": {"host": 0.0, "member": report_c["full_transfer"], "minimal": report_c["transfer"],
                        "random_seed1": controls[n]["random"]["runs"][0]["transfer"]},
            "stop": member_stop(minimal[n]["size_share"]),
            "go": member_go(minimal[n]["size_share"], report_c["share"], controls[n]["random"]["share"]["mean"],
                            member_score, set_score, skill)}
    if composed is not None and arms.get("composed") is not None:
        keeps = {}
        for n in names:
            skill = cfg["go_skill"][n]
            member_score = arms["member:" + n][skill] if arms.get("member:" + n) else None
            keeps[n] = None if member_score in (None, 0) else arms["composed"][skill] / member_score
        out["composed_keeps_of_member"] = keeps
        out["composed_at_95_percent"] = bool(all(v is not None and v >= COMPOSE_KEEP for v in keeps.values()))
    for line in ["arms " + json.dumps(arms)] + [f"member {n}: " + json.dumps(out["members"][n], default=str)
                                              for n in names]:
        emit(cell, line)
    return out


# ---------------------------------------------------------------- Modal function and entrypoint


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=24 * 3600)
def run_remote(name, smoke=False, only="", seeds=DEFAULT_SEEDS, redo=False):
    cell = cell_name(name, smoke)
    try:
        return run_cell(name, smoke, only, seeds, redo)
    except Exception as error:
        emit(cell, f"FAILED {repr(error)[:800]}")
        try:
            os.makedirs(cell_dir(cell), exist_ok=True)
            write_json(f"{cell_dir(cell)}/error.json", {"error": repr(error)[:2000], "time": time.time()})
            volume.commit()
        except Exception:
            pass
        raise


@app.local_entrypoint()
def main(family: str = "qwen7b", smoke: bool = False, only: str = "", seeds: int = DEFAULT_SEEDS,
         redo: bool = False, plan_only: bool = False):
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family}; the families are {', '.join(FAMILIES)}")
    cell = cell_name(family, smoke)
    plan = make_plan(family, smoke, seeds)
    print_plan(plan)
    if plan_only:
        return
    emit(cell, "launching; use modal run --detach so a lost client keeps the run")
    handle = run_remote.spawn(family, smoke, only, seeds, redo)
    try:
        row = handle.get()
    except Exception as error:
        emit(cell, f"cell failed: {repr(error)[:600]}")
        row = {"cell": cell, "status": "failed", "error": repr(error)[:600]}
    os.makedirs("results", exist_ok=True)
    with open(f"results/round25_{cell}.json", "w") as handle_out:
        json.dump(row, handle_out, indent=2, default=str)
    emit(cell, "done: " + json.dumps(row.get("members", row), default=str)[:1500])
