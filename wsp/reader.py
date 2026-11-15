"""Pure helpers for round 20c, the reader study.

Question: what makes a base able to use what a written-in donor block
computes. These helpers hold the parts of that study that need no GPU:
block and entry geometry, the matched trainable count of the distillation
control, checkpointed probes during a heal, and the go rule.

Nothing here trains on benchmark data. Probe texts come from wsp.healdata
pools, held out from every heal text, and the caller picks them.
"""

import contextlib
import time

import torch

from wsp import probe

REL_START = 8 / 28
CHECKPOINTS = (0, 250, 500, 1000, 2000, 4000, 8000)
ENTRIES = ("mid", "top", "feedback")
SCOPES = ("after", "all")
TOP_LEFT = 2


# ---------------------------------------------------------------- geometry


def block_range(donor_layers, size, rel_start=REL_START):
    """[first, last] donor layers of a block of `size` layers that starts at the same relative depth.

    The start is round(rel_start * donor_layers), moved down when the block would run past the last layer.
    """
    if size < 1 or size > donor_layers:
        raise ValueError(f"block size {size} does not fit {donor_layers} donor layers")
    first = int(round(rel_start * donor_layers))
    if first + size > donor_layers:
        first = donor_layers - size
    return [first, first + size - 1]


def mid_host_layer(first, donor_layers, host_layers):
    """Host layer at the matching relative depth (the rule of wsp.assemble.default_host_layer)."""
    k = int(round(first / donor_layers * host_layers))
    if k < 0:
        k = 0
    if k > host_layers - 1:
        k = host_layers - 1
    return k


def entry_layer(entry, first, donor_layers, host_layers):
    """The host layer where the block output re-enters, or for "feedback" the host layer it reads.

    mid: the matching depth. top: host_layers - 1 - TOP_LEFT, so two host layers are left to read the output.
    feedback: the read layer, one below the matching depth (the output goes to the next token's input).
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}, got {entry!r}")
    mid = mid_host_layer(first, donor_layers, host_layers)
    if entry == "mid":
        return mid
    if entry == "top":
        return host_layers - 1 - TOP_LEFT
    return max(0, mid - 1)


def lora_layer_list(scope, seam_layer, host_layers):
    """Host layers that get the reader LoRA: after the seam, or all of them."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    if scope == "all":
        return list(range(host_layers))
    return list(range(seam_layer + 1, host_layers))


def glue_params(d_host, d_donor):
    """Trainable parameters of one linear branch: T_in, T_out, and the token gate (weight and bias)."""
    return 2 * d_host * d_donor + d_host + 1


def trainable_target(d_host, d_donor, per_rank_layers, rank):
    """Trainable count of a real arm: glue plus a LoRA of `rank` on the layers that params_per_rank counted."""
    return glue_params(d_host, d_donor) + rank * per_rank_layers


def matched_rank(per_rank_all, target):
    """LoRA rank on all layers whose count is closest to target from below, at least 1."""
    rank = int(target) // int(per_rank_all)
    if rank < 1:
        rank = 1
    return rank


# ---------------------------------------------------------------- checkpoints


def checkpoint_steps(total, wanted=CHECKPOINTS):
    """Heal steps (updates done) at which to probe. Every wanted step below total; the final state is probed by the caller."""
    out = []
    for step in wanted:
        if 0 <= step < total and step not in out:
            out.append(int(step))
    return sorted(out)


class StepHook:
    """Calls fn(step) once, just before the update that follows `step` finished updates, for each step in steps."""

    def __init__(self, steps, fn):
        self.pending = set(int(s) for s in steps)
        self.fn = fn
        self.fired = []

    def maybe_fire(self, step):
        """Fire fn(step) when step is still pending. Returns True when it fired."""
        if step in self.pending:
            self.pending.discard(step)
            self.fired.append(step)
            self.fn(step)
            return True
        return False

    def wrap(self, next_loss):
        def wrapped(step):
            self.maybe_fire(step)
            return next_loss(step)

        return wrapped


class ForwardCounter:
    """Drives a StepHook from a heal loop that has no callback but calls a forward function once per step.

    tick() goes at the top of that forward function. The call index is the number of finished updates. The module
    is put in eval mode for the probe and returned to its mode after.
    """

    def __init__(self, hook, module=None):
        self.hook = hook
        self.module = module
        self.calls = 0

    def tick(self):
        index = self.calls
        self.calls += 1
        if index not in self.hook.pending:
            return False
        was_training = self.module.training if self.module is not None else False
        if self.module is not None:
            self.module.eval()
        try:
            return self.hook.maybe_fire(index)
        finally:
            if was_training:
                self.module.train()


@contextlib.contextmanager
def hooked_stage2(synth_module, hook):
    """Fire `hook` during the stage-2 heal of wsp.synth, with no change to wsp/synth.py.

    wsp.synth has no step callback. This replaces synth.run_heal for stage 2 only, and inside that call replaces
    synth.optimize_groups with a version that wraps next_loss. Both are put back on exit. It also covers
    synth.lora_baseline, which calls run_heal as stage 2.
    """
    original_run = synth_module.run_heal
    original_opt = synth_module.optimize_groups

    def run_heal(model, tokenizer, comps, branches, donor_map, data, stage, s, steps):
        if stage != 2:
            return original_run(model, tokenizer, comps, branches, donor_map, data, stage, s, steps)

        def opt(groups, n_steps, next_loss, warmup_frac=0.05, clip=1.0):
            return original_opt(groups, n_steps, hook.wrap(next_loss), warmup_frac, clip)

        synth_module.optimize_groups = opt
        try:
            return original_run(model, tokenizer, comps, branches, donor_map, data, stage, s, steps)
        finally:
            synth_module.optimize_groups = original_opt

    synth_module.run_heal = run_heal
    try:
        yield hook
    finally:
        synth_module.run_heal = original_run
        synth_module.optimize_groups = original_opt


@contextlib.contextmanager
def lora_on_all_layers(synth_module):
    """Make the stage-2 LoRA of wsp.synth cover every host layer, not only the layers after the seam.

    wsp.synth fixes the LoRA layers at first seam + 1 .. last. This replaces synth.apply_lora with a wrapper that
    passes all layers, and puts it back on exit.
    """
    original = synth_module.apply_lora

    def apply_all(model, rank, *args, layers=None, **kwargs):
        from wsp.assemble import host_layer_count

        return original(model, rank, *args, layers=list(range(host_layer_count(model))), **kwargs)

    synth_module.apply_lora = apply_all
    try:
        yield
    finally:
        synth_module.apply_lora = original


# ---------------------------------------------------------------- probe


class ContextView:
    """The frozen base inside a healed model, as a callable for wsp.probe.

    context_factory(model) returns a context manager under which the model equals the base
    (wsp.synth.frozen_base for a compiled model, wsp.synth.lora_off for a LoRA-only one).
    """

    def __init__(self, model, context_factory):
        self.model = model
        self.context_factory = context_factory

    def parameters(self):
        return self.model.parameters()

    def __call__(self, input_ids=None, attention_mask=None):
        with self.context_factory(self.model):
            return self.model(input_ids=input_ids, attention_mask=attention_mask)


def probe_point(model, base_view, donor, tokenizer, texts, skills, cache, max_length=384, batch_size=4):
    """One probe of a written model on held-out pool texts.

    texts: {name: [text, ...]}. skills: names scored as transfer toward the donor (KL(donor || model) against
    KL(donor || base), 0 is the base and 1 is the donor). Every other name in texts is scored as drift,
    KL(base || model) in nats. cache holds the base-side KLs, computed once per cell.
    Returns {"transfer": {name: {...}}, "drift": {name: {...}}}.
    """
    out = {"transfer": {}, "drift": {}}
    for name in texts:
        if name in skills:
            if name not in cache:
                cache[name] = probe.logit_kl(donor, base_view, tokenizer, texts[name], max_length, batch_size)
            base_kl = cache[name]
            written = probe.logit_kl(donor, model, tokenizer, texts[name], max_length, batch_size)
            out["transfer"][name] = {"kl_donor_base": base_kl["kl"], "kl_donor_written": written["kl"],
                                     "agree_base": base_kl["agree"], "agree_written": written["agree"],
                                     "transfer": probe.gap_closed(base_kl["kl"], written["kl"], 0.0),
                                     "tokens": written["tokens"]}
        else:
            out["drift"][name] = probe.drift(model, base_view, tokenizer, texts[name], max_length, batch_size)
    return out


def timed_point(point_fn, heal_steps):
    """Run point_fn() and stamp the result with the heal step count and the probe seconds."""
    start = time.time()
    point = point_fn()
    point["heal_steps"] = int(heal_steps)
    point["probe_seconds"] = round(time.time() - start, 2)
    return point


def mean_transfer_of(point, skills=None):
    """Mean transfer over the skills of one probe point (skills with no gap left out), or None."""
    values = []
    for name, entry in point["transfer"].items():
        if skills is not None and name not in skills:
            continue
        if entry.get("transfer") is not None:
            values.append(entry["transfer"])
    if len(values) == 0:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------- rows and the go rule


def find_rows(rows, pair=None, config=None, arm=None):
    out = []
    for row in rows:
        if pair is not None and row.get("pair") != pair:
            continue
        if config is not None and row.get("config") != config:
            continue
        if arm is not None and row.get("arm") != arm:
            continue
        out.append(row)
    return out


def values_of(rows, getter):
    out = []
    for row in rows:
        value = getter(row)
        if value is not None:
            out.append(float(value))
    return out


def spread_of(values):
    if len(values) < 2:
        return None
    return max(values) - min(values)


def mean_of(values):
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def transfer_of(row):
    return row.get("transfer")


def score_of(name):
    def getter(row):
        return (row.get("scores") or {}).get(name)

    return getter


def code_best(row):
    """HumanEval and MBPP of a row, as a pair (humaneval, mbpp), each None when missing."""
    scores = row.get("scores") or {}
    return scores.get("humaneval"), scores.get("mbpp")


def go_check(rows, pair, config="default", transfer_floor=0.0, code_points=0.03, mmlu_points=0.01):
    """The go rule for one config of one pair.

    A component is read when the real block beats kd_lora and the scrambled control on probe transfer by more than
    the two-seed spread, and beats both on HumanEval or MBPP by more than code_points (3 points), with MMLU within
    mmlu_points (1 point) of the base. The spread is the largest two-seed spread among the real arm and the two
    controls of this config; when this config has one seed only, the spread of the default config is used and the
    result says so (spread_source). Controls missing for this config fall back to the default config's controls
    and the result is marked approximate. Returns a dict with "read" (True, False, or None for not enough rows).
    """
    real = find_rows(rows, pair, config, "real")
    out = {"pair": pair, "config": config, "read": None, "approximate": False, "missing": []}
    if len(real) == 0:
        out["missing"].append("real")
        return out
    controls = {}
    for arm in ("scrambled", "kd_lora"):
        found = find_rows(rows, pair, config, arm)
        if len(found) == 0 and config != "default":
            found = find_rows(rows, pair, "default", arm)
            if len(found) > 0:
                out["approximate"] = True
        if len(found) == 0:
            out["missing"].append(arm)
        controls[arm] = found
    base = find_rows(rows, pair, None, "base")
    if len(base) == 0:
        out["missing"].append("base")
    if len(out["missing"]) > 0:
        return out
    real_transfer = values_of(real, transfer_of)
    spreads = []
    out["spread_source"] = "own seeds"
    if len(real) >= 2 or config == "default":
        groups = (real, controls["scrambled"], controls["kd_lora"])
    else:
        groups = (find_rows(rows, pair, "default", "real"), find_rows(rows, pair, "default", "scrambled"),
                  find_rows(rows, pair, "default", "kd_lora"))
        out["spread_source"] = "default config seeds"
    for group in groups:
        spread = spread_of(values_of(group, transfer_of))
        if spread is not None:
            spreads.append(spread)
    if len(spreads) == 0:
        out["missing"].append("two-seed spread")
        return out
    spread = max(spreads)
    real_mean = mean_of(real_transfer)
    out["spread"] = spread
    out["transfer_real"] = real_mean
    transfer_ok = True
    for arm in ("scrambled", "kd_lora"):
        control = mean_of(values_of(controls[arm], transfer_of))
        out["transfer_" + arm] = control
        if real_mean is None or control is None or real_mean - control <= spread or real_mean <= transfer_floor:
            transfer_ok = False
    code_ok = False
    for name in ("humaneval", "mbpp"):
        real_score = mean_of(values_of(real, score_of(name)))
        beats = True
        for arm in ("scrambled", "kd_lora"):
            control = mean_of(values_of(controls[arm], score_of(name)))
            if real_score is None or control is None or real_score - control <= code_points:
                beats = False
        out[name + "_real"] = real_score
        out[name + "_beats_controls_by_margin"] = beats
        base_score = mean_of(values_of(base, score_of(name)))
        out[name + "_base"] = base_score
        if real_score is not None and base_score is not None:
            out[name + "_over_base"] = real_score - base_score
        if beats:
            code_ok = True
    real_mmlu = mean_of(values_of(real, score_of("mmlu")))
    base_mmlu = mean_of(values_of(base, score_of("mmlu")))
    mmlu_ok = real_mmlu is not None and base_mmlu is not None and abs(real_mmlu - base_mmlu) <= mmlu_points
    out["mmlu_real"] = real_mmlu
    out["mmlu_base"] = base_mmlu
    out["transfer_ok"] = transfer_ok
    out["code_ok"] = code_ok
    out["mmlu_ok"] = mmlu_ok
    out["read"] = bool(transfer_ok and code_ok and mmlu_ok)
    first = real[0]
    out["params_added"] = first.get("params_added")
    out["size"] = (first.get("cfg") or {}).get("size")
    out["steps"] = (first.get("cfg") or {}).get("steps")
    return out


def read_summary(rows, pair):
    """Go check of every config of a pair, the smallest block and shortest heal that read, and the params added there."""
    configs = []
    for row in find_rows(rows, pair, None, "real"):
        if row["config"] not in configs:
            configs.append(row["config"])
    checks = []
    for config in configs:
        checks.append(go_check(rows, pair, config))
    passing = [c for c in checks if c["read"]]
    smallest = None
    shortest = None
    for c in passing:
        if c.get("size") is not None and (smallest is None or c["size"] < smallest["size"]):
            smallest = c
        if c.get("steps") is not None and (shortest is None or c["steps"] < shortest["steps"]):
            shortest = c
    return {"pair": pair, "checks": checks, "read_configs": [c["config"] for c in passing],
            "smallest_block": None if smallest is None else {"config": smallest["config"], "size": smallest["size"],
                                                             "params_added": smallest["params_added"]},
            "shortest_heal": None if shortest is None else {"config": shortest["config"], "steps": shortest["steps"],
                                                            "params_added": shortest["params_added"]}}


def pick_best(rows, pair, configs):
    """The config among `configs` whose real rows have the highest mean final transfer. None when no row has one."""
    best = None
    best_value = None
    for config in configs:
        value = mean_of(values_of(find_rows(rows, pair, config, "real"), transfer_of))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best = config
            best_value = value
    return best
