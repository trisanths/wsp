"""STUB compile, fit, heal and eval (spec section 7). Plumbing, not a result.

Every record and result made here carries "stub": True, every log row from
the stub backend carries "backend": "stub", and every number from it must be
labeled STUB.

The stub is deterministic given (program, split, fidelity, stub seed) and has
enough structure that reading signatures, fit and eval history beats random:

- Catalog: 40 graftable parts from four donors, in the real record shape,
  written to outputs/stub/catalog.jsonl (never to catalog.jsonl).
- Fit R squared depends only on the slot (donor model, donor layer, host
  depth `at`): high when at / L_host is near l / L_donor, lower for another
  family or a large hidden-size ratio, plus fixed noise. Two parts from one
  donor layer at one depth get the same fit, and a random-part control gets
  the fit of the part it replaces. Fit can veto a placement; it says nothing
  about a part's value.
- Each part has a hidden value per task. It shows only after heal and eval.
  Its signature carries a noisy hint of that value (Person B's probe).
- No heal block: the glue's W_out stays 0, so the model equals the host
  exactly. With heal steps s, c = 1 - exp(-s / 100). Strength per graft is
  fit + (1 - fit) * c * fit (300 steps closes most of the gap). Unhealed
  glue hurts: each graft costs harm * (1 - c), so heal 0 hurts.
- Eval: base + heal-only gain (capped, rises with steps) + sum of
  value * strength - harm - interference for grafts at the same or adjacent
  depth - a guard cost for many grafts, plus noise of about 0.01 (more at
  low fidelity). Many weak grafts can collapse the guard.
- A random-part control keeps the glue, heal and harm, with value near 0.
- Cost model (reported as compute_s, never slept): compile 20 s + 15 s per
  graft, probe = compile, heal 0.5 s per step per graft, eval 120 s times
  the fidelity fraction.
"""

import hashlib
import json
import math
import os
import random

from wsp.swarm import program_schema as ps

STUB_HOST = {"model": "unsloth/Llama-3.2-1B-Instruct", "family": "llama3.2", "layers": 16, "hidden": 2048}

STUB_DONORS = {
    "Qwen/Qwen2.5-Math-1.5B": {"short": "qmath", "family": "qwen2.5", "layers": 28, "hidden": 1536,
                               "intermediate": 8960, "value": {"gsm8k": 0.10, "if_guard": -0.010},
                               "trained_on": "GSM8K train activations"},
    "Qwen/Qwen2.5-Coder-1.5B": {"short": "qcoder", "family": "qwen2.5", "layers": 28, "hidden": 1536,
                                "intermediate": 8960, "value": {"gsm8k": 0.05, "if_guard": -0.005},
                                "trained_on": "code and GSM8K activations"},
    "meta-llama/Llama-3.2-3B-Instruct": {"short": "llama3b", "family": "llama3.2", "layers": 28, "hidden": 3072,
                                         "intermediate": 8192, "value": {"gsm8k": 0.03, "if_guard": 0.020},
                                         "trained_on": "instruction following activations"},
    "allenai/OLMoE-1B-7B-0924": {"short": "olmoe", "family": "olmoe", "layers": 16, "hidden": 2048,
                                 "intermediate": 1024, "value": {"gsm8k": 0.06, "if_guard": 0.0},
                                 "trained_on": "GSM8K expert routing statistics"},
}

# (donor model, kind, layer or [first, last], unit tag). 40 parts.
STUB_PART_PLAN = [
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 6, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 8, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 10, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 12, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 14, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 14, "c1"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 14, "c2"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 16, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 18, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "mlp_cluster", 20, "c0"),
    ("Qwen/Qwen2.5-Math-1.5B", "attn_heads", 10, "h0"),
    ("Qwen/Qwen2.5-Math-1.5B", "attn_heads", 14, "h0"),
    ("Qwen/Qwen2.5-Math-1.5B", "block", [10, 12], "blk"),
    ("Qwen/Qwen2.5-Math-1.5B", "block", [16, 18], "blk"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 8, "c0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 12, "c0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 12, "c1"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 16, "c0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 20, "c0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "mlp_cluster", 24, "c0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "attn_heads", 12, "h0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "attn_heads", 18, "h0"),
    ("Qwen/Qwen2.5-Coder-1.5B", "block", [8, 10], "blk"),
    ("Qwen/Qwen2.5-Coder-1.5B", "block", [14, 16], "blk"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlp_cluster", 6, "c0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlp_cluster", 10, "c0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlp_cluster", 14, "c0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlp_cluster", 18, "c0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "mlp_cluster", 22, "c0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "attn_heads", 8, "h0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "attn_heads", 16, "h0"),
    ("meta-llama/Llama-3.2-3B-Instruct", "block", [12, 14], "blk"),
    ("meta-llama/Llama-3.2-3B-Instruct", "block", [20, 22], "blk"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 4, "e3"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 6, "e11"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 8, "e17"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 8, "e40"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 10, "e5"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 12, "e22"),
    ("allenai/OLMoE-1B-7B-0924", "expert", 14, "e9"),
]

KIND_VALUE = {"mlp_cluster": 1.0, "attn_heads": 0.7, "expert": 1.0, "block": 1.6}
KIND_HARM = {"mlp_cluster": 1.0, "attn_heads": 1.0, "expert": 1.0, "block": 1.5}

STUB_BASE = {
    "search": {"gsm8k": 0.30, "if_guard": 0.70},
    "heldout": {"gsm8k": 0.32, "if_guard": 0.68},
}
UNKNOWN_TASK_BASE = 0.30
TARGET_HARM = 0.06        # per unhealed graft, on target tasks
GUARD_HARM = 0.10         # per unhealed graft, on guard tasks
GUARD_WEAK_HARM = 0.06    # per graft, times (1 - strength), on guard tasks
HEAL_GAIN_TARGET = 0.02   # heal-only gain at full heal (the controls subtract this)
HEAL_GAIN_GUARD = -0.005
SAME_DEPTH_PENALTY = 0.03
ADJACENT_DEPTH_PENALTY = 0.015
MANY_GRAFTS_GUARD = 0.02  # per graft beyond 2
EVAL_NOISE = 0.01


def seed_int(text):
    """A stable integer from a string (independent of PYTHONHASHSEED)."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15], 16)


def fixed_gauss(text, sigma):
    """Deterministic gaussian noise keyed by a string."""
    return random.Random(seed_int(text)).gauss(0.0, sigma)


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def part_id_for(donor_model, kind, layer, tag):
    short = STUB_DONORS[donor_model]["short"]
    if isinstance(layer, list):
        return "stub_" + short + "_l" + str(layer[0]) + "-" + str(layer[1]) + "_" + tag
    return "stub_" + short + "_l" + str(layer) + "_" + tag


def first_layer(layer):
    if isinstance(layer, list):
        return layer[0]
    return layer


def math_depth_profile(home):
    """Math skill sits mostly in the middle of a donor: 1.0 at mid depth, 0.2 at the ends."""
    return clamp(1.0 - abs(home - 0.5) * 1.6, 0.2, 1.0)


class StubWorld:
    """The hidden truth behind the stub: part values, sizes and the fit surface."""

    def __init__(self, stub_seed=0):
        self.seed = stub_seed
        self.parts = {}
        for donor_model, kind, layer, tag in STUB_PART_PLAN:
            part_id = part_id_for(donor_model, kind, layer, tag)
            self.parts[part_id] = self.make_part(part_id, donor_model, kind, layer)

    def make_part(self, part_id, donor_model, kind, layer):
        donor = STUB_DONORS[donor_model]
        rng = random.Random(seed_int("part|" + str(self.seed) + "|" + part_id))
        home = first_layer(layer) / float(donor["layers"])
        dud = rng.random() < 0.25
        values = {}
        for task in sorted(donor["value"]):
            scale = donor["value"][task] * KIND_VALUE[kind]
            if task == "gsm8k":
                scale = scale * math_depth_profile(home)
            value = scale * rng.uniform(0.4, 1.0)
            if dud and task == "gsm8k":
                value = value * 0.1
            values[task] = value
        hints = {}
        for task in sorted(values):
            hints[task] = round(values[task] + rng.gauss(0.0, 0.02), 4)
        return {"id": part_id, "model": donor_model, "kind": kind, "layer": layer, "home": home,
                "units": self.unit_count(kind, rng), "values": values, "hints": hints, "dud": dud}

    def unit_count(self, kind, rng):
        if kind == "mlp_cluster":
            return rng.choice([64, 128, 256])
        if kind == "attn_heads":
            return rng.choice([2, 4])
        if kind == "expert":
            return 1
        return 3

    def records(self):
        """The public catalog records, in the real record shape, sorted by id."""
        out = []
        for part_id in sorted(self.parts):
            part = self.parts[part_id]
            record = {"id": part_id, "type": part["kind"], "model": part["model"]}
            if isinstance(part["layer"], list):
                record["layers"] = list(part["layer"])
            else:
                record["layer"] = part["layer"]
            record["units"] = "stub/units/" + part_id + ".pt"
            record["trained_on"] = "STUB: " + STUB_DONORS[part["model"]]["trained_on"]
            record["signature"] = {"effect": dict(part["hints"]), "fit_r2": {}}
            record["stub"] = True
            out.append(record)
        return out

    # Fit.

    def fit_r2(self, donor_model, donor_layer, at, n_host):
        """R squared of the ridge map host[at] -> donor[layer]. Depends only on the slot."""
        donor = STUB_DONORS[donor_model]
        home = donor_layer / float(donor["layers"])
        rel = at / float(n_host)
        distance = abs(rel - home)
        depth = math.exp(-(distance / 0.10) ** 2)
        family = 1.0
        if donor["family"] != STUB_HOST["family"]:
            family = 0.85
        ratio = max(donor["hidden"], STUB_HOST["hidden"]) / float(min(donor["hidden"], STUB_HOST["hidden"]))
        size = 1.0 / (1.0 + 0.2 * math.log(ratio, 2))
        noise = fixed_gauss("fit|" + str(self.seed) + "|" + donor_model + "|" + str(donor_layer) + "|" + str(at), 0.03)
        return round(clamp(depth * family * size + noise, 0.0, 0.99), 4)

    def part_params(self, part):
        donor = STUB_DONORS[part["model"]]
        d = donor["hidden"]
        if part["kind"] == "mlp_cluster":
            return 3 * d * part["units"]
        if part["kind"] == "attn_heads":
            head_dim = 128
            return 2 * d * part["units"] * head_dim + 2 * d * head_dim
        if part["kind"] == "expert":
            return 3 * d * donor["intermediate"]
        first = part["layer"][0]
        last = part["layer"][1]
        return (last - first + 1) * 12 * d * d

    def graft_report(self, program, n_host, default_depth):
        """Per-graft compile report entries, in the real compile_report.json shape.

        default_depth(part_id) returns the default host depth of a part.
        Raises KeyError for a part the stub world does not know.
        """
        entries = []
        for graft in ps.grafts(program):
            part_id = ps.graft_id(graft)
            if part_id not in self.parts:
                raise KeyError(part_id)
            part = self.parts[part_id]
            donor = STUB_DONORS[part["model"]]
            at = ps.graft_at(graft)
            if at is None:
                at = default_depth(part_id)
            donor_layer = first_layer(part["layer"])
            entries.append({
                "id": part_id,
                "kind": part["kind"],
                "control": ps.graft_control(graft),
                "donor": part["model"],
                "donor_layer": donor_layer,
                "host_layer": at,
                "d_host": STUB_HOST["hidden"],
                "d_donor": donor["hidden"],
                "part_params": self.part_params(part),
                "glue_params": 2 * STUB_HOST["hidden"] * donor["hidden"],
                "fit": "ridge",
                "fit_r2": self.fit_r2(part["model"], donor_layer, at, n_host),
                "fit_pairs": 4096,
                "stub": True,
            })
        return entries

    # Heal and eval.

    def heal_closeness(self, steps):
        if steps is None or steps <= 0:
            return 0.0
        return 1.0 - math.exp(-steps / 100.0)

    def heal_loss(self, program):
        steps = ps.heal_steps(program)
        if steps is None or steps <= 0:
            return []
        c = self.heal_closeness(steps)
        losses = []
        points = 8
        for i in range(points + 1):
            fraction = i / float(points)
            losses.append(round(2.4 - 0.6 * c * (1.0 - math.exp(-4.0 * fraction)), 4))
        return losses

    def base_scores(self, split, tasks):
        table = STUB_BASE.get(split, STUB_BASE["search"])
        out = {}
        for task in tasks:
            out[task] = table.get(task, UNKNOWN_TASK_BASE)
        return out

    def scores(self, program, report, split, tasks, fidelity, fraction, program_hash, guards):
        """Task scores for a compiled program. report is graft_report() for it."""
        base = self.base_scores(split, tasks)
        steps = ps.heal_steps(program)
        if steps is None:
            # W_out stays 0 without a heal pass, so the model is the host exactly.
            return dict(base)
        c = self.heal_closeness(steps)
        grafts = ps.grafts(program)
        out = {}
        for task in tasks:
            is_guard = task in guards
            score = base[task]
            if is_guard:
                score += HEAL_GAIN_GUARD * c
            else:
                score += HEAL_GAIN_TARGET * c
            for index in range(len(grafts)):
                graft = grafts[index]
                entry = report[index]
                part = self.parts[ps.graft_id(graft)]
                fit = entry["fit_r2"]
                strength = fit + (1.0 - fit) * c * fit
                if ps.graft_control(graft) is not None:
                    seed_text = "control|" + str(self.seed) + "|" + part["id"] + "|" + str(ps.graft_control_seed(graft))
                    value = fixed_gauss(seed_text + "|" + task, 0.003)
                else:
                    value = part["values"].get(task, 0.0)
                score += value * strength
                if is_guard:
                    score -= GUARD_HARM * KIND_HARM[part["kind"]] * (1.0 - c)
                    score -= GUARD_WEAK_HARM * (1.0 - strength)
                else:
                    score -= TARGET_HARM * KIND_HARM[part["kind"]] * (1.0 - c)
            score -= self.interference(report)
            if is_guard and len(grafts) > 2:
                score -= MANY_GRAFTS_GUARD * (len(grafts) - 2)
            sigma = EVAL_NOISE / math.sqrt(fraction)
            noise_text = "eval|" + str(self.seed) + "|" + program_hash + "|" + split + "|" + fidelity + "|" + task
            score += fixed_gauss(noise_text, sigma)
            out[task] = round(clamp(score, 0.0, 1.0), 4)
        return out

    def interference(self, report):
        """Penalty for grafts at the same or adjacent host depth."""
        penalty = 0.0
        for i in range(len(report)):
            for j in range(i + 1, len(report)):
                gap = abs(report[i]["host_layer"] - report[j]["host_layer"])
                if gap == 0:
                    penalty += SAME_DEPTH_PENALTY
                elif gap == 1:
                    penalty += ADJACENT_DEPTH_PENALTY
        return penalty


# Cost model, in seconds. Reported, never slept.

def compile_cost(program):
    return 20.0 + 15.0 * len(ps.grafts(program))


def heal_cost(program):
    steps = ps.heal_steps(program)
    if steps is None or steps <= 0:
        return 0.0
    count = len(ps.grafts(program))
    if count < 1:
        count = 1
    return 0.5 * steps * count


def eval_cost(fraction):
    return 120.0 * fraction


def write_stub_catalog(path, stub_seed=0):
    """Write the stub catalog JSONL to path (under outputs/stub/). Return the path.

    Writes to a temp file first, then replaces, so parallel runs never see a
    half-written file. Refuses to write the repo's catalog.jsonl.
    """
    if os.path.basename(path) == "catalog.jsonl" and os.path.basename(os.path.dirname(os.path.abspath(path))) != "stub":
        raise ValueError("the stub catalog must live under outputs/stub/, not " + path)
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    world = StubWorld(stub_seed)
    temp_path = path + "." + str(os.getpid()) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        for record in world.records():
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temp_path, path)
    return path
