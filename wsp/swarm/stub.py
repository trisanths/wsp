"""STUB compile, fit, heal and eval (spec section 7). Plumbing, not a result.

Every record and result made here carries "stub": True, every log row from
the stub backend carries "backend": "stub", and every number from it must be
labeled STUB.

The stub is deterministic given (program, split, fidelity, stub seed). Its
shape follows Trisanth's round-2 sweep (results/README.md: host
Llama-3.2-1B-Instruct, donor Qwen2.5-Math-1.5B, 512-neuron MLP clusters from
donor layers 10, 18 and 24 at host layers 6, 10 and 14, full or rank-64 glue,
300 heal steps, 3 seeds, GSM8K search split), so stub comparisons point the
same way the real ones do:

    real (search split, mean of 3 seeds)    stub (expected, heal 300)
    host, no heal: 25.0%                    0.25
    host layer 6, rank-64 glue: 39.0%       about 0.39
    host layer 6, full glue: 34.3%          about 0.35
    host layers 10 and 14: 22 to 26%        about 0.25
    part minus random control: -1 point     0 (within noise)
    per-cell sd: 2.3 points (pooled)        eval noise sd 0.023

- Catalog: 40 graftable parts from four donors, in the real record shape,
  written to outputs/stub/catalog.jsonl (never to catalog.jsonl).
- Fit R squared depends only on the slot (donor model, donor layer, host
  depth `at`): high when at / L_host is near l / L_donor, lower for another
  family or a large hidden-size ratio, plus fixed noise. Two parts from one
  donor layer at one depth get the same fit, and a random-part control gets
  the fit of the part it replaces. Fit can veto a placement; it says nothing
  about a part's value, and nothing about which host depth scores best (in
  round 2 every cell had fit near its home depth, yet host layer 6 won).
- Placement dominates. The healed glue acts as an adapter whose gain depends
  only on the relative host depth of `at` and the glue rank, the same for a
  part and its random control: a bump peaking at layer 6 of 16, worth
  GLUE_PEAK_GAIN with rank-64 glue and FULL_GLUE_FACTOR of that with full
  glue. Far from the peak a graft gives back the heal-only gain, so it lands
  near the host, as layers 10 and 14 did.
- Part content. Single-layer parts (mlp_cluster, attn_heads, expert) add
  about nothing over their random control: each gets a fixed draw of sd
  SINGLE_LAYER_CONTENT_SD around 0. Only mlp_cluster was measured; attn_heads
  and expert are treated the same way (STUB ASSUMPTION).
- Blocks (multi-layer parts) carry a small real content effect, up to about
  +4 points for math-donor blocks, so a search can still find something.
  STUB ASSUMPTION: round 3 (whole donor blocks against a random-block
  control and a matched LoRA) replaces this with data.
- Heal-only (no graft, heal on) gains HEAL_GAIN_TARGET. STUB ASSUMPTION:
  round 2 had no heal-only arm; round 3 adds it.
- No heal block: the glue's W_out stays 0, so the model equals the host
  exactly. With heal steps s, c = 1 - exp(-s / 100). Content strength per
  graft is fit + (1 - fit) * c * fit (300 steps closes most of the gap).
  Unhealed glue hurts: each graft costs harm * (1 - c), so heal 0 hurts.
- Cascade short heal: below full fidelity the env heals with config
  low_fidelity_heal_steps (50 by default) and the backend scores the steps
  it actually ran. 50 steps give c = 0.39 (300 give 0.95), so each graft's
  strength, glue gain and leftover harm sit between the unhealed and the
  fully healed values, and the score lands between the heal-0 and the
  full-heal scores. STUB ASSUMPTION: no short heal has been measured yet;
  the curve shape is the same c(s) used for 0, 100, 300 and 600 steps.
- Eval: base + heal-only gain + the best graft's glue gain (glue gains do
  not stack across grafts, a STUB ASSUMPTION since round 2 only had single
  grafts) + sum of content * strength - harm - interference for grafts at the
  same or adjacent depth - a guard cost for many grafts, plus noise (more at
  low fidelity). Many weak grafts can collapse the guard.
- A random-part control keeps the glue, heal and harm, with content near 0.
- Cost model (reported as compute_s, never slept): compile 10 s + 15 s per
  graft (25 s for one graft, the round-2 median), probe = compile, heal 0.5 s
  per step per graft, eval 120 s times the fidelity fraction.
"""

import hashlib
import json
import math
import os
import random

from wsp.swarm import program_schema as ps

STUB_HOST = {"model": "unsloth/Llama-3.2-1B-Instruct", "family": "llama3.2", "layers": 16, "hidden": 2048}

# block_value: the content scale of the donor's block parts per task (at full
# strength, before the math depth profile). Single-layer parts do not use it.
STUB_DONORS = {
    "Qwen/Qwen2.5-Math-1.5B": {"short": "qmath", "family": "qwen2.5", "layers": 28, "hidden": 1536,
                               "intermediate": 8960, "block_value": {"gsm8k": 0.06, "if_guard": -0.010},
                               "trained_on": "GSM8K train activations"},
    "Qwen/Qwen2.5-Coder-1.5B": {"short": "qcoder", "family": "qwen2.5", "layers": 28, "hidden": 1536,
                                "intermediate": 8960, "block_value": {"gsm8k": 0.03, "if_guard": -0.005},
                                "trained_on": "code and GSM8K activations"},
    "meta-llama/Llama-3.2-3B-Instruct": {"short": "llama3b", "family": "llama3.2", "layers": 28, "hidden": 3072,
                                         "intermediate": 8192, "block_value": {"gsm8k": 0.015, "if_guard": 0.010},
                                         "trained_on": "instruction following activations"},
    "allenai/OLMoE-1B-7B-0924": {"short": "olmoe", "family": "olmoe", "layers": 16, "hidden": 2048,
                                 "intermediate": 1024, "block_value": {"gsm8k": 0.0, "if_guard": 0.0},
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

KIND_HARM = {"mlp_cluster": 1.0, "attn_heads": 1.0, "expert": 1.0, "block": 1.5}

# Host scores. gsm8k search is run 1's host with no heal (25.0%). The host's
# held-out score is not measured yet, so held-out uses the same value.
STUB_BASE = {
    "search": {"gsm8k": 0.25, "if_guard": 0.70},
    "heldout": {"gsm8k": 0.25, "if_guard": 0.68},
}
UNKNOWN_TASK_BASE = 0.25

# Placement: the glue as an adapter (round 2, see the module docstring).
GLUE_BEST_DEPTH = 6 / 16.0     # host layer 6 of 16 was best of the measured 6, 10 and 14
GLUE_DEPTH_WIDTH = 2 / 16.0    # STUB guess: only three depths were measured
GLUE_PEAK_GAIN = 0.13          # rank-64 glue at layer 6: 39.0% = host 25% + heal-only gain + this, healed
FULL_GLUE_FACTOR = 0.7         # full glue at layer 6: 34.3%, about 0.7 of rank 64's gain over heal-only
HEAL_GAIN_TARGET = 0.02        # heal-only gain at full heal (STUB ASSUMPTION; the controls subtract it)
OFF_PEAK_GLUE_COST = HEAL_GAIN_TARGET  # far from layer 6 a graft lands back at the host (layers 10, 14)
HEAL_GAIN_GUARD = -0.005

# Content (beyond the random control).
SINGLE_LAYER_CONTENT_SD = 0.005  # round 2: no selection method beat the random control
HINT_NOISE = 0.02                # the signature's effect hint is the content plus this noise

TARGET_HARM = 0.06        # per unhealed graft, on target tasks
GUARD_HARM = 0.10         # per unhealed graft, on guard tasks
GUARD_WEAK_HARM = 0.06    # per graft, times (1 - strength), on guard tasks
SAME_DEPTH_PENALTY = 0.03
ADJACENT_DEPTH_PENALTY = 0.015
MANY_GRAFTS_GUARD = 0.02  # per graft beyond 2
EVAL_NOISE = 0.023        # round 2 pooled within-cell sd over 3 seeds (24 cells)

# Cost model, in seconds. Reported, never slept.
COMPILE_BASE_S = 10.0
COMPILE_PER_GRAFT_S = 15.0     # one graft: 25 s, the round-2 median compile
HEAL_S_PER_STEP = 0.5          # per step per graft
EVAL_S = 120.0                 # one full-fidelity eval


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


def glue_depth_profile(host_layer, n_host):
    """How good a host depth is for the glue as an adapter: 1.0 at layer 6 of 16, near 0 by layer 10."""
    rel = host_layer / float(n_host)
    return math.exp(-((rel - GLUE_BEST_DEPTH) / GLUE_DEPTH_WIDTH) ** 2)


def glue_rank_factor(glue_rank):
    """The glue gain of a rank, relative to rank-64 glue.

    Full glue (None) and rank 64 are measured. Any other rank is a STUB guess
    and behaves like rank 64.
    """
    if glue_rank is None:
        return FULL_GLUE_FACTOR
    return 1.0


def glue_params(d_host, d_donor, glue_rank):
    """Parameters in w_in plus w_out: 2 * d_host * d_donor at full rank, 2 * r * (d_host + d_donor) at rank r."""
    if glue_rank is None:
        return 2 * d_host * d_donor
    return 2 * glue_rank * (d_host + d_donor)


class StubWorld:
    """The hidden truth behind the stub: part content, sizes, the fit surface and the glue."""

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
        values = {}
        for task in sorted(donor["block_value"]):
            values[task] = self.content_value(kind, donor, task, home, rng)
        hints = {}
        for task in sorted(values):
            hints[task] = round(values[task] + rng.gauss(0.0, HINT_NOISE), 4)
        return {"id": part_id, "model": donor_model, "kind": kind, "layer": layer, "home": home,
                "units": self.unit_count(kind, rng), "values": values, "hints": hints}

    def content_value(self, kind, donor, task, home, rng):
        """What a part's own units add on a task at full strength, beyond its random control.

        Single-layer parts: a fixed draw around 0 (round 2: part selection did
        not beat the random control). Blocks: a small effect scaled by the
        donor's block_value and, for gsm8k, the math depth profile
        (STUB ASSUMPTION until round 3).
        """
        if kind != "block":
            return rng.gauss(0.0, SINGLE_LAYER_CONTENT_SD)
        scale = donor["block_value"][task]
        if task == "gsm8k":
            scale = scale * math_depth_profile(home)
        return scale * rng.uniform(0.5, 1.0)

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
            rank = ps.graft_glue_rank(graft)
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
                "glue_params": glue_params(STUB_HOST["hidden"], donor["hidden"], rank),
                "glue_rank": rank,
                "fit": "ridge",
                "fit_r2": self.fit_r2(part["model"], donor_layer, at, n_host),
                "fit_pairs": 4096,
                "stub": True,
            })
        return entries

    # Heal and eval.

    def heal_closeness(self, steps):
        """c in [0, 1): how far a heal of this many steps closes the gap (0 steps or no heal: 0)."""
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

    def glue_gain(self, host_layer, n_host, glue_rank, c):
        """What one healed graft's glue adds as an adapter at this depth, for any part or control."""
        profile = glue_depth_profile(host_layer, n_host)
        gain = GLUE_PEAK_GAIN * glue_rank_factor(glue_rank) * profile
        cost = OFF_PEAK_GLUE_COST * (1.0 - profile)
        return c * (gain - cost)

    def best_glue_gain(self, report, n_host, c):
        """The largest glue gain over the grafts (gains do not stack). 0.0 with no grafts."""
        best = None
        for entry in report:
            gain = self.glue_gain(entry["host_layer"], n_host, entry.get("glue_rank"), c)
            if best is None or gain > best:
                best = gain
        if best is None:
            return 0.0
        return best

    def content(self, graft, part, task):
        """The content a graft adds on a task at full strength: the part's, or about 0 for a control."""
        if ps.graft_control(graft) is not None:
            seed_text = "control|" + str(self.seed) + "|" + part["id"] + "|" + str(ps.graft_control_seed(graft))
            return fixed_gauss(seed_text + "|" + task, 0.003)
        return part["values"].get(task, 0.0)

    def expected_scores(self, program, report, split, tasks, guards, n_host=None):
        """Noise-free task scores for a compiled program (the stub's hidden truth).

        report is graft_report() for the program. n_host defaults to the stub
        host's 16 layers.
        """
        base = self.base_scores(split, tasks)
        steps = ps.heal_steps(program)
        if steps is None:
            # W_out stays 0 without a heal pass, so the model is the host exactly.
            return dict(base)
        if n_host is None:
            n_host = STUB_HOST["layers"]
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
                score += self.best_glue_gain(report, n_host, c)
            for index in range(len(grafts)):
                graft = grafts[index]
                part = self.parts[ps.graft_id(graft)]
                fit = report[index]["fit_r2"]
                strength = fit + (1.0 - fit) * c * fit
                score += self.content(graft, part, task) * strength
                if is_guard:
                    score -= GUARD_HARM * KIND_HARM[part["kind"]] * (1.0 - c)
                    score -= GUARD_WEAK_HARM * (1.0 - strength)
                else:
                    score -= TARGET_HARM * KIND_HARM[part["kind"]] * (1.0 - c)
            score -= self.interference(report)
            if is_guard and len(grafts) > 2:
                score -= MANY_GRAFTS_GUARD * (len(grafts) - 2)
            out[task] = score
        return out

    def scores(self, program, report, split, tasks, fidelity, fraction, program_hash, guards, n_host=None):
        """Task scores for a compiled program: expected_scores plus fixed eval noise, clamped to [0, 1]."""
        expected = self.expected_scores(program, report, split, tasks, guards, n_host)
        if ps.heal_steps(program) is None:
            return expected
        sigma = EVAL_NOISE / math.sqrt(fraction)
        out = {}
        for task in tasks:
            noise_text = "eval|" + str(self.seed) + "|" + program_hash + "|" + split + "|" + fidelity + "|" + task
            score = expected[task] + fixed_gauss(noise_text, sigma)
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
    return COMPILE_BASE_S + COMPILE_PER_GRAFT_S * len(ps.grafts(program))


def heal_cost(program):
    steps = ps.heal_steps(program)
    if steps is None or steps <= 0:
        return 0.0
    count = len(ps.grafts(program))
    if count < 1:
        count = 1
    return HEAL_S_PER_STEP * steps * count


def eval_cost(fraction):
    return EVAL_S * fraction


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
