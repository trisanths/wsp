"""The studio's offline backend: real library parts, scored by a simulator, with no GPU.

OfflineCompileEval replaces the swarm's StubCompileEval for the "stub" backend. It reads the same compiler
catalog the Modal backend uses, so every real part and every real deletion mask compiles. The cost model, heal
curve, eval noise and the glue and interference terms come from wsp/swarm/stub.py. What changes is where a
part's effect comes from. It is built from the part's library record:

- its skill tags matched to the task (a math part helps GSM8K, a law part does about nothing),
- its status (a rejected part is slightly harmful, a validated one helps a little more),
- its measured score against its control, when the library has one,
- its donor's shape (layer count and hidden size) for the fit and the part size.

Everything is deterministic for a seed. Nothing here is a measurement, so the API keeps "stub": true and the
web app shows an "Estimate" badge.
"""

import math
import random
import re

from wsp.swarm import factory
from wsp.swarm import program_schema as ps
from wsp.swarm import space, stub
from wsp.swarm.backends import StubCompileEval
from wsp.swarm.config import guard_tasks

from studio.server import catalog as catalog_mod
from studio.server import model_profiles
from studio.server import library
from studio.server import sizing

DEFAULT_DONOR_LAYERS = 28
DEFAULT_DONOR_HIDDEN = 2048
DEFAULT_UNITS = 512
HOST_FAMILY = stub.STUB_HOST["family"]

MATCH_GAIN = 0.03          # a fully matching part adds about this on its task at full strength
MEASURED_WEIGHT = 0.5      # share of the measured edge over control that carries into the estimate
REJECTED_SHIFT = -0.012    # a rejected part is slightly harmful
VALIDATED_SHIFT = 0.005
CONTENT_NOISE = 0.004
MASK_GUARD_COST = 0.003

MATH_TASKS = {"gsm8k": {"gsm8k": 1.0, "multilingual_math": 0.8, "two_digit_addition": 0.7, "arithmetic": 0.7,
                        "math": 0.55, "mathematics_theory": 0.5}}
SUBJECTS = ["law", "medicine", "chemistry", "physics", "biology", "history", "finance", "computer_science",
            "philosophy_ethics", "mathematics_theory", "economics", "psychology", "business", "engineering"]
KNOWLEDGE = {"mmlu": 1.0, "knowledge": 0.8, "facts": 0.8, "general": 0.4}
for _subject in SUBJECTS:
    KNOWLEDGE[_subject] = 0.6
CODE = {"code": 1.0, "sql": 0.5, "tool_calling": 0.4}
for _name in ["code_cpp", "code_csharp", "code_java", "code_php", "code_rust", "code_shell", "code_typescript"]:
    CODE[_name] = 0.8
REASONING = {"reasoning": 1.0, "decision": 0.5, "logic": 0.7, "general": 0.3}
TASK_SKILLS = {
    "gsm8k": MATH_TASKS["gsm8k"],
    "math": dict(MATH_TASKS["gsm8k"], math=1.0),
    "mmlu": KNOWLEDGE,
    "mmlu_pro": KNOWLEDGE,
    "humaneval": CODE,
    "mbpp": CODE,
    "bbeh_mini": REASONING,
    "if_guard": {"instruction": 1.0, "general": 0.3},
    "ifeval": {"instruction": 1.0, "general": 0.3},
}
MATH_FAMILY = ["gsm8k", "math", "mmlupro:math"]

AGAINST = re.compile(r"([-+]?\d*\.?\d+) against ([-+]?\d*\.?\d+)")


def clamp(value, low, high):
    return max(low, min(high, value))


def skill_weights(task):
    """{skill tag: weight} for a task. An unknown task matches its own name only."""
    if task in TASK_SKILLS:
        return TASK_SKILLS[task]
    if str(task).startswith("mmlupro:"):
        category = str(task).split(":", 1)[1].replace(" ", "_")
        if category == "math":
            return dict(KNOWLEDGE, **dict(MATH_TASKS["gsm8k"], math=1.0))
        weights = dict(KNOWLEDGE)
        weights[category] = 1.0
        if category == "computer_science":
            weights.update(CODE)
        return weights
    return {str(task): 1.0}


def skill_match(task, skills, model):
    """How well a part's skill tags and donor fit a task, from 0 to 1."""
    weights = skill_weights(task)
    best = 0.0
    for tag in skills or []:
        best = max(best, weights.get(str(tag), 0.0))
    if task in MATH_FAMILY or task in MATH_TASKS:
        if "math" in str(model).lower():
            best = min(1.0, best + 0.25)
    return best


def measured_edge(evidence):
    """The measured score minus its control from an evidence line such as "... : 0.41 against 0.4033 for ...", or None."""
    found = AGAINST.search(str(evidence or ""))
    if found is None:
        return None
    try:
        return float(found.group(1)) - float(found.group(2))
    except ValueError:
        return None


def depth_factor(kind, home):
    """Math skill sits in the middle of a donor, so a mid-depth part is worth more than one at either end."""
    return 0.6 + 0.4 * stub.math_depth_profile(home)


def part_content(part, task, seed):
    """What a library part adds on a task at full strength, from its library record. Deterministic."""
    match = skill_match(task, part["skills"], part["model"])
    value = MATCH_GAIN * match
    edge = part["edge"]
    if edge is not None and match > 0:
        value += clamp(edge, -0.05, 0.05) * MEASURED_WEIGHT * max(match, 0.3)
    if match > 0 and task in MATH_FAMILY:
        value *= depth_factor(part["kind"], part["home"])
    status = part["status"]
    if status == "validated":
        value = value * 1.2 + VALIDATED_SHIFT * match
    elif status == "rejected":
        value = value * 0.3 + REJECTED_SHIFT
    value += stub.fixed_gauss("offline|" + str(seed) + "|" + part["id"] + "|" + task, CONTENT_NOISE)
    return value


def entry_for(record):
    """The library entry behind a compiler record, or None."""
    return library.by_catalog_id().get(record["id"])


class PartTable(dict):
    """{part id: part}, built from the compiler catalog the first time a part is asked for."""

    def __init__(self, world):
        super().__init__()
        self.world = world

    def __missing__(self, part_id):
        part = self.world.make_offline_part(part_id)
        self[part_id] = part
        return part


class OfflineWorld(stub.StubWorld):
    """The simulator's hidden truth, built from the real compiler catalog instead of made-up parts."""

    def __init__(self, stub_seed, config, catalog):
        self.seed = stub_seed
        self.config = config
        self.catalog = catalog
        self.models = model_profiles.merged(library.get()["models"])
        self.entries = library.by_catalog_id()
        self.size_ctx = sizing.SizeContext(config, catalog, catalog_mod.host_params_b(config["host"]))
        self.parts = PartTable(self)

    # Donor facts.

    def donor_layers(self, model):
        info = self.models.get(model) or {}
        return int(info.get("layers") or DEFAULT_DONOR_LAYERS)

    def donor_hidden(self, model):
        info = self.models.get(model) or {}
        return int(info.get("hidden") or DEFAULT_DONOR_HIDDEN)

    def donor_family(self, model):
        info = self.models.get(model) or {}
        return str(info.get("family") or str(model).split("/")[-1].split("-")[0].lower())

    # Parts.

    def make_offline_part(self, part_id):
        record = self.catalog.records.get(part_id)
        if record is None or record.get("type") == "deletion_mask":
            raise KeyError(part_id)
        entry = self.entries.get(part_id)
        layer = record.get("layers")
        if layer is None:
            layer = record.get("layer")
        first = stub.first_layer(layer) if layer is not None else 0
        model = record.get("model")
        kind = record.get("type")
        if kind == "inserted_layer":
            kind = "block"
        skills = []
        status = "candidate"
        edge = None
        if entry is not None:
            skills = list(entry.get("skills") or [])
            status = entry.get("status") or "candidate"
            edge = measured_edge(entry.get("evidence"))
        home = first / float(self.donor_layers(model))
        return {"id": part_id, "model": model, "kind": kind, "layer": layer, "home": home, "skills": skills,
                "status": status, "edge": edge, "record": record}

    def part_params(self, part):
        found = sizing.part_size_b(part["id"], self.size_ctx)
        if found is not None and found[0] > 0:
            return int(found[0] * 1e9)
        d = self.donor_hidden(part["model"])
        if part["kind"] == "block":
            layers = part["layer"]
            count = 1
            if isinstance(layers, list) and len(layers) > 1:
                count = max(1, int(layers[-1]) - int(layers[0]) + 1)
            return count * 12 * d * d
        return 3 * d * DEFAULT_UNITS

    def default_depth(self, part, n_host):
        """The host depth round(l / L_donor * L_host), clamped."""
        depth = int(round(part["home"] * n_host))
        return max(0, min(n_host - 1, depth))

    def fit_r2(self, donor_model, donor_layer, at, n_host):
        """R squared of the ridge map host[at] -> donor[layer]. Depends only on the slot."""
        home = donor_layer / float(self.donor_layers(donor_model))
        rel = at / float(n_host)
        depth = math.exp(-((abs(rel - home)) / 0.10) ** 2)
        family = 1.0
        if self.donor_family(donor_model) != HOST_FAMILY:
            family = 0.85
        hidden = self.donor_hidden(donor_model)
        ratio = max(hidden, stub.STUB_HOST["hidden"]) / float(min(hidden, stub.STUB_HOST["hidden"]))
        size = 1.0 / (1.0 + 0.2 * math.log(ratio, 2))
        noise = stub.fixed_gauss("fit|" + str(self.seed) + "|" + str(donor_model) + "|" + str(donor_layer) + "|" +
                                 str(at), 0.03)
        return round(clamp(depth * family * size + noise, 0.0, 0.99), 4)

    def graft_report(self, program, n_host, default_depth=None):
        """Per-graft compile report entries in the real compile_report.json shape. KeyError for an unknown part."""
        entries = []
        for graft in ps.grafts(program):
            part_id = ps.graft_id(graft)
            part = self.parts[part_id]
            at = ps.graft_at(graft)
            if at is None:
                at = self.default_depth(part, n_host)
            donor_layer = stub.first_layer(part["layer"]) if part["layer"] is not None else 0
            rank = ps.graft_glue_rank(graft)
            d_donor = self.donor_hidden(part["model"])
            entries.append({
                "id": part_id,
                "kind": part["kind"],
                "control": ps.graft_control(graft),
                "donor": part["model"],
                "donor_layer": donor_layer,
                "host_layer": at,
                "d_host": stub.STUB_HOST["hidden"],
                "d_donor": d_donor,
                "part_params": self.part_params(part),
                "glue_params": stub.glue_params(stub.STUB_HOST["hidden"], d_donor, rank),
                "glue_rank": rank,
                "fit": "ridge",
                "fit_r2": self.fit_r2(part["model"], donor_layer, at, n_host),
                "fit_pairs": 4096,
                "stub": True,
            })
        return entries

    # Content.

    def content(self, graft, part, task):
        """The content a graft adds on a task at full strength: from the part's record, or about 0 for a control."""
        if ps.graft_control(graft) is not None:
            seed_text = "control|" + str(self.seed) + "|" + part["id"] + "|" + str(ps.graft_control_seed(graft))
            return stub.fixed_gauss(seed_text + "|" + task, 0.003)
        return part_content(part, task, self.seed)

    def mask_effect(self, mask_id, task, is_guard):
        """A real mask moves the host a little: a small cost on guard tasks, noise on the others."""
        if is_guard:
            return -MASK_GUARD_COST
        return stub.fixed_gauss("mask|" + str(self.seed) + "|" + mask_id + "|" + task, 0.003)

    def expected_scores(self, program, report, split, tasks, guards, n_host=None):
        """Noise-free task scores for a compiled program, as StubWorld.expected_scores with record-based content."""
        base = self.base_scores(split, tasks)
        steps = ps.heal_steps(program)
        if steps is None:
            return dict(base)
        if n_host is None:
            n_host = stub.STUB_HOST["layers"]
        c = self.heal_closeness(steps)
        grafts = ps.grafts(program)
        masks = ps.mask_ids(program)
        out = {}
        for task in tasks:
            is_guard = task in guards
            score = base[task]
            if is_guard:
                score += stub.HEAL_GAIN_GUARD * c
            else:
                score += stub.HEAL_GAIN_TARGET * c
                score += self.best_glue_gain(report, n_host, c)
            for index in range(len(grafts)):
                graft = grafts[index]
                part = self.parts[ps.graft_id(graft)]
                fit = report[index]["fit_r2"]
                strength = fit + (1.0 - fit) * c * fit
                score += self.content(graft, part, task) * strength
                harm = stub.KIND_HARM.get(part["kind"], 1.0)
                if is_guard:
                    score -= stub.GUARD_HARM * harm * (1.0 - c)
                    score -= stub.GUARD_WEAK_HARM * (1.0 - strength)
                else:
                    score -= stub.TARGET_HARM * harm * (1.0 - c)
            for mask_id in masks:
                score += self.mask_effect(mask_id, task, is_guard)
            score -= self.interference(report)
            if is_guard and len(grafts) > 2:
                score -= stub.MANY_GRAFTS_GUARD * (len(grafts) - 2)
            out[task] = score
        return out


class OfflineCompileEval(StubCompileEval):
    """The "stub" backend of the studio: StubCompileEval with an OfflineWorld on the real compiler catalog."""

    name = "stub"

    def __init__(self, config, catalog):
        self.config = config
        self.catalog = catalog
        self.world = OfflineWorld(config["stub_seed"], config, catalog)
        self.n_host = space.host_layers(config, catalog)
        self.guards = guard_tasks(config)

    def compile(self, program, fit_only=False):
        phase_time = stub.compile_cost(program)
        try:
            report = self.world.graft_report(program, self.n_host)
        except KeyError as error:
            return {"ok": False, "error": "The compiler catalog has no part " + str(error), "handle": None,
                    "graft_report": [], "compute_s": {"compile": phase_time}, "stub": True}
        for mask_id in ps.mask_ids(program):
            if mask_id not in self.catalog.records:
                return {"ok": False, "error": "The compiler catalog has no mask " + str(mask_id), "handle": None,
                        "graft_report": [], "compute_s": {"compile": phase_time}, "stub": True}
        handle = {"program": ps.normalize(program), "graft_report": report,
                  "program_hash": space.program_hash(program)}
        return {"ok": True, "error": None, "handle": handle, "graft_report": report,
                "compute_s": {"compile": phase_time}, "stub": True}


def register():
    """Point the factory's "stub" backend at OfflineCompileEval. The wsp package is not changed."""
    factory.CHOICES["backend"]["stub"] = ("studio.server.offline_backend", "OfflineCompileEval")


def component_effect(record, tasks, seed=0):
    """The offline estimate of what one catalog record adds on the first target task, or None when unknown.

    The builder's per-part verdicts use it, so a verdict agrees with the score the same part gets.
    """
    entry = entry_for(record)
    if entry is None or not tasks:
        return None
    layer = record.get("layers")
    if layer is None:
        layer = record.get("layer")
    first = stub.first_layer(layer) if layer is not None else 0
    info = model_profiles.merged(library.get()["models"]).get(record.get("model")) or {}
    home = first / float(info.get("layers") or DEFAULT_DONOR_LAYERS)
    edge = measured_edge(entry.get("evidence"))
    part = {"id": record["id"], "model": record.get("model"), "kind": record.get("type"), "home": home,
            "skills": list(entry.get("skills") or []), "status": entry.get("status") or "candidate", "edge": edge}
    return part_content(part, tasks[0], seed)
