"""The composer's tools: atlas, catalog, language_reference, compile_and_practice, finish.

These are the only tools the LLM gets. None of them reads or writes raw
weight values: atlas and catalog return facts about models and mined
parts (sizes, layers, families, structure), and sanitize() removes any
long numeric list. compile_and_practice sends a program to Modal, which
compiles it with wsp (stitches and a heal are the only training, on
wsp.healdata pools) and returns the benchmark-free probe of wsp/probe.py
on held-out pool text: transfer per skill against that skill's donor, and
drift on general text. No tool shows a score measured on a reported
benchmark (rule of 2026-11-14), and the benchmarks score the final model
once, in the final report. The request's limits are checked on an estimate
before the compile and on the measured numbers after it. A program with taps
is rejected: a tap is not Sapien.
"""

import hashlib
import json
import os

from composer import sizes
from composer import probes
from composer.skills import (DEFAULT_DRIFT_TOLERANCE, HEAL_DATA, PROBE_COUNT, SET_NOTES, SKILLS, check_heal_sources,
                             probe_summary)
from composer.sources import sanitize
from wsp.program import Program

# Rule 2 (owner, 2026-11-14): the composer must not see a score whose source is a reported benchmark.
# A score shows only when its name starts with a probe or pool-text prefix and its note names no benchmark.
ALLOWED_SCORE_PREFIXES = ["probe_", "pool_", "kl_", "nll_", "ppl_"]
# Words that name a reported benchmark, from healdata.BANNED and the benchmarks of composer/skills.py.
BENCHMARK_WORDS = ["gsm8k", "math", "mbpp", "humaneval", "medmcqa", "medqa", "mmlu", "arc", "bbh", "bigbench",
                   "mgsm", "legalbench", "casehold", "fast-decisions", "benchmark", "test", "held-out", "heldout"]
# Atlas entry keys built from benchmark screens: skill scores and the layer and expert maps drawn from them.
ATLAS_HIDDEN_KEYS = ["skills", "skill_details", "layer_maps", "layer_map", "expert_maps"]
# Signature keys of a catalog part that are structure facts, not scores.
STRUCTURE_SIGNATURE_KEYS = ["num_deleted", "deleted_per_layer", "top_overlaps", "overlap_with", "num_units",
                            "layers", "d_model", "heads", "num_heads", "num_experts", "params"]

CATALOG_KEYS = ["id", "type", "model", "base", "family", "layer", "layers", "num_units", "d_model", "label",
                "trained_on", "source"]
GRAFT_TYPES = ["block", "mlp_cluster", "attn_heads", "expert"]


def program_hash(program_dict):
    text = json.dumps(program_dict, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def tool_specs():
    """Function tool definitions for the OpenAI Responses API."""
    return [
        {
            "type": "function",
            "name": "atlas",
            "description": ("Rows from the atlas of open models: sizes (params_b, active_params_b, layers, "
                            "hidden), tokenizer family, and other structure facts. Scores measured on a "
                            "reported benchmark are never shown; a score shows only when it is a probe or "
                            "pool-text measure. Filter by skill "
                            "(math, code, medicine, knowledge, law) and/or by a model id substring."),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill": {"type": ["string", "null"], "description": "skill name, or null for all"},
                    "model": {"type": ["string", "null"], "description": "model id substring, or null for all"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "catalog",
            "description": ("Mined parts with their signatures: MLP clusters, attention heads, experts, "
                            "deletion masks, task vectors. Use a part in a program by its id. Any donor "
                            "layer range also works as a graft id: block/<org>/<model>/<first>-<last>. "
                            "The experts and insert statements take whole donor models from atlas(), "
                            "not catalog ids."),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": ["string", "null"], "description": "part type filter, or null"},
                    "model": {"type": ["string", "null"], "description": "model id substring, or null"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "language_reference",
            "description": "The full reference for the wsp program language (docs/language.md) plus composer notes.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "type": "function",
            "name": "compile_and_practice",
            "description": ("Parse a Sapien program, check it against the request's limits, compile it on "
                            "Modal (stitches and a heal are the only training, on approved pool text), and "
                            "run the benchmark-free probe on held-out pool text. The request's limits become "
                            "the program's budget. Returns per skill the transfer against that skill's donor "
                            "(0 is the base, 1 is the donor), the drift on general text, practice_objective "
                            "(mean transfer over the practiced skills minus a drift penalty), "
                            "request_objective (all request skills, an unpracticed skill as 0; the loop ranks "
                            "programs by it), params, decode tok/s, compile seconds, and errors. A compile costs one unit of the compile budget; a parse error or a "
                            "constraint rejection costs nothing, and an identical program returns its cached "
                            "result for free."),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_text": {"type": "string", "description": "the program in wsp text syntax"},
                    "practice_tasks": {"type": ["array", "null"], "items": {"type": "string"},
                                       "description": "skills to probe; null means all request skills"},
                },
                "required": ["program_text"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "finish",
            "description": ("Submit the final program. It must be a program you compiled and practiced within "
                            "the constraints. Benchmarks score it after this, once."),
            "parameters": {
                "type": "object",
                "properties": {
                    "program_text": {"type": "string"},
                    "rationale": {"type": "string", "description": "why this program, in a few sentences"},
                },
                "required": ["program_text", "rationale"],
                "additionalProperties": False,
            },
        },
    ]


def composer_notes():
    lines = [
        "",
        "## Composer notes",
        "",
        "- You can write every Sapien statement except `tap` and `tap_train`: grafts (with `gate on`),",
        "  `experts`, `insert`, `lora`, `heal` (with `distill`), `mask`, `tv`, `from`, `carve`, `splice`,",
        "  and `budget`. A program with a tap is rejected, because a tap runs a source model at inference.",
        "- A program holds one kind of added component: grafts, or experts, or inserts.",
        "- The request's limits are added to your program as its budget. A `budget` line of your own can only",
        "  tighten them. The compile checks the measured size and heal time again.",
        "- The model may grow past the base when the limits allow it. Total params and params run per token",
        "  are separate limits.",
        "- Graft a whole donor layer range with the id block/<org>/<model>/<first>-<last>, for example",
        "  `graft block/Qwen/Qwen2.5-7B-Instruct/8-20 at 7 rank 64 stitch mlp objective donor_lm steps 800`.",
        "  The model must be in the atlas, so its size is known.",
        "- `experts DONOR1,DONOR2 top_k 1` and `insert DONOR layers A..B at N` take whole donor models from the",
        "  atlas.",
        "- `heal ... distill TEACHER` needs a teacher with the host's tokenizer family. The teacher must be in the",
        "  atlas. A host with only a LoRA and a distillation heal is a baseline and is rejected.",
        "- Heal and carve data names: " + ", ".join(HEAL_DATA) + ". Each is approved pool text (wsp.healdata);",
        "  composer_mix is an even mix of math, code, medicine, and general. No benchmark data trains anything.",
        "- compile_and_practice returns the probe, not a benchmark accuracy. Transfer is the share of the gap",
        "  between the base and the skill's donor that your model closes on held-out pool text (0 is the base,",
        "  1 is the donor). The donor of a skill is the source model of the components you used for it; a skill",
        "  with no component has transfer 0. Drift is the KL from the frozen base on general text. The objective",
        "  is the mean transfer minus a penalty when drift passes the request's drift_tolerance.",
        "- heal and each stitch have their own step count, and each must be at or under the cap.",
        "- A host with only a LoRA and heal is the fine-tuning baseline, so it is rejected. A LoRA beside",
        "  composed parts is fine.",
        "- The final report scores your program against the base, the base with a LoRA, the base with a",
        "  distillation heal from each source with the host's tokenizer, your program with `control random` on",
        "  every component, and, when the size limit is far above the base, the best existing model of that size.",
        "  All arms get the same heal data and budget. Components that beat their random control carry the skill.",
        "  The benchmarks score the final program and these arms once, after finish.",
    ]
    return "\n".join(lines)


def matches(text, pattern):
    if pattern is None or pattern == "":
        return True
    if text is None:
        return False
    return pattern.lower() in str(text).lower()


def compact_signature(signature):
    """Keep a signature short: the top three mask overlaps, and no per-layer overlap tables."""
    if not isinstance(signature, dict):
        return sanitize(signature)
    out = {}
    for key in signature:
        if key == "overlap_with" and isinstance(signature[key], dict):
            pairs = []
            for other in signature[key]:
                item = signature[key][other]
                if isinstance(item, dict) and is_float(item.get("jaccard")):
                    pairs.append((item["jaccard"], other))
            pairs.sort(reverse=True)
            top = []
            for jaccard, other in pairs[:3]:
                top.append({"id": other, "jaccard": round(jaccard, 4)})
            out["top_overlaps"] = top
            continue
        out[key] = signature[key]
    return sanitize(out)


def is_float(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def mentions_benchmark(text):
    """True when a note or key names a reported benchmark."""
    lowered = " " + str(text).lower() + " "
    for word in BENCHMARK_WORDS:
        if word in lowered:
            return True
    return False


def score_is_hidden(name, note=None):
    """True for every score whose source is a reported benchmark.

    A score shows only when its name starts with a probe or pool-text prefix and its note names no
    benchmark. A score with an unknown source is hidden.
    """
    allowed = False
    for prefix in ALLOWED_SCORE_PREFIXES:
        if str(name).startswith(prefix):
            allowed = True
    if not allowed:
        return True
    if isinstance(note, str) and mentions_benchmark(note):
        return True
    return False


def structure_signature(signature):
    """A catalog signature with only its structure facts (sizes and mask overlaps), no measured effects."""
    if not isinstance(signature, dict):
        return "hidden: its source is not a known probe measure"
    kept = {}
    for key in signature:
        if key in STRUCTURE_SIGNATURE_KEYS:
            kept[key] = signature[key]
    return compact_signature(kept)


def atlas_entry_row(entry):
    """One atlas entry for the composer: structure facts and probe-style measures, no benchmark scores or maps."""
    row = {"from": entry.get("atlas_file", "atlas")}
    for key in entry:
        if key == "atlas_file":
            continue
        value = entry[key]
        if key in ATLAS_HIDDEN_KEYS:
            continue
        if key == "scores" and isinstance(value, dict):
            kept = {}
            notes = entry.get("score_sources") or {}
            for name in value:
                if not score_is_hidden(name, notes.get(name) if isinstance(notes, dict) else None):
                    kept[name] = value[name]
            value = kept
        elif key == "score_sources":
            continue
        row[key] = value
    return row


class Toolbox:
    """Runs the composer's tools for one request.

    practice_fn(program_dict, payload) compiles and practices on Modal and
    returns a dict (see modal_composer.compile_and_practice).
    """

    def __init__(self, request, sources, practice_fn, language_text):
        self.request = request
        self.sources = sources
        self.practice_fn = practice_fn
        self.language_text = language_text
        self.constraints = request.get("constraints", {})
        self.budget = int(self.constraints.get("compile_budget", 10))
        self.budget_used = 0
        self.drift_tolerance = float(request.get("drift_tolerance", DEFAULT_DRIFT_TOLERANCE))
        self.practiced = {}
        self.finished = None
        self.records = sources.records_by_id()

    def skills(self):
        return list(self.request["skills"])

    def call(self, name, args):
        """Run one tool. Return a JSON-safe dict."""
        if args is None:
            args = {}
        if name == "atlas":
            return sanitize(self.atlas(args.get("skill"), args.get("model")))
        if name == "catalog":
            return sanitize(self.catalog(args.get("type"), args.get("model")))
        if name == "language_reference":
            return self.language_reference()
        if name == "compile_and_practice":
            if "program_text" not in args:
                return {"ok": False, "error": "compile_and_practice needs program_text"}
            return sanitize(self.compile_and_practice(args["program_text"], args.get("practice_tasks")))
        if name == "finish":
            if "program_text" not in args:
                return {"ok": False, "error": "finish needs program_text"}
            return sanitize(self.finish(args["program_text"], args.get("rationale", "")))
        return {"ok": False, "error": f"unknown tool {name!r}; the tools are atlas, catalog, language_reference, "
                                      f"compile_and_practice, finish"}

    # atlas

    def atlas(self, skill=None, model=None):
        score_names = None
        if skill is not None and skill != "":
            if skill in SKILLS:
                score_names = SKILLS[skill]["tags"]
            else:
                score_names = [skill]
        rows = []
        for profile in self.sources.profiles:
            if not (matches(profile.get("hf_id"), model) or matches(profile.get("key"), model)):
                continue
            row = {"from": "profiles.json"}
            for key in ["key", "hf_id", "family", "params_b", "active_params_b", "layers", "hidden",
                        "tokenizer_family", "decode_tok_s"]:
                row[key] = profile.get(key)
            scores = profile.get("scores") or {}
            notes = profile.get("score_sources") or {}
            kept = {}
            kept_notes = {}
            for name in scores:
                if score_is_hidden(name, notes.get(name)):
                    continue
                if score_names is None or self.score_matches(name, score_names):
                    kept[name] = scores[name]
                    if name in notes:
                        kept_notes[name] = notes[name]
            row["scores"] = kept
            row["score_notes"] = kept_notes
            rows.append(row)
        for entry in self.sources.atlas_entries:
            name = entry.get("hf_id") or entry.get("model") or entry.get("id")
            if not matches(name, model):
                continue
            if score_names is not None and not self.entry_mentions(entry, skill, score_names):
                continue
            rows.append(atlas_entry_row(entry))
        note = ("Scores measured on a reported benchmark are hidden, whatever the split (rule of 2026-11-14): "
                "benchmarks are chosen per user request and score only the final model. This table shows "
                "sizes, layers, families, and structure, plus any probe or pool-text measure. Use "
                "compile_and_practice for the probe.")
        if len(self.sources.atlas_entries) == 0:
            note += " The atlas/entries folder is empty, so only profiles.json rows are shown."
        return {"count": len(rows), "rows": rows, "note": note}

    def score_matches(self, name, score_names):
        for wanted in score_names:
            if name == wanted or name.startswith(wanted + "_") or wanted in name:
                return True
        return False

    def entry_mentions(self, entry, skill, score_names):
        for key in ["skill", "skills", "tags", "domain", "domains"]:
            value = entry.get(key)
            if isinstance(value, str) and matches(value, skill):
                return True
            if isinstance(value, list):
                for item in value:
                    if matches(item, skill):
                        return True
            if isinstance(value, dict) and skill in value:
                return True
        scores = entry.get("scores")
        if isinstance(scores, dict):
            for name in scores:
                if self.score_matches(name, score_names):
                    return True
        return False

    # catalog

    def catalog(self, part_type=None, model=None):
        rows = []
        for record in self.sources.catalog_records:
            if part_type and record.get("type") != part_type:
                continue
            if not (matches(record.get("model"), model) or matches(record.get("base"), model)):
                continue
            row = {}
            for key in CATALOG_KEYS:
                if key in record:
                    row[key] = record[key]
            if record.get("heldout_overlap"):
                row["heldout_overlap"] = record["heldout_overlap"]
            if "signature" in record:
                # Measured effects come from benchmark items or an unknown source, so only structure shows.
                row["signature"] = structure_signature(record["signature"])
            rows.append(row)
        return {"count": len(rows), "rows": rows,
                "note": "Any donor layer range is also a graft id: block/<org>/<model>/<first>-<last>."}

    def language_reference(self):
        return {"text": self.language_text + "\n" + composer_notes()}

    # compile and practice

    def parse(self, program_text):
        """Return (program, error dict or None). The request's limits become the program's budget."""
        from wsp.lang import LangError, parse

        try:
            program = parse(program_text)
            sizes.with_request_budget(program, self.constraints)
            return program, None
        except LangError as error:
            return None, {"ok": False, "stage": "parse", "error": str(error)}
        except (ValueError, KeyError, TypeError) as error:
            return None, {"ok": False, "stage": "parse", "error": f"{type(error).__name__}: {error}"}

    def check(self, program):
        """Check a program before it compiles. A failed check costs no compile."""
        result = sizes.check_program(program, self.constraints, self.sources.models, self.records)
        violations = result["violations"]
        base = self.request.get("base")
        if base is not None and program.host != base:
            violations.append(f"the request names the base {base}, so the program's host must be {base}, "
                              f"not {program.host}")
        for component_id in program.host_component_ids():
            record = self.records.get(component_id)
            if record is None:
                violations.append(f"unknown catalog id {component_id!r}")
                continue
            # The compiler builds masks and task vectors on their own base only.
            base = record.get("base")
            if base is not None and base != program.host:
                violations.append(f"{component_id} is built on {base}, not on the host {program.host}; "
                                  f"use a graft to bring in a part from another model")
        for graft in program.grafts:
            record = self.records.get(graft.component_id)
            if record is None:
                record = sizes.block_record(graft.component_id)
            if record is None:
                message = f"unknown graft id {graft.component_id!r}"
                if message not in violations:
                    violations.append(message)
                continue
            if record.get("type") not in GRAFT_TYPES:
                violations.append(f"{graft.component_id} is a {record.get('type')}; a graft takes one of "
                                  + ", ".join(GRAFT_TYPES))
        known = HEAL_DATA
        data_names = []
        if program.heal_spec is not None:
            data_names.append(("heal", program.heal_spec.data))
        if program.tap_train_spec is not None:
            data_names.append(("tap_train", program.tap_train_spec.data))
        if program.carve_spec is not None and program.carve_spec.domain_data is not None:
            data_names.append(("carve domain_data", program.carve_spec.domain_data))
        for what, name in data_names:
            if name not in known:
                extra = ""
                if name == "gsm8k_train":
                    extra = " (the language default; write `data <name>` on the heal, because GSM8K train is a benchmark)"
                violations.append(f"{what} data {name!r} is unknown{extra}; the names are " + ", ".join(known))
                continue
            try:
                check_heal_sources(name)
            except ValueError as error:
                violations.append(f"{what} data {name!r}: {error}")
        result["ok"] = len(violations) == 0
        return result

    def used_records(self, program):
        """The catalog records the compile needs, with block ids turned into block records."""
        records = []
        seen = set()
        ids = program.component_ids()
        for component_id in ids:
            if component_id in seen:
                continue
            seen.add(component_id)
            if component_id in self.records:
                records.append(self.records[component_id])
            else:
                record = sizes.block_record(component_id)
                if record is not None:
                    records.append(record)
        return records

    def records_with_blocks(self, program):
        """Catalog records by id, plus block records for the program's block ids."""
        records = dict(self.records)
        for component_id in program.component_ids():
            if component_id not in records:
                record = sizes.block_record(component_id)
                if record is not None:
                    records[component_id] = record
        return records

    def budget_note(self):
        return {"budget_used": self.budget_used, "budget_left": self.budget - self.budget_used}

    def compile_and_practice(self, program_text, practice_tasks=None):
        if self.finished is not None:
            return {"ok": False, "stage": "finished", "error": "finish was already called"}
        program, error = self.parse(program_text)
        if error is not None:
            error.update(self.budget_note())
            return error
        if practice_tasks is None or len(practice_tasks) == 0:
            practice_tasks = self.skills()
        for skill in practice_tasks:
            if skill not in self.request["skills"]:
                out = {"ok": False, "stage": "arguments",
                       "error": f"practice task {skill!r} is not a skill of this request: {self.skills()}"}
                out.update(self.budget_note())
                return out
        check = self.check(program)
        if not check["ok"]:
            out = {"ok": False, "stage": "constraints", "violations": check["violations"],
                   "estimate": check["estimate"]}
            out.update(self.budget_note())
            return out

        program_dict = program.to_dict()
        key = program_hash(program_dict)
        cache_key = key + ":" + ",".join(sorted(practice_tasks))
        if cache_key in self.practiced:
            out = dict(self.practiced[cache_key]["result"])
            out["cached"] = True
            out.update(self.budget_note())
            return out
        if self.budget_used >= self.budget:
            out = {"ok": False, "stage": "budget",
                   "error": f"the compile budget is spent ({self.budget_used} of {self.budget}). "
                            f"Call finish with your best practiced program."}
            out.update(self.budget_note())
            return out

        self.budget_used += 1
        payload = {
            "request": self.request["name"],
            "skills": list(practice_tasks),
            "host": program.host,
            "probe_plan": probes.probe_plan(program_dict, list(practice_tasks), self.records_with_blocks(program),
                                            self.sources.models),
            "drift_tolerance": self.drift_tolerance,
            "constraints": dict(self.constraints),
            "probe_items": self.request.get("practice_items") or PROBE_COUNT,
            "catalog": self.used_records(program),
            "program_hash": key,
            "estimate": check["estimate"],
        }
        raised = False
        try:
            raw = self.practice_fn(program_dict, payload)
        except Exception as error:
            # A Modal failure (preemption, timeout) is not cached, so the same
            # program can run again. It still costs its compile.
            raised = True
            raw = {"ok": False, "errors": [f"practice call failed: {type(error).__name__}: {str(error)[:2000]}"]}
        if not isinstance(raw, dict):
            raw = {"ok": False, "errors": [f"practice returned {type(raw).__name__}, not a dict"]}

        probe = raw.get("probe") if raw.get("ok") else None
        weights = {}
        for skill in practice_tasks:
            weights[skill] = self.request["skills"][skill]
        summary = probe_summary(probe, weights, self.drift_tolerance) if probe is not None else None
        result = {
            "ok": bool(raw.get("ok")) and probe is not None,
            "stage": "practice" if raw.get("ok") else "compile",
            "program_hash": key,
            "probe": probe_view(probe, practice_tasks),
            "transfer": summary["transfer"] if summary else None,
            "drift": summary["drift"] if summary else None,
            "drift_tolerance": self.drift_tolerance,
            "drift_penalty": summary["penalty"] if summary else None,
            "practice_objective": summary["objective"] if summary else None,
            "params_total_b": raw.get("params_total_b"),
            "params_run_b": raw.get("params_run_b"),
            "decode_tok_s": raw.get("decode_tok_s"),
            "heal_seconds": raw.get("heal_seconds"),
            "heal_steps_run": raw.get("heal_steps_run"),
            "compile_seconds": raw.get("compile_seconds"),
            "practice_seconds": raw.get("practice_seconds"),
            "errors": raw.get("errors") or [],
            "compile_summary": raw.get("compile_summary"),
            "estimate": check["estimate"],
        }
        if raw.get("ok") and probe is None:
            result["errors"] = result["errors"] + ["the compile returned no probe report"]
        result["request_objective"] = None
        if result["ok"]:
            # The objective over every request skill, an unprobed skill as transfer 0,
            # so programs probed on fewer skills rank fairly.
            probed = {"skills": {}, "drift": probe.get("drift")}
            for skill in practice_tasks:
                probed["skills"][skill] = (probe.get("skills") or {}).get(skill) or {}
            result["request_objective"] = probe_summary(probed, self.request["skills"],
                                                        self.drift_tolerance)["objective"]
        violations = sizes.check_measured(raw, self.constraints)
        if result["ok"] and self.constraints.get("max_heal_minutes") is not None and \
                raw.get("heal_seconds") is None:
            violations.append("the compile did not report heal_seconds, so the max_heal_minutes cap is not checked")
        for name, cap_name in [("params_total_b", "max_total_params_b"), ("params_run_b", "max_run_params_b")]:
            if result["ok"] and self.constraints.get(cap_name) is not None and raw.get(name) is None:
                violations.append(f"the compile did not report {name}, so the {cap_name} cap is not checked")
        result["within_constraints"] = result["ok"] and len(violations) == 0
        result["violations"] = violations
        if not raised:
            self.practiced[cache_key] = {"program_text": program_text, "program": program_dict, "result": result,
                                         "model_dir": raw.get("model_dir"), "skills": list(practice_tasks)}
        out = dict(result)
        out.update(self.budget_note())
        return out

    def good_runs(self):
        """Practiced programs that compiled and fit the constraints, best first."""
        runs = []
        for cache_key in self.practiced:
            entry = self.practiced[cache_key]
            if entry["result"]["within_constraints"] and not has_control(Program.from_dict(entry["program"])):
                runs.append(entry)
        runs.sort(key=lambda entry: entry["result"].get("request_objective") or 0.0, reverse=True)
        return runs

    def finish(self, program_text, rationale):
        if self.finished is not None:
            return {"ok": False, "error": "finish was already called"}
        program, error = self.parse(program_text)
        if error is not None:
            return error
        if has_control(program):
            return {"ok": False, "error": "finish needs the real program, not a control: remove `control random`. "
                                          "The final report builds the random control itself."}
        key = program_hash(program.to_dict())
        match = None
        for entry in self.good_runs():
            if entry["result"]["program_hash"] == key:
                match = entry
                break
        if match is None:
            if len(self.good_runs()) > 0 or self.budget_used < self.budget:
                hashes = []
                for entry in self.good_runs():
                    hashes.append({"program_hash": entry["result"]["program_hash"],
                                   "request_objective": entry["result"].get("request_objective")})
                return {"ok": False, "error": "finish needs a program you compiled and practiced within the "
                                              "constraints. Practiced programs that qualify: " + json.dumps(hashes)}
            self.finished = {"program_text": program_text, "program": program.to_dict(), "rationale": rationale,
                             "program_hash": key, "practice": None, "model_dir": None, "no_model": True,
                             "fallback": False}
            return {"ok": True, "note": "recorded, but no program compiled within the constraints, so there is "
                                        "no model to test"}
        self.finished = {"program_text": program_text, "program": match["program"], "rationale": rationale,
                         "program_hash": key, "practice": match["result"], "model_dir": match["model_dir"],
                         "no_model": False, "fallback": False}
        return {"ok": True, "program_hash": key, "note": "submitted; held-out tests run next"}

    def fallback_finish(self, reason):
        """When the loop stops without finish, submit the best practiced program."""
        runs = self.good_runs()
        if len(runs) == 0:
            self.finished = {"program_text": None, "program": None, "rationale": reason, "program_hash": None,
                             "practice": None, "model_dir": None, "no_model": True, "fallback": True}
            return self.finished
        best = runs[0]
        self.finished = {"program_text": best["program_text"], "program": best["program"],
                         "rationale": reason + " (best practiced program chosen by the loop)",
                         "program_hash": best["result"]["program_hash"], "practice": best["result"],
                         "model_dir": best["model_dir"], "no_model": False, "fallback": True}
        return self.finished


def probe_view(probe, skills):
    """The probe report as the composer sees it: per skill transfer, donor, path, and verdict, and drift."""
    if probe is None:
        return None
    rows = {}
    for skill in skills:
        row = (probe.get("skills") or {}).get(skill) or {}
        rows[skill] = {key: row.get(key) for key in ("transfer", "donors", "donor", "path", "tokens",
                                                       "content_gain", "random_transfer")
                       if row.get(key) is not None}
        if row.get("transfer") is None:
            rows[skill]["transfer"] = 0.0
            rows[skill]["note"] = row.get("note") or "no donor component or no gap, so transfer is 0"
        verdict = (probe.get("verdict") or {}).get(skill)
        if verdict is not None:
            rows[skill]["verdict"] = verdict
    return {"skills": rows, "drift": probe.get("drift")}


def has_control(program):
    """True when any graft, experts, or insert statement of the program carries a control."""
    for graft in program.grafts:
        if graft.control is not None:
            return True
    if program.experts_spec is not None and program.experts_spec.control is not None:
        return True
    for spec in program.inserts:
        if spec.control is not None:
            return True
    return False


def read_language_text(root):
    path = os.path.join(root, "docs", "language.md")
    if not os.path.exists(path):
        return "docs/language.md is missing."
    with open(path) as f:
        return f.read()


__all__ = ["Toolbox", "tool_specs", "program_hash", "read_language_text", "SET_NOTES"]
