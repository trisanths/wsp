"""Interp agent loop: choose probe and overlap experiments, run them, write signatures.

The agent keeps a queue of proposed experiments. Each step it takes the
highest-priority one, runs it, logs (hypothesis, action, result) to
interp_log.jsonl, and may propose follow-ups based on the result:

  1. overlap every pair of masks on the same base (free, no eval budget)
  2. probe each component at all of its layers (= component vs base)
  3. if a component has a clear effect on its own skill, probe layer bands
  4. split the band with the biggest drop into halves, while budget lasts
  5. write each record's signature, including a capability map like
     "mmlupro:biology lives mostly in layers 8 to 11"

All probe numbers are search-split numbers and are labelled as such.
The planner is a plain class, so an LLM planner can replace `propose_followups`
later without touching the rest of the loop.
"""
import datetime
import json
import math

from wsp_eval import budget, catalog, components, config, harness, overlap_tool as overlap_module, probe_tool as probe_module


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def standard_error(score, n):
    """Binomial standard error of a score measured on n items."""
    if n <= 0:
        return 0.0
    return math.sqrt(max(score * (1.0 - score), 0.0) / n)


def band_name(layers):
    if len(layers) == 1:
        return str(layers[0])
    return str(layers[0]) + "-" + str(layers[-1])


def make_bands(layers, band_size):
    bands = []
    for start in range(0, len(layers), band_size):
        bands.append(layers[start:start + band_size])
    return bands


def capability_map(task, band_drops):
    """Smallest run of adjacent bands holding at least half of the positive drop.

    band_drops: list of (layers, drop) for non-overlapping bands, in layer order.
    Band drops are not additive (layers interact), so the share is only a guide.
    """
    positive_total = 0.0
    for layers, drop in band_drops:
        if drop > 0:
            positive_total += drop
    if positive_total <= 0:
        return None

    best = None
    for start in range(len(band_drops)):
        mass = 0.0
        for end in range(start, len(band_drops)):
            drop = band_drops[end][1]
            if drop > 0:
                mass += drop
            if mass >= 0.5 * positive_total:
                width = end - start
                if best is None or width < best[0] or (width == best[0] and mass > best[3]):
                    best = (width, start, end, mass)
                break
    width, start, end, mass = best
    first_layer = band_drops[start][0][0]
    last_layer = band_drops[end][0][-1]
    share = mass / positive_total
    return {
        "task": task,
        "layers": [first_layer, last_layer],
        "share_of_band_drop": share,
        "text": (task + " lives mostly in layers " + str(first_layer) + " to " + str(last_layer) +
                 " (" + str(round(100 * share)) + "% of the summed band-ablation drop, search split)"),
    }


class InterpAgent:
    def __init__(self, catalog_path=None, tasks=None, split="search", max_evals=80, band_size=4,
                 min_band_size=1, namespace="interp", log_path=None, ids=None, noise_multiplier=2.0):
        self.catalog_path = catalog_path
        self.tasks = tasks
        self.split = split
        self.max_evals = max_evals
        self.band_size = band_size
        self.min_band_size = min_band_size
        self.namespace = namespace
        self.log_path = log_path or config.interp_log_path()
        self.only_ids = ids
        # A drop counts as real only if it exceeds noise_multiplier binomial standard errors.
        self.noise_multiplier = noise_multiplier
        self.evals_used = 0
        self.queue = []
        self.full_scores = {}      # id -> {task: score} with the full component
        self.effects = {}          # id -> probe result for all layers
        self.band_results = {}     # id -> {band name: probe result}
        self.top_level_bands = {}  # id -> list of top-level bands (list of layers)
        self.overlaps = {}         # id -> {other id: summary}

    # -- logging ------------------------------------------------------------

    def log(self, hypothesis, action, result):
        record = {"time": _now(), "hypothesis": hypothesis, "action": action, "result": result,
                  "evals_used": self.evals_used, "split": self.split}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("[interp]", action["kind"], action.get("component", ""), action.get("layers", ""),
              "->", _short(result))

    # -- planning -----------------------------------------------------------

    def propose(self, priority, experiment):
        self.queue.append((priority, len(self.queue), experiment))

    def next_experiment(self):
        if len(self.queue) == 0:
            return None
        best_index = 0
        for i in range(1, len(self.queue)):
            if self.queue[i][0] > self.queue[best_index][0]:
                best_index = i
        item = self.queue.pop(best_index)
        return item[2]

    def own_tasks(self, record):
        try:
            return probe_module.default_tasks(record)
        except ValueError:
            return None

    def tasks_for(self, record):
        if self.tasks is not None:
            return self.tasks
        tasks = []
        for other in self.records:
            if other["base"] != record["base"]:
                continue
            own = self.own_tasks(other)
            if own is None:
                continue
            for task in own:
                if task not in tasks:
                    tasks.append(task)
        if len(tasks) == 0:
            raise ValueError("no tasks for " + record["id"] + "; pass tasks=")
        return tasks

    def plan_initial(self):
        masks = []
        for record in self.records:
            if record["type"] == "deletion_mask":
                masks.append(record)
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if masks[i]["base"] == masks[j]["base"]:
                    self.propose(100, {"kind": "overlap", "a": masks[i]["id"], "b": masks[j]["id"],
                                       "hypothesis": "masks " + masks[i]["id"] + " and " + masks[j]["id"] +
                                       " delete mostly different weights, so their union may compose"})
        for record in self.records:
            if record["type"] not in ("deletion_mask", "task_vector"):
                continue
            own = self.own_tasks(record)
            hypothesis = record["id"] + " improves "
            if own is None:
                hypothesis += "some task over the base model"
            else:
                hypothesis += ", ".join(own) + " over the base model and changes other skills less"
            self.propose(50, {"kind": "probe_all", "component": record["id"], "hypothesis": hypothesis})

    def propose_followups(self, experiment, result):
        """Decide which experiments to run next, given a result."""
        record = self.by_id[experiment["component"]]
        own = self.own_tasks(record)
        if own is None:
            own = self.tasks_for(record)
        focus = own[0]

        if experiment["kind"] == "probe_all":
            effect = result["drop"][focus]
            noise = self.noise_multiplier * standard_error(result["full"][focus], result["n"])
            if effect <= noise:
                self.log("localize " + record["id"], {"kind": "skip_localization", "component": record["id"]},
                         {"reason": "effect on " + focus + " (" + _fmt(effect) + ") is within noise (" +
                          _fmt(noise) + ") on the search split"})
                return
            layers = components.component_layers(record, self.catalog_path)
            bands = make_bands(layers, self.band_size)
            self.top_level_bands[record["id"]] = bands
            for band in bands:
                self.propose(40 + effect, {
                    "kind": "probe_band", "component": record["id"], "layers": band, "focus": focus,
                    "hypothesis": "layers " + band_name(band) + " carry part of " + record["id"] +
                                  "'s effect on " + focus})
            return

        if experiment["kind"] == "probe_band":
            band = experiment["layers"]
            drop = result["drop"][focus]
            total = self.effects[record["id"]]["drop"][focus]
            noise = self.noise_multiplier * standard_error(result["full"][focus], result["n"])
            if len(band) // 2 < self.min_band_size:
                return
            if drop > noise and drop >= 0.25 * total:
                half = len(band) // 2
                for part in (band[:half], band[half:]):
                    self.propose(20 + drop, {
                        "kind": "probe_band", "component": record["id"], "layers": part, "focus": focus,
                        "hypothesis": "within layers " + band_name(band) + ", layers " + band_name(part) +
                                      " carry most of the drop on " + focus})

    # -- running ------------------------------------------------------------

    def evals_left(self):
        return self.max_evals - self.evals_used

    def run_experiment(self, experiment):
        kind = experiment["kind"]
        if kind == "overlap":
            stats = overlap_module.overlap(experiment["a"], experiment["b"], by="layer",
                                           catalog_path=self.catalog_path)
            per_layer = {}
            for layer, values in stats["groups"].items():
                per_layer[str(layer)] = values["jaccard"]
            summary = {"jaccard": stats["total"]["jaccard"], "shared": stats["total"]["shared"],
                       "jaccard_per_layer": per_layer}
            for one, other in ((experiment["a"], experiment["b"]), (experiment["b"], experiment["a"])):
                if one not in self.overlaps:
                    self.overlaps[one] = {}
                self.overlaps[one][other] = summary
            self.log(experiment["hypothesis"], {"kind": "overlap", "a": experiment["a"], "b": experiment["b"]},
                     {"total": stats["total"]})
            return

        record = self.by_id[experiment["component"]]
        tasks = self.tasks_for(record)
        # One eval for the ablated model, plus one for the full model the first time.
        cost = 1
        if record["id"] not in self.full_scores:
            cost = 2
        if cost > self.evals_left():
            self.log(experiment["hypothesis"], {"kind": "skipped", "component": record["id"]},
                     {"reason": "eval budget for the interp agent is used up"})
            return

        if kind == "probe_all":
            layers = components.component_layers(record, self.catalog_path)
        else:
            layers = experiment["layers"]

        result = probe_module.probe(record, layers, tasks=tasks, split=self.split, namespace=self.namespace,
                                    full_scores=self.full_scores.get(record["id"]),
                                    catalog_path=self.catalog_path)
        self.evals_used += cost
        self.full_scores[record["id"]] = result["full"]

        if kind == "probe_all":
            self.effects[record["id"]] = result
        else:
            if record["id"] not in self.band_results:
                self.band_results[record["id"]] = {}
            self.band_results[record["id"]][band_name(layers)] = result

        action = {"kind": kind, "component": record["id"], "layers": band_name(layers), "tasks": tasks}
        self.log(experiment["hypothesis"], action,
                 {"label": result["label"], "full": result["full"], "ablated": result["ablated"],
                  "drop": result["drop"], "n": result["n"]})
        self.propose_followups(experiment, result)

    def run(self):
        self.records = []
        for record in catalog.load(self.catalog_path):
            if self.only_ids is None or record["id"] in self.only_ids:
                self.records.append(record)
        self.by_id = {}
        for record in self.records:
            self.by_id[record["id"]] = record

        self.plan_initial()
        while True:
            experiment = self.next_experiment()
            if experiment is None:
                break
            try:
                self.run_experiment(experiment)
            except budget.BudgetExhausted as error:
                self.log(experiment.get("hypothesis", ""), {"kind": "stopped"}, {"reason": str(error)})
                break
            except Exception as error:
                # Report the failure and move on; never invent a number.
                self.log(experiment.get("hypothesis", ""), {"kind": "failed", "experiment": experiment},
                         {"error": type(error).__name__ + ": " + str(error)})

        self.write_signatures()
        return self.summary()

    # -- signatures ---------------------------------------------------------

    def signature_for(self, record):
        signature = {"split": self.split, "label": harness.SEARCH_LABEL if self.split == "search" else "HELD-OUT",
                     "updated": _now()}

        try:
            if record["type"] == "deletion_mask":
                mask = components.load_component_mask(record, self.catalog_path)
                num_layers = components.num_layers_of_base(record["base"])
                signature["deleted_per_layer"] = components.deleted_per_layer(mask, num_layers)
                signature["num_deleted"] = sum(signature["deleted_per_layer"])
        except Exception as error:
            signature["error"] = "could not load component: " + str(error)

        if record["id"] in self.effects:
            result = self.effects[record["id"]]
            signature["effect"] = result["drop"]   # component minus base, per task
            signature["scores_with_component"] = result["full"]
            signature["scores_base"] = result["ablated"]
            signature["n_items"] = result["n"]

        if record["id"] in self.overlaps:
            signature["overlap_with"] = self.overlaps[record["id"]]

        if record["id"] in self.band_results:
            layer_effects = {}
            for name, result in self.band_results[record["id"]].items():
                layer_effects[name] = result["drop"]
            signature["layer_effects"] = layer_effects

            maps = []
            bands = self.top_level_bands.get(record["id"], [])
            tasks = self.tasks_for(record)
            for task in tasks:
                band_drops = []
                complete = True
                for band in bands:
                    name = band_name(band)
                    if name not in self.band_results[record["id"]]:
                        complete = False
                        break
                    band_drops.append((band, self.band_results[record["id"]][name]["drop"][task]))
                if not complete:
                    continue
                entry = capability_map(task, band_drops)
                if entry is not None:
                    maps.append(entry)
            signature["capability_map"] = maps
        return signature

    def write_signatures(self):
        for record in self.records:
            if record["type"] not in ("deletion_mask", "task_vector"):
                continue
            signature = self.signature_for(record)
            catalog.update_signature(record["id"], signature, self.catalog_path)
            self.log("signature for " + record["id"], {"kind": "write_signature", "component": record["id"]},
                     {"keys": sorted(signature.keys())})

    def summary(self):
        lines = []
        for record in self.records:
            if record["id"] in self.effects:
                lines.append(record["id"] + " effect (search split): " + _short(self.effects[record["id"]]["drop"]))
        for record in self.records:
            signature = self.signature_for(record)
            for entry in signature.get("capability_map", []):
                lines.append(record["id"] + ": " + entry["text"])
        lines.append("evals used: " + str(self.evals_used) + " of " + str(self.max_evals))
        return "\n".join(lines)


def _fmt(x):
    return "%.3f" % x


def _short(value):
    text = json.dumps(value)
    if len(text) > 160:
        text = text[:157] + "..."
    return text
