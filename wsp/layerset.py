"""Minimal layer set per skill: which layers of a true fine-tune carry its skill.

A member (wsp.weave.Member) is a fine-tune of the host. Its difference from
the host sits in every layer, but the skill may need only some of them. This
module finds the smallest set of layers S such that writing the member's
difference on S alone (attention and MLP of each layer together) keeps most
of the member's probe transfer and does no more harm than the tolerance.

Everything is forward-only. A candidate is written into the host with
wsp.weave.Weaver.apply, scored against the cached teacher distributions of a
wsp.weave.ProbeObjective, and removed again with Weaver.restore. No
generation and no benchmark item is read.

Terms:
  layer set      a collection of layer indices for one member.
  assignment     {member: layer set}. Members are named by name or by index.
  transfer       gap closed between KL(member || host) and 0 on the member's
                 probe pool. The host scores 0 and the member itself scores 1.
  harm           next-token loss of the written model on general text minus
                 the host's, in nats per token.
  share          transfer of a layer set divided by the transfer of the full
                 member (every layer).
  ranking        layers ordered by the transfer of each layer written alone.

Steps:
  layer_importance   one layer at a time, together and as attention only and
                     MLP only. Gives the ranking.
  minimal_set        bisect the prefixes of the ranking for the smallest
                     prefix that keeps the target share within the harm
                     limit, then drop layers one at a time while it still
                     holds.
  compose            write two members, each on its own set. A layer in both
                     sets goes to the member with the larger difference norm,
                     or to the owner that gives the better probe value.
  controls           random sets of the same size, a set chosen by difference
                     norm only, the cross-skill swap, and the scrambled
                     difference on the set.

The Weaver here has one bucket per layer, so genome weight [member, layer,
kind] is the weight of that layer. Members must be built with the groups
("attn", "mlp") only, so norms and embeddings stay with the host.
"""

import copy
import math

import numpy as np
import torch

from wsp import weave

KIND_CHOICES = ("both", "attn", "mlp")


# ---------------------------------------------------------------- setup


def layer_count(model):
    """Number of decoder layers, from the classified parameter names."""
    layers = 0
    for name, _ in model.named_parameters():
        info = weave.classify(name)
        if info is not None and info[1] is not None:
            layers = max(layers, info[1] + 1)
    return layers


def layer_weaver(model, members, density_step=0.05):
    """A Weaver with one bucket per layer. Build it before any change to the model."""
    return weave.Weaver(model, members, n_buckets=layer_count(model), density_step=density_step)


def restrict_objective(objective, names):
    """A copy of a ProbeObjective that scores only some members (calibration is kept, so no new forward pass)."""
    names = list(names)
    for name in names:
        if name not in objective.names:
            raise ValueError(f"{name} is not a member of the objective")
    out = copy.copy(objective)
    out.names = names
    out.caches = {n: objective.caches[n] for n in names}
    out.host_pool_caches = {n: v for n, v in objective.host_pool_caches.items() if n in names}
    out.baseline = None if objective.baseline is None else {n: objective.baseline[n] for n in names}
    return out


def layer_genome(n_members, n_layers, assignment, kinds="both"):
    """A genome with weight 1 on the listed layers of each member and 0 elsewhere.

    assignment: {member index: layers}. kinds: "both", "attn", or "mlp", or a
    {member index: kind} dict.
    """
    genome = weave.Genome(n_members, n_layers)
    for index, layers in assignment.items():
        kind = kinds.get(index, "both") if isinstance(kinds, dict) else kinds
        if kind not in KIND_CHOICES:
            raise ValueError("kind must be both, attn, or mlp")
        for layer in layers:
            if not 0 <= int(layer) < n_layers:
                raise ValueError(f"layer {layer} is outside 0 to {n_layers - 1}")
            if kind in ("both", "attn"):
                genome.weights[index, int(layer), 0] = 1.0
            if kind in ("both", "mlp"):
                genome.weights[index, int(layer), 1] = 1.0
    return genome


def full_member_genome(n_members, n_layers, index):
    """Every layer of one member at weight 1 (attention and MLP only), built with weave.one_hot_genome."""
    return weave.one_hot_genome(n_members, n_layers, index, 1.0, with_globals=False)


def host_matches(weaver, model):
    """True when every tensor a member can write holds the host's value, bit for bit."""
    params = dict(model.named_parameters())
    for name in weaver.names:
        if not torch.equal(params[name].detach().cpu(), weaver.host[name]):
            return False
    return True


def layer_norms(weaver, member):
    """Difference norms per layer: {"attn": [...], "mlp": [...], "both": [...]} (L2 of the float32 difference)."""
    index = weaver.members.index(member) if not isinstance(member, int) else member
    sums = {"attn": np.zeros(weaver.n_layers), "mlp": np.zeros(weaver.n_layers)}
    for name in weaver.names:
        kind, layer = weaver.info[name]
        if kind not in weave.KINDS or not weaver.members[index].covers(name, kind):
            continue
        d = weaver.delta(index, name)
        sums[kind][layer] += float(d.double().pow(2).sum())
    attn = np.sqrt(sums["attn"])
    mlp = np.sqrt(sums["mlp"])
    both = np.sqrt(sums["attn"] + sums["mlp"])
    return {"attn": attn.tolist(), "mlp": mlp.tolist(), "both": both.tolist()}


# ---------------------------------------------------------------- scorer


class LayerScorer:
    """Writes layer sets into the host, scores them with a ProbeObjective, and restores the host.

    One scorer holds one weaver, one model, and one objective (one probe
    split). Results are memoized by assignment and kind, so a repeated set
    costs no forward pass. forward_sets counts the sets that did run.
    """

    def __init__(self, weaver, model, objective, log=None):
        if weaver.n_buckets != weaver.n_layers:
            raise ValueError("the Weaver needs one bucket per layer (use layer_weaver)")
        self.weaver = weaver
        self.model = model
        self.objective = objective
        self.log = log
        self.names = [m.name for m in weaver.members]
        for name in objective.names:
            if name not in self.names:
                raise ValueError(f"objective member {name} has no Weaver member")
        self.n_layers = weaver.n_layers
        self.memo = {}
        self.forward_sets = 0
        weaver.restore(model)
        if objective.baseline is None:
            objective.calibrate(model)

    def index(self, member):
        if isinstance(member, (int, np.integer)):
            return int(member)
        return self.names.index(member)

    def name(self, member):
        return self.names[self.index(member)]

    def _assignment(self, assignment):
        out = {}
        for member, layers in assignment.items():
            layers = sorted(set(int(layer) for layer in layers))
            if len(layers) > 0:
                out[self.index(member)] = layers
        return out

    def score(self, assignment, kinds="both", weaver=None):
        """Score an assignment. Returns {"transfer", "harm", "drift", "mean_transfer", "value", "member_kl"}.

        weaver: an optional clone of the Weaver (for example weaver.shuffled(seed)); those results are not memoized.
        """
        assignment = self._assignment(assignment)
        kind_key = tuple(sorted(kinds.items())) if isinstance(kinds, dict) else kinds
        key = (tuple(sorted((i, tuple(ls)) for i, ls in assignment.items())), kind_key)
        if weaver is None and key in self.memo:
            return self.memo[key]
        use = self.weaver if weaver is None else weaver
        genome = layer_genome(len(self.names), self.n_layers, assignment, kinds)
        try:
            use.apply(self.model, genome)
            scores = self.objective.scores(self.model)
        finally:
            use.restore(self.model)
            self.weaver.restore(self.model)
        value, _ = self.objective(scores)
        out = {"transfer": dict(scores["transfer"]), "harm": scores["harm"], "drift": scores["drift"],
               "mean_transfer": self.objective.mean_transfer(scores), "value": value,
               "member_kl": dict(scores["member_kl"])}
        self.forward_sets += 1
        if weaver is None:
            self.memo[key] = out
        return out

    def full(self, member):
        """The whole member: every layer, attention and MLP."""
        index = self.index(member)
        return self.score({index: range(self.n_layers)})

    def transfer_of(self, result, member):
        return result["transfer"][self.name(member)]

    def share(self, result, member, full=None):
        """Transfer of a result over the transfer of the full member, or None when either is missing or not positive."""
        full = self.full(member) if full is None else full
        top = self.transfer_of(full, member)
        value = self.transfer_of(result, member)
        if top is None or value is None or top <= 0.0:
            return None
        return value / top


# ---------------------------------------------------------------- importance


def rank_layers(rows, norms=None):
    """Layers ordered by single-layer transfer (best first). Ties go to the larger difference norm, then the lower index."""
    def key(row):
        transfer = row["both"]["transfer"]
        value = -math.inf if transfer is None else transfer
        norm = 0.0 if norms is None else norms["both"][row["layer"]]
        return (-value, -norm, row["layer"])

    return [row["layer"] for row in sorted(rows, key=key)]


def layer_importance(scorer, member, kinds=KIND_CHOICES, with_norms=True):
    """Write the member's difference on one layer at a time and record transfer and harm.

    For each layer: "both" (attention and MLP together), "attn" only, and
    "mlp" only, as listed in kinds. Returns {"member", "full", "layers": [one
    row per layer], "ranking", "norms"}. The ranking uses the "both" rows.
    """
    index = scorer.index(member)
    name = scorer.names[index]
    before = scorer.forward_sets
    full = scorer.full(index)
    norms = layer_norms(scorer.weaver, index) if with_norms else None
    rows = []
    for layer in range(scorer.n_layers):
        row = {"layer": layer}
        for kind in kinds:
            result = scorer.score({index: [layer]}, kind)
            row[kind] = {"transfer": result["transfer"][name], "harm": result["harm"], "drift": result["drift"]}
        if norms is not None:
            row["norm"] = {"both": norms["both"][layer], "attn": norms["attn"][layer], "mlp": norms["mlp"][layer]}
        rows.append(row)
        if scorer.log is not None:
            scorer.log(f"importance {name} layer {layer}: both {row['both']['transfer']}")
    if "both" not in kinds:
        raise ValueError("kinds must include both")
    return {"member": name, "full": {"transfer": full["transfer"][name], "harm": full["harm"],
                                     "drift": full["drift"]},
            "layers": rows, "ranking": rank_layers(rows, norms), "norms": norms,
            "forward_sets": scorer.forward_sets - before}


# ---------------------------------------------------------------- minimal set


def minimal_set(scorer, member, ranking, target_share=0.9, harm_limit=None, greedy=True):
    """The smallest prefix of the ranking that keeps target_share of the member's transfer within the harm limit.

    The harm limit defaults to the larger of the objective's tolerance and
    the harm of the full member, because a true fine-tune can sit above the
    tolerance by itself and no subset should be held to a stricter bar than
    the member. The record says which limit applied.

    Step 1 bisects the prefix length (the full set passes by construction).
    Step 2 walks the chosen layers from the lowest rank up and drops each one
    that the set can lose and still pass. Returns a dict with "layers",
    "size", "size_share", "share", "transfer", "harm", "stop_rule" (the set
    is more than 75 percent of the layers), "within_half", and the bisection
    size before the greedy check.
    """
    index = scorer.index(member)
    name = scorer.names[index]
    n = scorer.n_layers
    before = scorer.forward_sets
    full = scorer.full(index)
    top = full["transfer"][name]
    if top is None or top <= 0.0:
        return {"member": name, "status": "no_signal", "full_transfer": top, "layers": [], "size": 0}
    limit = harm_limit if harm_limit is not None else max(scorer.objective.drift_tolerance, full["harm"])
    ranking = [int(layer) for layer in ranking]
    if sorted(ranking) != list(range(n)):
        raise ValueError("ranking must list every layer once")

    def passes(layers):
        result = scorer.score({index: layers})
        share = scorer.share(result, index, full)
        return bool(share is not None and share >= target_share and result["harm"] <= limit + 1e-12), result, share

    lo, hi = 0, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, _, _ = passes(ranking[:mid])
        if ok:
            hi = mid
        else:
            lo = mid
    chosen = list(ranking[:hi])
    bisect_size = len(chosen)
    removed = []
    if greedy:
        for layer in reversed(list(chosen)):
            if len(chosen) <= 1:
                break
            trial = [x for x in chosen if x != layer]
            ok, _, _ = passes(trial)
            if ok:
                chosen = trial
                removed.append(layer)
    ok, result, share = passes(chosen)
    size_share = len(chosen) / n
    return {"member": name, "status": "ok" if ok else "full_set_fails", "layers": sorted(chosen),
            "size": len(chosen), "n_layers": n, "size_share": size_share, "bisect_size": bisect_size,
            "greedy_removed": removed, "share": share, "transfer": result["transfer"][name],
            "harm": result["harm"], "harm_limit": limit, "target_share": target_share, "full_transfer": top,
            "full_harm": full["harm"], "stop_rule": bool(size_share > 0.75), "within_half": bool(size_share <= 0.5),
            "forward_sets": scorer.forward_sets - before}


# ---------------------------------------------------------------- composition


def compose(scorer, sets, resolve="norm", norms=None):
    """Write two or more members, each on its own layer set.

    sets: {member: layers}. A layer in more than one set goes to one owner.
      resolve "norm"      the owner whose difference at that layer has the larger L2 norm (ties: first listed).
      resolve "try_both"  start from the norm choice, then for each shared layer try every other owner and keep the
                          assignment with the higher probe value (mean transfer minus the harm penalty).
    The scorer's objective should hold every member in sets. norms:
    {member name: layer_norms(...)}, computed when missing. Returns
    {"assignment": {name: layers}, "overlap": [layers], "decisions": [...], "result": probe scores, "value"}.
    """
    if resolve not in ("norm", "try_both"):
        raise ValueError("resolve must be norm or try_both")
    owners = {}
    order = []
    for member, layers in sets.items():
        index = scorer.index(member)
        order.append(index)
        for layer in layers:
            owners.setdefault(int(layer), []).append(index)
    if norms is None:
        norms = {}
    for index in order:
        name = scorer.names[index]
        if name not in norms:
            norms[name] = layer_norms(scorer.weaver, index)

    def norm_of(index, layer):
        return norms[scorer.names[index]]["both"][layer]

    assignment = {index: [] for index in order}
    overlap = sorted(layer for layer, who in owners.items() if len(who) > 1)
    decisions = []
    for layer, who in sorted(owners.items()):
        pick = who[0]
        for other in who[1:]:
            if norm_of(other, layer) > norm_of(pick, layer):
                pick = other
        assignment[pick].append(layer)
        if len(who) > 1:
            decisions.append({"layer": layer, "owners": [scorer.names[i] for i in who],
                              "norms": [norm_of(i, layer) for i in who], "by_norm": scorer.names[pick]})
    result = scorer.score(assignment)
    if resolve == "try_both" and len(overlap) > 0:
        best_value = result["value"]
        for layer in overlap:
            current = [i for i in order if layer in assignment[i]][0]
            for other in owners[layer]:
                if other == current:
                    continue
                trial = {i: [x for x in ls if x != layer or i == other] for i, ls in assignment.items()}
                trial[other] = sorted(set(trial[other]) | {layer})
                trial_result = scorer.score(trial)
                if trial_result["value"] > best_value + 1e-12:
                    assignment = trial
                    best_value = trial_result["value"]
                    result = trial_result
                    current = other
            for entry in decisions:
                if entry["layer"] == layer:
                    entry["by_probe"] = scorer.names[current]
    return {"resolve": resolve, "assignment": {scorer.names[i]: sorted(ls) for i, ls in assignment.items()},
            "overlap": overlap, "decisions": decisions, "result": result, "value": result["value"]}


# ---------------------------------------------------------------- controls


def random_layer_sets(n_layers, size, seeds):
    """One sorted random layer set of the given size per seed."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        out.append(sorted(int(x) for x in rng.choice(n_layers, size=size, replace=False)))
    return out


def summarize(values):
    """Mean and standard deviation of the numbers in a list (None entries are skipped)."""
    values = [float(v) for v in values if v is not None]
    if len(values) == 0:
        return {"mean": None, "std": None, "n": 0}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1) if len(values) > 1 else 0.0
    return {"mean": mean, "std": math.sqrt(var), "n": len(values)}


def random_set_control(scorer, member, size, seeds, reference=None):
    """Score random layer sets of the same size. Returns {"runs": [...], "share": summary, "transfer": summary}.

    reference: the real set; each run records how many layers it shares with it.
    """
    index = scorer.index(member)
    full = scorer.full(index)
    runs = []
    for seed, layers in zip(seeds, random_layer_sets(scorer.n_layers, size, seeds)):
        result = scorer.score({index: layers})
        runs.append({"seed": int(seed), "layers": layers, "transfer": result["transfer"][scorer.names[index]],
                     "share": scorer.share(result, index, full), "harm": result["harm"],
                     "overlap": None if reference is None else len(set(layers) & set(reference))})
    return {"runs": runs, "share": summarize([r["share"] for r in runs]),
            "transfer": summarize([r["transfer"] for r in runs])}


def norm_set(norms, size):
    """The layers with the largest difference norm (attention and MLP together), as a sorted list."""
    order = sorted(range(len(norms["both"])), key=lambda layer: (-norms["both"][layer], layer))
    return sorted(order[:size])


def norm_set_control(scorer, member, size, norms=None, reference=None):
    """Score the set chosen by difference norm only, with no probe."""
    index = scorer.index(member)
    norms = layer_norms(scorer.weaver, index) if norms is None else norms
    layers = norm_set(norms, size)
    full = scorer.full(index)
    result = scorer.score({index: layers})
    return {"layers": layers, "transfer": result["transfer"][scorer.names[index]],
            "share": scorer.share(result, index, full), "harm": result["harm"],
            "overlap": None if reference is None else len(set(layers) & set(reference))}


def cross_skill_control(scorer, source, target_layers):
    """Member `source` written on the layers chosen for another member. Transfer is toward the source's own pool."""
    index = scorer.index(source)
    full = scorer.full(index)
    result = scorer.score({index: list(target_layers)})
    return {"source": scorer.names[index], "layers": sorted(int(x) for x in target_layers),
            "transfer": result["transfer"][scorer.names[index]], "share": scorer.share(result, index, full),
            "harm": result["harm"]}


def scrambled_control(scorer, member, layers, seeds):
    """The member's difference shuffled inside each tensor, written on the same layers (same norms, no knowledge)."""
    index = scorer.index(member)
    full = scorer.full(index)
    runs = []
    for seed in seeds:
        clone = scorer.weaver.shuffled(int(seed))
        result = scorer.score({index: layers}, weaver=clone)
        runs.append({"seed": int(seed), "transfer": result["transfer"][scorer.names[index]],
                     "share": scorer.share(result, index, full), "harm": result["harm"]})
    return {"runs": runs, "share": summarize([r["share"] for r in runs])}
