"""Identity-seam skill tops: a member that is a true fine-tune of the host needs no learned stitch.

A true fine-tune keeps the host's shapes, tokenizer ids, and positional base, and
every weight moved only a little. So the host's state at layer k is already in
the member's coordinates, to the precision the fine-tune left. The network is
the one in wsp.tops with one change: the seam between the trunk and a top is
the identity.

    host layers 0..cut-1 (shared trunk) -> member layers cut..end -> member norm -> member head

Nothing trains. There is no sink, because the trunk's state at the first token
is already the sink state the member expects. The router is the closed-form
ridge router of wsp.tops (SkillTops.fit_router): one-hot labels regressed on the
trunk's mean state at the cut, over the prompt, lambda 0.01, with a base margin.

Use:
    net = IdentityTops(host_model, host_tokenizer, base_id=host_id)
    net.add_top("code", member_id, cut, lineage=relation_dict)
    net.fit_router({"base": [...], "code": [...]}, lam=0.01)

Stored params of a top are its layers from the cut, its norm, and its head. The
stitch and the sink count as zero.

Gate. add_top refuses a member whose shapes or positional base differ from the
host (a layer-for-layer top needs the same widths and the same layer count) and
a lineage relation that is not fine_tune or fine_tune_of_ancestor. The
relation comes from wsp.lineage.relation. A caller that has none passes
lineage=None and only the shape gate runs.
"""

import torch
from torch import nn

from wsp import tops as tops_lib
from wsp.parts import _text_config
from wsp.tops import RESERVED, SkillTops, TopModule, count, layer_dtype

TRUE_FINE_TUNE = ("fine_tune", "fine_tune_of_ancestor")
SHAPE_FIELDS = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim")


class IdentityTop(TopModule):
    """A top with the identity seam: the trunk state goes into the member's layers unchanged."""

    def map_state(self, h, first):
        return h.to(layer_dtype(self.layers))


def shape_report(host_config, member_config):
    """Compare two configs on the fields a layer-for-layer top needs. Return (equal, differences)."""
    host = _text_config(host_config)
    member = _text_config(member_config)
    differences = {}
    for field in SHAPE_FIELDS:
        a = getattr(host, field, None)
        b = getattr(member, field, None)
        if field == "head_dim" and (a is None or b is None):
            continue
        if a != b:
            differences[field] = (a, b)
    a = getattr(host, "rope_theta", None)
    b = getattr(member, "rope_theta", None)
    if a != b:
        differences["rope_theta"] = (a, b)
    if bool(getattr(host, "tie_word_embeddings", False)) != bool(getattr(member, "tie_word_embeddings", False)):
        differences["tie_word_embeddings"] = (getattr(host, "tie_word_embeddings", False),
                                              getattr(member, "tie_word_embeddings", False))
    return len(differences) == 0, differences


def relation_kind(lineage):
    """The kind string of a lineage relation (a dict from wsp.lineage.relation, or the kind itself)."""
    if lineage is None:
        return None
    if isinstance(lineage, str):
        return lineage
    return lineage.get("kind")


def require_true_fine_tune(host_config, member_config, lineage=None):
    """Raise ValueError unless the member is a layer-for-layer fine-tune of the host."""
    equal, differences = shape_report(host_config, member_config)
    if not equal:
        raise ValueError(f"the member is not shaped like the host, so an identity seam cannot hold: {differences}")
    kind = relation_kind(lineage)
    if lineage is not None and kind not in TRUE_FINE_TUNE:
        raise ValueError(f"lineage says {kind!r}; an identity seam needs one of {TRUE_FINE_TUNE}")


class IdentityTops(SkillTops):
    """SkillTops where every top has the identity seam and nothing is fitted but the router."""

    def add_top(self, name, donor, cut_host, cut_donor=None, control=None, control_seed=0, donor_tokenizer=None,
                check=True, device=None, cache_dir=None, lineage=None, **ignored):
        """Add an identity-seam top built from a member (repo id, directory, or loaded model).

        The member's layers start at the same index as the host's cut (the
        layer counts match). control="scrambled" rewrites the member's
        layers with wsp.scramble before use. Extra keyword arguments that
        belong to learned stitches (stitch_kind, stitch_hidden) are ignored.
        """
        if isinstance(donor, nn.Module):
            member_config = donor.config
        else:
            member_config = tops_lib.donor_config_of(donor)
        require_true_fine_tune(self.config, member_config, lineage)
        if cut_donor is not None and cut_donor != cut_host:
            raise ValueError("an identity seam keeps the layer index: cut_donor must equal cut_host")
        if ignored.get("cross_tokenizer"):
            raise ValueError("an identity seam needs the host's tokenizer")
        top = super().add_top(name, donor, cut_host, cut_donor=cut_host, stitch_kind="linear", own_head=True,
                              control=control, control_seed=control_seed, donor_tokenizer=donor_tokenizer,
                              check=check, device=device, cache_dir=cache_dir)
        top.stitch = nn.Identity()
        top.__class__ = IdentityTop
        top.has_sink = False
        top.spec["seam"] = "identity"
        top.spec["stitch_kind"] = "identity"
        top.spec["lineage"] = relation_kind(lineage)
        top.accepted = True
        top.flag = None
        return top

    def fit_stitch(self, *args, **kwargs):
        raise NotImplementedError("an identity-seam top has no stitch to fit")


def fit_identity_router(net, texts_by_class, lam=0.01, scale=tops_lib.ROUTER_SCALE, base_margin=0.0,
                        probe_by_class=None, margins=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0), max_length=384, batch_size=8):
    """Fit the closed-form router (lambda 0.01), then optionally pick the base margin on probe texts.

    Returns the fit report with the margin table under "tune" when probe texts are given.
    """
    report = net.fit_router(texts_by_class, lam=lam, scale=scale, base_margin=base_margin, max_length=max_length,
                            batch_size=batch_size)
    if probe_by_class is not None:
        report["tune"] = net.tune_base_margin(probe_by_class, margins=margins, max_length=max_length)
        report["base_margin"] = net.base_margin
    return report


def route_accuracy(net, texts_by_class, max_length=384, batch_size=8):
    """Share of each class's texts the router sends to that class, and the overall mean."""
    out = {}
    total = 0
    hits = 0
    for name, texts in texts_by_class.items():
        picks = net.route_texts(texts, max_length, batch_size)
        right = sum(1 for pick in picks if pick == name)
        out[name] = right / max(1, len(picks))
        total += len(picks)
        hits += right
    out["overall"] = hits / max(1, total)
    return out


def identity_sizes(base_config, tops, router_classes=None):
    """Stored and per-token params of an identity-seam network, from configs alone.

    tops: list of dicts with name, donor_config, cut_host. The stitch and the
    sink count as zero. Matches count_params of the built network.
    """
    plain = tops_lib.sizes_from_configs(base_config, tops, router_classes=router_classes)
    stored = plain["base_total"] + plain["router"]
    run = {RESERVED: plain["base_run"]}
    entries = {}
    for name, entry in plain["tops"].items():
        entry = dict(entry)
        entry["stitch"] = 0
        entry["sink"] = 0
        entry["stored"] = entry["layers"] + entry["norm"] + entry["head"] + entry["lora"]
        entry["run"] = entry["trunk"] + entry["stored"]
        entries[name] = entry
        stored += entry["stored"]
        run[name] = entry["run"]
    return {"base_total": plain["base_total"], "base_run": plain["base_run"], "tops": entries,
            "router": plain["router"], "stored": stored, "stored_over_base": stored / plain["base_total"],
            "run": run}


def trainable_count(net):
    """Parameters that any fit would train: zero for an identity network (the router is closed form)."""
    total = 0
    for top in net.tops.values():
        total += count(top.trainable())
    return total
