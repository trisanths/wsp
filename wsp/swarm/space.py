"""The graft program space: hashing, validation, random programs and edits.

Programs are dicts in the locked schema (wsp/program.py). This module reads
and builds them only through program_schema, so a schema change does not
touch it. The catalog is a wsp.catalog.Catalog.

A graft's "slot" is (donor model, donor layer, host depth). The compile-time
fit R squared depends only on the slot, never on which units the part holds,
so fit can veto a placement and help choose `at`, but cannot rank parts.

Each graft also has a glue rank: None for full-rank glue, or an int r. The
allowed ranks come from config "glue_ranks" (default [None, 64]); rank 64
was as good as full glue at 7% of the parameters in round 2.
"""

import hashlib
import json

from wsp.swarm import program_schema as ps
from wsp.swarm.config import DEFAULTS, heal_data_allowed

GRAFT_TYPES = ["mlp_cluster", "attn_heads", "expert", "block"]

# The glue ranks a graft may use when config has no "glue_ranks" key.
DEFAULT_GLUE_RANKS = [None, 64]

# Mutation moves and rates. The first five follow the edit menu in papers.md
# section 7.2 and the move set of its recommendation 5: add 25%, remove 15%,
# swap 15%, change layer restriction 20% (here: move `at`), consensus 15%
# (here: change heal steps), and the fresh-draft 10% goes to "move", whose
# third option resets `at` to the default depth. "rank" (change one graft's
# glue rank) gets 10%, taken from the other five in proportion (each times
# 0.9). Final odds: add 22.5%, drop 13.5%, swap 13.5%, move 27%, heal 13.5%,
# rank 10%.
MOVE_RATES = [("add", 0.225), ("drop", 0.135), ("swap", 0.135), ("move", 0.27), ("heal", 0.135),
              ("rank", 0.10)]


# Hashing.

def canonical_json(program):
    """Sorted keys, grafts in listed order (order matters for compile), no whitespace."""
    return json.dumps(ps.normalize(program), sort_keys=True, separators=(",", ":"))


def program_hash(program):
    """sha256 of the canonical JSON, first 16 hex chars."""
    return hashlib.sha256(canonical_json(program).encode("utf-8")).hexdigest()[:16]


def probe_hash(program):
    """The hash of the program with heal set to None (the probe cache key)."""
    return program_hash(ps.without_heal(program))


# Catalog helpers.

def graftable_parts(catalog):
    """Every catalog record with a graftable type, sorted by id."""
    parts = []
    for part_id in sorted(catalog.records):
        record = catalog.records[part_id]
        if record.get("type") in GRAFT_TYPES:
            parts.append(record)
    return parts


def part_layer(record):
    """The donor layer of a part: "layer", or the first of "layers" for a block."""
    if record.get("layer") is not None:
        return record["layer"]
    layers = record.get("layers")
    if layers is not None and len(layers) > 0:
        return layers[0]
    return None


def host_layers(config, catalog=None):
    """L_host: from a catalog record for the host with a "layers" count, else config."""
    if catalog is not None and config["host"] in catalog.records:
        record = catalog.records[config["host"]]
        if isinstance(record.get("layers"), int):
            return record["layers"]
    return config["host_layers"]


def glue_ranks(config):
    """The glue ranks a graft may use: config "glue_ranks", else [None, 64]. None is full rank."""
    ranks = config.get("glue_ranks")
    if ranks is None:
        ranks = DEFAULT_GLUE_RANKS
    return list(ranks)


def rank_text(glue_rank):
    """"full" for full-rank glue, else "r<rank>" (for example r64)."""
    if glue_rank is None:
        return "full"
    return "r" + str(glue_rank)


def donor_layer_count(record, config):
    """L_donor for a part's model, from config "donor_layers", or None if unknown."""
    return config["donor_layers"].get(record.get("model"))


def default_at(record, config, n_host=None):
    """The default host depth round(l / L_donor * L_host), clamped (same rule as wsp.assemble).

    Returns None when the donor's layer count is not in config.
    """
    if n_host is None:
        n_host = config["host_layers"]
    layer = part_layer(record)
    n_donor = donor_layer_count(record, config)
    if layer is None or n_donor is None:
        return None
    k = int(round(layer / n_donor * n_host))
    if k < 0:
        k = 0
    if k > n_host - 1:
        k = n_host - 1
    return k


def resolved_at(graft, catalog, config):
    """The host depth a graft lands at: its `at`, or the default depth (None if unknown)."""
    at = ps.graft_at(graft)
    if at is not None:
        return at
    if ps.graft_id(graft) not in catalog.records:
        return None
    return default_at(catalog.records[ps.graft_id(graft)], config, host_layers(config, catalog))


def graft_key(part_id, at, control=None):
    """The fit key used in log rows: "<id>@<host layer>", with "~random" for a control."""
    if at is None:
        key = part_id + "@default"
    else:
        key = part_id + "@" + str(at)
    if control is not None:
        key = key + "~" + control
    return key


def slot_key(donor, donor_layer, at):
    """The fit slot: fit R squared depends only on donor, donor layer and host depth."""
    return str(donor) + "|l" + str(donor_layer) + "@" + str(at)


def graft_slot(graft, catalog, config):
    """The slot key of a graft, or None when its record or depth is unknown."""
    part_id = ps.graft_id(graft)
    if part_id not in catalog.records:
        return None
    at = resolved_at(graft, catalog, config)
    if at is None:
        return None
    record = catalog.records[part_id]
    return slot_key(record.get("model"), part_layer(record), at)


def depth_band(at, n_host):
    """"early", "middle" or "late" for a host depth (thirds of the host), "unknown" if None."""
    if at is None:
        return "unknown"
    if at < n_host / 3.0:
        return "early"
    if at < 2.0 * n_host / 3.0:
        return "middle"
    return "late"


def graft_pattern(graft, program, catalog, config):
    """The judge's pattern for one graft: part id, depth band, and whether heal is used."""
    steps = ps.heal_steps(program)
    healed = "noheal"
    if steps is not None and steps > 0:
        healed = "heal"
    band = depth_band(resolved_at(graft, catalog, config), host_layers(config, catalog))
    return ps.graft_id(graft) + "|" + band + "|" + healed


def program_patterns(program, catalog, config):
    """The patterns of every graft in a program, in graft order."""
    patterns = []
    for graft in ps.grafts(program):
        patterns.append(graft_pattern(graft, program, catalog, config))
    return patterns


# Validation.

def validate(program, catalog, config=None):
    """Return a list of error strings. An empty list means the program is valid."""
    if config is None:
        config = DEFAULTS
    try:
        program = ps.normalize(program)
    except ValueError as error:
        return [str(error)]
    errors = []
    host = ps.host(program)
    n_host = host_layers(config, catalog)
    if host != config["host"]:
        errors.append("host " + str(host) + " is not this run's host " + config["host"])

    grafts = ps.grafts(program)
    if len(grafts) > config["max_grafts"]:
        errors.append("at most " + str(config["max_grafts"]) + " grafts, got " + str(len(grafts)))
    seen = []
    for graft in grafts:
        errors.extend(validate_graft(graft, catalog, n_host, glue_ranks(config)))
        pair = (ps.graft_id(graft), ps.graft_at(graft))
        if pair in seen:
            errors.append("graft " + str(pair[0]) + " at " + str(pair[1]) + " appears twice")
        seen.append(pair)

    errors.extend(validate_heal(ps.heal(program), config))
    errors.extend(validate_host_edits(program, catalog, n_host))
    return errors


def validate_graft(graft, catalog, n_host, ranks):
    """Errors for one graft: known graftable part, `at` in range, glue rank in ranks."""
    errors = []
    part_id = ps.graft_id(graft)
    if not isinstance(part_id, str):
        return ["graft id must be a string, got " + repr(part_id)]
    if part_id not in catalog.records:
        return ["unknown part id: " + str(part_id)]
    kind = catalog.records[part_id].get("type")
    if kind not in GRAFT_TYPES:
        errors.append(str(part_id) + " has type " + str(kind) + ", which cannot be grafted")
    at = ps.graft_at(graft)
    if at is not None:
        if isinstance(at, bool) or not isinstance(at, int):
            errors.append(str(part_id) + ": at must be an integer, got " + repr(at))
        elif at < 0 or at > n_host - 1:
            errors.append(str(part_id) + ": at " + str(at) + " is outside 0.." + str(n_host - 1))
    rank = ps.graft_glue_rank(graft)
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int)):
        errors.append(str(part_id) + ": glue_rank must be None or an integer, got " + repr(rank))
    elif rank not in ranks:
        errors.append(str(part_id) + ": glue_rank " + str(rank) + " not in " + str(ranks))
    return errors


def validate_heal(heal, config):
    if heal is None:
        return []
    errors = []
    if heal.get("steps") not in config["heal_steps_allowed"]:
        errors.append("heal steps " + str(heal.get("steps")) + " not in " + str(config["heal_steps_allowed"]))
    if heal.get("data") not in heal_data_allowed(config):
        errors.append("heal data " + str(heal.get("data")) + " not in " + str(heal_data_allowed(config)))
    if heal.get("lr") not in config["heal_lr_allowed"]:
        errors.append("heal lr " + str(heal.get("lr")) + " not in " + str(config["heal_lr_allowed"]))
    if heal.get("batch_size") not in config["heal_batch_size_allowed"]:
        errors.append("heal batch_size " + str(heal.get("batch_size")) + " not allowed")
    if heal.get("seq_len") not in config["heal_seq_len_allowed"]:
        errors.append("heal seq_len " + str(heal.get("seq_len")) + " not allowed")
    return errors


def validate_host_edits(program, catalog, n_host):
    """Host masks and task vectors: version 1 rules (built on the host, layers in range)."""
    errors = []
    host = ps.host(program)
    for mask_id in ps.mask_ids(program):
        if mask_id not in catalog.records:
            errors.append("unknown mask id: " + str(mask_id))
            continue
        record = catalog.records[mask_id]
        if record.get("type") != "deletion_mask":
            errors.append(str(mask_id) + " is a " + str(record.get("type")) + ", not a deletion_mask")
        elif record.get("base") != host:
            errors.append(str(mask_id) + " is built on " + str(record.get("base")) + ", not the host")
    for layers in ps.mask_restrict_layers(program):
        for layer in layers:
            if layer < 0 or layer > n_host - 1:
                errors.append("restrict layer " + str(layer) + " is outside 0.." + str(n_host - 1))
    for tv_id in ps.task_vector_ids(program):
        if tv_id not in catalog.records:
            errors.append("unknown task vector id: " + str(tv_id))
            continue
        record = catalog.records[tv_id]
        if record.get("type") != "task_vector":
            errors.append(str(tv_id) + " is a " + str(record.get("type")) + ", not a task_vector")
        elif record.get("base") != host:
            errors.append(str(tv_id) + " is built on " + str(record.get("base")) + ", not the host")
    return errors


# Random programs and edits.

def random_heal(rng, config):
    steps = rng.choice(config["heal_steps_allowed"])
    return ps.make_heal(steps=steps, data=heal_data_allowed(config)[0],
                        lr=config["heal_lr_allowed"][0],
                        batch_size=config["heal_batch_size_allowed"][0],
                        seq_len=config["heal_seq_len_allowed"][0])


def heal_with_steps(steps, config):
    return ps.make_heal(steps=steps, data=heal_data_allowed(config)[0],
                        lr=config["heal_lr_allowed"][0],
                        batch_size=config["heal_batch_size_allowed"][0],
                        seq_len=config["heal_seq_len_allowed"][0])


def taken_pairs(grafts):
    pairs = []
    for graft in grafts:
        pairs.append((ps.graft_id(graft), ps.graft_at(graft)))
    return pairs


def random_program(rng, catalog, config):
    """One to three random grafts, each at a random host depth with a random glue rank, and random heal steps."""
    parts = graftable_parts(catalog)
    n_host = host_layers(config, catalog)
    ranks = glue_ranks(config)
    count = rng.randint(1, min(3, config["max_grafts"]))
    grafts = []
    attempts = 0
    while len(grafts) < count and attempts < 50:
        attempts += 1
        record = rng.choice(parts)
        at = rng.randrange(n_host)
        if (record["id"], at) in taken_pairs(grafts):
            continue
        grafts.append(ps.make_graft(record["id"], at, glue_rank=rng.choice(ranks)))
    return ps.make_program(config["host"], grafts, random_heal(rng, config))


def pick_move(rng):
    roll = rng.random()
    total = 0.0
    for name, rate in MOVE_RATES:
        total += rate
        if roll < total:
            return name
    return MOVE_RATES[-1][0]


def swap_candidates(record, parts):
    """Parts that share the donor model or the kind of a record (not the record itself)."""
    close = []
    for other in parts:
        if other["id"] == record["id"]:
            continue
        if other.get("model") == record.get("model") or other.get("type") == record.get("type"):
            close.append(other)
    return close


def apply_move(move, program, rng, catalog, config):
    """Apply one mutation move. Return (child, note), or (None, "") if the move does not apply."""
    parts = graftable_parts(catalog)
    n_host = host_layers(config, catalog)
    grafts = ps.grafts(program)

    if move == "add":
        if len(grafts) >= config["max_grafts"]:
            return None, ""
        record = rng.choice(parts)
        at = default_at(record, config, n_host)
        if at is None or rng.random() < 0.5:
            at = rng.randrange(n_host)
        rank = rng.choice(glue_ranks(config))
        grafts.append(ps.make_graft(record["id"], at, glue_rank=rank))
        note = "add " + short_name(record["id"]) + "@" + str(at) + " with " + rank_text(rank) + " glue"
        return ps.with_grafts(program, grafts), note

    if len(grafts) == 0:
        return None, ""
    index = rng.randrange(len(grafts))
    graft = grafts[index]

    if move == "drop":
        if len(grafts) < 2:
            return None, ""
        grafts.pop(index)
        return ps.with_grafts(program, grafts), "drop " + short_name(ps.graft_id(graft))

    if move == "swap":
        if ps.graft_id(graft) not in catalog.records:
            return None, ""
        record = catalog.records[ps.graft_id(graft)]
        pool = swap_candidates(record, parts)
        if len(pool) == 0 or rng.random() < 0.3:
            pool = parts
        new_record = rng.choice(pool)
        if new_record["id"] == record["id"]:
            return None, ""
        at = ps.graft_at(graft)
        if at is None or rng.random() < 0.5:
            at = default_at(new_record, config, n_host)
        # The new part keeps the old graft's glue rank.
        grafts[index] = ps.make_graft(new_record["id"], at, glue_rank=ps.graft_glue_rank(graft))
        note = "swap " + short_name(record["id"]) + " for " + short_name(new_record["id"])
        return ps.with_grafts(program, grafts), note

    if move == "move":
        at = resolved_at(graft, catalog, config)
        if rng.random() < 1.0 / 3.0:
            record = catalog.records.get(ps.graft_id(graft))
            new_at = None
            if record is not None:
                new_at = default_at(record, config, n_host)
            note = "reset " + short_name(ps.graft_id(graft)) + " to its default depth"
        else:
            if at is None:
                at = n_host // 2
            new_at = at + rng.choice([-2, -1, 1, 2])
            if new_at < 0:
                new_at = 0
            if new_at > n_host - 1:
                new_at = n_host - 1
            note = "move " + short_name(ps.graft_id(graft)) + " from " + str(at) + " to " + str(new_at)
        if new_at == ps.graft_at(graft):
            return None, ""
        grafts[index] = ps.graft_with_at(graft, new_at)
        return ps.with_grafts(program, grafts), note

    if move == "heal":
        current = ps.heal_steps(program)
        options = []
        for steps in config["heal_steps_allowed"]:
            if steps != current:
                options.append(steps)
        steps = rng.choice(options)
        return ps.with_heal(program, heal_with_steps(steps, config)), "heal " + str(steps) + " steps"

    if move == "rank":
        current = ps.graft_glue_rank(graft)
        options = []
        for rank in glue_ranks(config):
            if rank != current:
                options.append(rank)
        if len(options) == 0:
            return None, ""
        rank = rng.choice(options)
        grafts[index] = ps.graft_with_glue_rank(graft, rank)
        note = ("glue rank of " + short_name(ps.graft_id(graft)) + " from " + rank_text(current) +
                " to " + rank_text(rank))
        return ps.with_grafts(program, grafts), note

    raise ValueError("unknown move: " + move)


def mutate_with_note(program, rng, catalog, config):
    """One mutation. Return (child, note). The child is valid and differs from the parent.

    Falls back to a random program after 20 failed tries.
    """
    parent_hash = program_hash(program)
    for attempt in range(20):
        move = pick_move(rng)
        child, note = apply_move(move, program, rng, catalog, config)
        if child is None:
            continue
        if program_hash(child) == parent_hash:
            continue
        if len(validate(child, catalog, config)) > 0:
            continue
        return child, note
    return random_program(rng, catalog, config), "fresh random draft"


def mutate(program, rng, catalog, config):
    """One mutation move (see MOVE_RATES). Returns a valid child program."""
    child, note = mutate_with_note(program, rng, catalog, config)
    return child


def crossover(a, b, rng, config=None):
    """Mix two graft lists. Host, seed and host edits come from a; heal from either.

    Each graft is copied whole, so it keeps its own `at` and glue rank.
    """
    if config is None:
        config = DEFAULTS
    pool = ps.grafts(a) + ps.grafts(b)
    child_grafts = []
    for graft in pool:
        if rng.random() < 0.5:
            child_grafts.append(graft)
    if len(child_grafts) == 0 and len(pool) > 0:
        child_grafts.append(rng.choice(pool))
    unique = []
    for graft in child_grafts:
        if (ps.graft_id(graft), ps.graft_at(graft)) in taken_pairs(unique):
            continue
        if len(unique) >= config["max_grafts"]:
            break
        unique.append(graft)
    heal = ps.heal(a)
    if rng.random() < 0.5:
        heal = ps.heal(b)
    child = ps.with_grafts(a, unique)
    return ps.with_heal(child, heal)


# Text and controls.

def short_name(part_id):
    """A short part name: stub_qmath_l14_c3 -> qmath.l14.c3."""
    name = str(part_id)
    for prefix in ("stub_", "part_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", ".")


def describe(program):
    """Short text, for example: host + qmath.l14.c3@9 + rand(qcoder.l10.c0)@5/r64, heal 300 on gsm8k_train.

    A "/r<rank>" suffix marks low-rank glue; full-rank glue has no suffix.
    """
    program = ps.normalize(program)
    text = "host"
    if ps.has_host_edits(program):
        text = "host[edited]"
    for graft in ps.grafts(program):
        name = short_name(ps.graft_id(graft))
        if ps.graft_control(graft) is not None:
            name = "rand(" + name + ")"
        at = ps.graft_at(graft)
        if at is None:
            text += " + " + name + "@default"
        else:
            text += " + " + name + "@" + str(at)
        if ps.graft_glue_rank(graft) is not None:
            text += "/" + rank_text(ps.graft_glue_rank(graft))
    steps = ps.heal_steps(program)
    if steps is None:
        text += ", no heal"
    else:
        text += ", heal " + str(steps) + " on " + str(ps.heal_data(program))
    return text


def random_glue_control(program, rng, catalog=None):
    """Control 1: the same glue and heal around a random part of the same size from the same donor layer.

    Each graft becomes {"id", "at", "control": "random", "control_seed": s,
    "glue_rank"}, with the glue rank of the graft it replaces.
    """
    grafts = []
    for graft in ps.grafts(program):
        grafts.append(ps.as_random_control(graft, rng.randrange(1, 1000000)))
    return ps.with_grafts(program, grafts)


def no_graft_control(program):
    """Control 2 in the env: drop every graft and keep the heal (the heal-only gain)."""
    return ps.with_grafts(program, [])
