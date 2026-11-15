"""The one adapter that knows the program JSON layout.

The schema is locked in wsp/program.py. A program dict looks like this:

    {"host": "unsloth/Llama-3.2-1B-Instruct", "seed": 0,
     "masks": null, "task_vectors": [],
     "grafts": [{"id": "qmath_l14_c3", "at": 9, "control": null, "control_seed": 0,
                 "glue_rank": null}],
     "heal": {"steps": 300, "data": "gsm8k_train", "lr": 0.0003,
              "batch_size": 8, "seq_len": 384}}

Everything else in wsp/swarm reads and writes programs through the functions
here, so a schema change touches only this file. normalize() round-trips a
dict through Trisanth's real class, Program.from_dict(d).to_dict(), so part 3
never invents a field. Old mask-only dicts that use "base" still load.

A graft's "glue_rank" is None for full-rank glue, or an int r for rank-r
w_in and w_out. Every builder here that copies a graft keeps its glue rank,
so a moved graft, a swapped part and a random-part control all keep the
same glue.
"""

from wsp.program import GraftSpec, HealSpec, Program, collect_ids, expr_from_dict

CONTROL_RANDOM = "random"


def normalize(data):
    """Return the canonical program dict for a program dict.

    Accepts "base" as an alias of "host". Raises ValueError with a short
    message when the dict cannot be read as a program.
    """
    if not isinstance(data, dict):
        raise ValueError("a program must be a JSON object")
    try:
        program = Program.from_dict(data)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("cannot read program: " + type(error).__name__ + " " + str(error))
    return program.to_dict()


def to_program(program):
    """Return a wsp.program.Program object for a program dict."""
    return Program.from_dict(program)


def make_graft(part_id, at=None, control=None, control_seed=0, glue_rank=None):
    """Return one graft dict. glue_rank None is full-rank glue."""
    return GraftSpec(part_id, at, control, control_seed, glue_rank).to_dict()


def make_heal(steps=300, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384):
    """Return one heal dict (the defaults are part 2's defaults)."""
    return HealSpec(steps, data, lr, batch_size, seq_len).to_dict()


def make_program(host, grafts=None, heal=None, seed=0):
    """Return a program dict with the given host, graft dicts and heal dict."""
    program = Program(host=host, seed=seed)
    data = program.to_dict()
    if grafts is not None:
        for graft in grafts:
            data["grafts"].append(dict(graft))
    if heal is not None:
        data["heal"] = dict(heal)
    return normalize(data)


# Reading a program.

def host(program):
    if program.get("host") is not None:
        return program["host"]
    return program.get("base")


def seed(program):
    return program.get("seed", 0)


def grafts(program):
    """Return a list of copies of the program's graft dicts, in order."""
    out = []
    for graft in program.get("grafts") or []:
        out.append(dict(graft))
    return out


def heal(program):
    """Return a copy of the heal dict, or None when the program has no heal."""
    value = program.get("heal")
    if value is None:
        return None
    return dict(value)


def heal_steps(program):
    """Return the heal steps, or None when the program has no heal."""
    value = program.get("heal")
    if value is None:
        return None
    return value.get("steps")


def heal_data(program):
    value = program.get("heal")
    if value is None:
        return None
    return value.get("data")


def lora(program):
    """Return a copy of the host LoRA dict ({"rank", "budget_params", ...}), or None."""
    value = program.get("lora")
    if value is None:
        return None
    return dict(value)


def make_lora_program(host, rank, heal, seed=0):
    """A host LoRA program: no grafts, a rank-`rank` LoRA on the host, trained by the heal pass."""
    program = Program(host=host, seed=seed).lora(rank=rank)
    data = program.to_dict()
    data["heal"] = dict(heal)
    return normalize(data)


def has_host_edits(program):
    """True when the program has host masks or task vectors."""
    if program.get("masks") is not None:
        return True
    return len(program.get("task_vectors") or []) > 0


def mask_ids(program):
    """Return the catalog ids used by the host mask expression."""
    masks = program.get("masks")
    if masks is None:
        return []
    ids = []
    collect_ids(expr_from_dict(masks), ids)
    return ids


def mask_restrict_layers(program):
    """Return every layer list named by a restrict() in the mask expression."""
    masks = program.get("masks")
    found = []
    if masks is None:
        return found
    pending = [masks]
    while len(pending) > 0:
        node = pending.pop()
        if node.get("op") == "restrict":
            if node.get("layers") is not None:
                found.append(list(node["layers"]))
            pending.append(node["inner"])
        elif node.get("op") in ("union", "intersect", "difference"):
            pending.append(node["a"])
            pending.append(node["b"])
    return found


def task_vector_ids(program):
    ids = []
    for tv in program.get("task_vectors") or []:
        ids.append(tv["id"])
    return ids


# Reading a graft.

def graft_id(graft):
    return graft["id"]


def graft_at(graft):
    """Return the host depth, or None for the default depth."""
    return graft.get("at")


def graft_control(graft):
    """Return None, or "random" for a random-part control graft."""
    return graft.get("control")


def graft_control_seed(graft):
    return graft.get("control_seed", 0)


def graft_glue_rank(graft):
    """Return None for full-rank glue, or the int rank of the glue."""
    return graft.get("glue_rank")


# Building changed copies. Each returns a new normalized dict.

def with_grafts(program, new_grafts):
    data = normalize(program)
    data["grafts"] = []
    for graft in new_grafts:
        data["grafts"].append(dict(graft))
    return normalize(data)


def with_heal(program, new_heal):
    data = normalize(program)
    if new_heal is None:
        data["heal"] = None
    else:
        data["heal"] = dict(new_heal)
    return normalize(data)


def without_heal(program):
    """The same program with heal set to None. Compiling it is a fit probe."""
    return with_heal(program, None)


def with_heal_steps(program, steps, default_data="gsm8k_train"):
    """Change the heal steps, keeping the other heal settings (or defaults)."""
    current = heal(program)
    if current is None:
        current = make_heal(steps=steps, data=default_data)
    current["steps"] = steps
    return with_heal(program, current)


def graft_with_at(graft, at):
    return make_graft(graft_id(graft), at, graft_control(graft), graft_control_seed(graft),
                      graft_glue_rank(graft))


def graft_with_id(graft, part_id):
    return make_graft(part_id, graft_at(graft), graft_control(graft), graft_control_seed(graft),
                      graft_glue_rank(graft))


def graft_with_glue_rank(graft, glue_rank):
    return make_graft(graft_id(graft), graft_at(graft), graft_control(graft), graft_control_seed(graft),
                      glue_rank)


def as_random_control(graft, control_seed):
    """The random-part control of a graft: same slot and glue rank, random units of the same count."""
    return make_graft(graft_id(graft), graft_at(graft), CONTROL_RANDOM, control_seed, graft_glue_rank(graft))
