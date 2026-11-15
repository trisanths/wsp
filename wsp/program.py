"""The program API: describe a model as components, then compile it.

Example:

    from wsp import Program, use

    bio = use("mask_mmlupro_biology_llama1b")
    chem = use("mask_mmlupro_chemistry_llama1b")

    program = Program(
        base="unsloth/Llama-3.2-1B-Instruct",
        masks=(bio | chem).restrict(layers=range(4, 16)),
    )
    model_path = program.compile("outputs/bio_chem", catalog)

Mask algebra works on deletions:
- a | b   deletes a weight if a or b deletes it.
- a & b   deletes a weight only if a and b both delete it.
- a - b   deletes what a deletes and b does not.
- expr.restrict(layers=..., modules=...) keeps deletions only there.
"""

import math

from wsp import masks as mask_ops


class MaskExpr:
    """A node in a mask expression. Evaluate it to get a deletion mask."""

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersect(self, other)

    def __sub__(self, other):
        return Difference(self, other)

    def restrict(self, layers=None, modules=None):
        return Restrict(self, layers, modules)


class Use(MaskExpr):
    def __init__(self, component_id):
        self.component_id = component_id

    def evaluate(self, load_mask):
        return load_mask(self.component_id)

    def to_dict(self):
        return {"op": "use", "id": self.component_id}

    def __repr__(self):
        return self.component_id


class Union(MaskExpr):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, load_mask):
        return mask_ops.union(self.a.evaluate(load_mask), self.b.evaluate(load_mask))

    def to_dict(self):
        return {"op": "union", "a": self.a.to_dict(), "b": self.b.to_dict()}

    def __repr__(self):
        return f"({self.a!r} | {self.b!r})"


class Intersect(MaskExpr):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, load_mask):
        return mask_ops.intersect(self.a.evaluate(load_mask), self.b.evaluate(load_mask))

    def to_dict(self):
        return {"op": "intersect", "a": self.a.to_dict(), "b": self.b.to_dict()}

    def __repr__(self):
        return f"({self.a!r} & {self.b!r})"


class Difference(MaskExpr):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def evaluate(self, load_mask):
        return mask_ops.difference(self.a.evaluate(load_mask), self.b.evaluate(load_mask))

    def to_dict(self):
        return {"op": "difference", "a": self.a.to_dict(), "b": self.b.to_dict()}

    def __repr__(self):
        return f"({self.a!r} - {self.b!r})"


class Restrict(MaskExpr):
    def __init__(self, inner, layers=None, modules=None):
        self.inner = inner
        self.layers = None if layers is None else sorted(list(layers))
        self.modules = None if modules is None else sorted(list(modules))

    def evaluate(self, load_mask):
        return mask_ops.restrict(self.inner.evaluate(load_mask), self.layers, self.modules)

    def to_dict(self):
        return {"op": "restrict", "inner": self.inner.to_dict(), "layers": self.layers, "modules": self.modules}

    def __repr__(self):
        return f"{self.inner!r}.restrict(layers={self.layers}, modules={self.modules})"


def use(component_id):
    """Reference a deletion mask in the catalog by id."""
    return Use(component_id)


def expr_from_dict(data):
    op = data["op"]
    if op == "use":
        return Use(data["id"])
    if op == "union":
        return Union(expr_from_dict(data["a"]), expr_from_dict(data["b"]))
    if op == "intersect":
        return Intersect(expr_from_dict(data["a"]), expr_from_dict(data["b"]))
    if op == "difference":
        return Difference(expr_from_dict(data["a"]), expr_from_dict(data["b"]))
    if op == "restrict":
        return Restrict(expr_from_dict(data["inner"]), data["layers"], data["modules"])
    raise ValueError(f"unknown mask op: {op}")


class TaskVector:
    """Add weight * (fine-tuned - base) on the selected layers and modules."""

    def __init__(self, component_id, weight=1.0, layers=None, modules=None):
        self.component_id = component_id
        self.weight = weight
        self.layers = None if layers is None else sorted(list(layers))
        self.modules = None if modules is None else sorted(list(modules))

    def to_dict(self):
        return {"id": self.component_id, "weight": self.weight, "layers": self.layers, "modules": self.modules}


class GraftSpec:
    """Graft a catalog part into the host.

    at: host layer index, or None for the default depth
        round(donor_layer / donor_layers * host_layers).
    control: None, or "random" to replace the part's units with random
        units of the same count from the same donor layer (the control).
    glue_rank: None for full-rank glue, or r for rank-r w_in and w_out.
    stitch: None, or a dict that trains w_in as a stitch before heal, for
        example {"kind": "mlp", "objective": "donor_lm", "steps": 500,
        "lr": 1e-3}. Keys:
        kind       "linear", "lowrank", "mlp", or "multi".
        objective  "mse", "donor_lm" (same tokenizer only), or
                   "downstream". Default "donor_lm".
        steps, lr, batch_size, seq_len, seed
                   training settings (default 500, 1e-3, 8, 256, 0).
        rank       lowrank only (default: glue_rank).
        hidden     mlp only (default 2 * d_donor).
        offsets    multi only: host layers relative to the graft's host
                   layer, all 0 or less (default [-2, 0]).
        end_layer  downstream only (default min(donor_layer + 4, last)).
        The compiler trains the stitch on the align texts.
    sink: True feeds the part the donor's mean state at its first position
        (the attention sink) in place of the mapped vector at each
        sequence's first real position. The compiler measures it on the
        align texts, so without align texts the sink stays off. The stitch
        trains with the same sink. False turns it off.
    rms_match: True scales each other mapped vector to the donor's mean RMS
        at the input point (measured on the align texts) times a trained
        scalar that starts at 1.
    part_lora_rank: None, or r for a rank-r LoRA on the linears inside the
        donor part (the receiver), trained by heal with the glue.
    """

    def __init__(self, component_id, at=None, control=None, control_seed=0, glue_rank=None, stitch=None,
                 sink=True, rms_match=False, part_lora_rank=None):
        if control not in (None, "random"):
            raise ValueError(f"unknown control: {control!r}")
        if sink not in (True, False):
            raise ValueError(f"sink is True or False, got {sink!r}")
        if rms_match not in (True, False):
            raise ValueError(f"rms_match is True or False, got {rms_match!r}")
        if part_lora_rank is not None:
            if isinstance(part_lora_rank, bool) or not isinstance(part_lora_rank, int) or part_lora_rank < 1:
                raise ValueError(f"part_lora_rank is None or an integer of 1 or more, got {part_lora_rank!r}")
        self.component_id = component_id
        self.at = at
        self.control = control
        self.control_seed = control_seed
        self.glue_rank = glue_rank
        self.stitch = None
        if stitch is not None:
            check_stitch(stitch)
            self.stitch = dict(stitch)
        self.sink = bool(sink)
        self.rms_match = bool(rms_match)
        self.part_lora_rank = part_lora_rank

    def to_dict(self):
        data = {"id": self.component_id, "at": self.at, "control": self.control,
                "control_seed": self.control_seed, "glue_rank": self.glue_rank}
        # The newer keys are written only when they differ from the default,
        # so older programs keep the same JSON and the same hash.
        if self.stitch is not None:
            data["stitch"] = dict(self.stitch)
        if not self.sink:
            data["sink"] = False
        if self.rms_match:
            data["rms_match"] = True
        if self.part_lora_rank is not None:
            data["part_lora_rank"] = self.part_lora_rank
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data.get("at"), data.get("control"), data.get("control_seed", 0),
                   data.get("glue_rank"), data.get("stitch"), data.get("sink", True), data.get("rms_match", False),
                   data.get("part_lora_rank"))


STITCH_KINDS = ("linear", "lowrank", "mlp", "multi")
STITCH_OBJECTIVES = ("mse", "donor_lm", "downstream")


def check_stitch(stitch):
    """Raise ValueError for a stitch dict with an unknown kind or objective."""
    if not isinstance(stitch, dict):
        raise ValueError(f"a stitch is a dict, got {type(stitch).__name__}")
    if stitch.get("kind", "linear") not in STITCH_KINDS:
        raise ValueError(f"unknown stitch kind: {stitch.get('kind')!r}")
    if stitch.get("objective", "donor_lm") not in STITCH_OBJECTIVES:
        raise ValueError(f"unknown stitch objective: {stitch.get('objective')!r}")


class HealSpec:
    """Train only the graft glue on task data after assembly."""

    def __init__(self, steps=300, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384):
        self.steps = steps
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len

    def to_dict(self):
        return {"steps": self.steps, "data": self.data, "lr": self.lr,
                "batch_size": self.batch_size, "seq_len": self.seq_len}


class LoraSpec:
    """A LoRA on the host decoder layers, trained by the heal pass.

    This is the regular fine-tuning control for a graft.
    rank: the LoRA rank, or None to pick it from budget_params.
    budget_params: a trainable parameter count. With rank None, the compiler
        picks the largest rank whose LoRA count is at or under it. Use
        wsp.lora.graft_budget(report["grafts"]) to match a graft's glue
        plus part parameters.
    targets: linear names to wrap, or None for wsp.lora.DEFAULT_TARGETS.
    alpha: None for alpha = 2 * rank.
    layers: host decoder layer indices, or None for all of them.
    """

    def __init__(self, rank=None, budget_params=None, targets=None, alpha=None, layers=None):
        if rank is None and budget_params is None:
            raise ValueError("a LoRA needs a rank or a budget_params")
        self.rank = rank
        self.budget_params = budget_params
        if isinstance(targets, str):
            targets = [targets]
        self.targets = None if targets is None else list(targets)
        self.alpha = alpha
        self.layers = None if layers is None else sorted(list(layers))

    def to_dict(self):
        return {"rank": self.rank, "budget_params": self.budget_params, "targets": self.targets,
                "alpha": self.alpha, "layers": self.layers}

    @classmethod
    def from_dict(cls, data):
        return cls(rank=data.get("rank"), budget_params=data.get("budget_params"), targets=data.get("targets"),
                   alpha=data.get("alpha"), layers=data.get("layers"))


def check_int(value, what, low):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{what} is an integer of {low} or more, got {value!r}")


def check_lr(value, what):
    """A learning rate is a finite number of 0 or more. JSON can hold a string or null there."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{what} is a number of 0 or more, got {value!r}")


class TapSpec:
    """A tap: host layer `at` reads the donor's own layer `donor_layer` states.

    donor: a Hugging Face model id (any family) or a local model folder.
        The donor reads the prompt with its own tokenizer and runs only up
        to donor_layer.
    donor_layer: the donor decoder layer whose output the tap reads.
    at: the host decoder layer whose output the branch changes.
    attn_dim, heads: the size of the branch's cross-attention.
    tap_id: a short name for the tap in reports, or None.

    Taps that read the same donor share one donor run. Taps that read the
    same donor and donor_layer also share one memory.
    """

    def __init__(self, donor, donor_layer, at, attn_dim=256, heads=8, tap_id=None):
        if not isinstance(donor, str) or donor == "":
            raise ValueError(f"a tap donor is a model id, got {donor!r}")
        check_int(donor_layer, "donor_layer", 0)
        check_int(at, "at", 0)
        check_int(attn_dim, "attn_dim", 1)
        check_int(heads, "heads", 1)
        if attn_dim % heads != 0:
            raise ValueError(f"attn_dim {attn_dim} must be a multiple of heads {heads}")
        if tap_id is not None and (not isinstance(tap_id, str) or tap_id == ""):
            raise ValueError(f"tap_id is None or a name, got {tap_id!r}")
        self.donor = donor
        self.donor_layer = donor_layer
        self.at = at
        self.attn_dim = attn_dim
        self.heads = heads
        self.tap_id = tap_id

    def to_dict(self):
        return {"donor": self.donor, "donor_layer": self.donor_layer, "at": self.at, "attn_dim": self.attn_dim,
                "heads": self.heads, "tap_id": self.tap_id}

    @classmethod
    def from_dict(cls, data):
        return cls(data["donor"], data["donor_layer"], data["at"], data.get("attn_dim", 256), data.get("heads", 8),
                   data.get("tap_id"))


class TapTrainSpec:
    """How the compiler trains the taps, on (prompt, completion) pairs.

    data: a name in wsp.pairs (gsm8k_train, mmlu_aux, trivia_train).
    The loss covers only completion tokens. The gates train at gate_lr and
    the rest of each branch at branch_lr. The defaults are the round 6 MMLU
    settings. The taps train first. A LoRA or graft heal trains after
    them, with the taps frozen (joint training collapsed the tap in run 6).
    """

    def __init__(self, data="mmlu_aux", steps=500, batch_size=8, seq_len=256, branch_lr=3e-4, gate_lr=3e-2):
        if not isinstance(data, str) or data == "":
            raise ValueError(f"tap_train data is a name, got {data!r}")
        check_int(steps, "steps", 0)
        check_int(batch_size, "batch_size", 1)
        check_int(seq_len, "seq_len", 1)
        check_lr(branch_lr, "branch_lr")
        check_lr(gate_lr, "gate_lr")
        self.data = data
        self.steps = steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.branch_lr = branch_lr
        self.gate_lr = gate_lr

    def to_dict(self):
        return {"data": self.data, "steps": self.steps, "batch_size": self.batch_size, "seq_len": self.seq_len,
                "branch_lr": self.branch_lr, "gate_lr": self.gate_lr}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class Program:
    """A model described as a host plus components.

    Compile order: task vectors, then the mask, then the LoRA, then grafts,
    then taps (trained first), then heal. `base` is an older name for
    `host` and still works.
    """

    def __init__(self, host=None, masks=None, task_vectors=None, seed=0, base=None, grafts=None, heal=None,
                 lora=None, taps=None, tap_train=None):
        if host is None:
            host = base
        if host is None:
            raise ValueError("a program needs a host model")
        self.host = host
        self.masks = masks
        self.task_vectors = []
        if task_vectors is not None:
            self.task_vectors = list(task_vectors)
        self.grafts = []
        if grafts is not None:
            self.grafts = list(grafts)
        self.heal_spec = heal
        self.lora_spec = lora
        self.taps = []
        if taps is not None:
            self.taps = list(taps)
        self.tap_train_spec = tap_train
        self.seed = seed

    @property
    def base(self):
        return self.host

    def tv(self, component_id, weight=1.0, layers=None, modules=None):
        """Add a task vector. Returns self so calls can chain."""
        self.task_vectors.append(TaskVector(component_id, weight, layers, modules))
        return self

    def graft(self, component_id, at=None, control=None, control_seed=0, glue_rank=None, stitch=None, sink=True,
              rms_match=False, part_lora_rank=None):
        """Add a graft. Returns self so calls can chain."""
        self.grafts.append(GraftSpec(component_id, at, control, control_seed, glue_rank, stitch, sink, rms_match,
                                     part_lora_rank))
        return self

    def heal(self, steps=300, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384):
        """Set the heal pass. Returns self so calls can chain."""
        self.heal_spec = HealSpec(steps, data, lr, batch_size, seq_len)
        return self

    def lora(self, rank=None, budget_params=None, targets=None, alpha=None, layers=None):
        """Add a LoRA on the host. Returns self so calls can chain."""
        self.lora_spec = LoraSpec(rank, budget_params, targets, alpha, layers)
        return self

    def tap(self, donor, donor_layer, at, attn_dim=256, heads=8, tap_id=None):
        """Add a tap from a donor model's layer into host layer at. Returns self so calls can chain."""
        self.taps.append(TapSpec(donor, donor_layer, at, attn_dim, heads, tap_id))
        return self

    def tap_train(self, data="mmlu_aux", steps=500, batch_size=8, seq_len=256, branch_lr=3e-4, gate_lr=3e-2):
        """Set how the taps train. Returns self so calls can chain."""
        self.tap_train_spec = TapTrainSpec(data, steps, batch_size, seq_len, branch_lr, gate_lr)
        return self

    def component_ids(self):
        """Return every catalog id the program uses."""
        ids = []
        if self.masks is not None:
            collect_ids(self.masks, ids)
        for tv in self.task_vectors:
            ids.append(tv.component_id)
        for graft in self.grafts:
            ids.append(graft.component_id)
        return ids

    def host_component_ids(self):
        """Return the ids that must be built on the host (masks and task vectors)."""
        ids = []
        if self.masks is not None:
            collect_ids(self.masks, ids)
        for tv in self.task_vectors:
            ids.append(tv.component_id)
        return ids

    def to_dict(self):
        data = {"host": self.host, "seed": self.seed, "masks": None, "task_vectors": [], "grafts": [], "heal": None}
        if self.masks is not None:
            data["masks"] = self.masks.to_dict()
        for tv in self.task_vectors:
            data["task_vectors"].append(tv.to_dict())
        for graft in self.grafts:
            data["grafts"].append(graft.to_dict())
        if self.heal_spec is not None:
            data["heal"] = self.heal_spec.to_dict()
        if self.lora_spec is not None:
            data["lora"] = self.lora_spec.to_dict()
        # The tap keys are written only when used, so older programs keep
        # the same JSON and the same hash.
        if len(self.taps) > 0:
            data["taps"] = []
            for tap in self.taps:
                data["taps"].append(tap.to_dict())
        if self.tap_train_spec is not None:
            data["tap_train"] = self.tap_train_spec.to_dict()
        return data

    @classmethod
    def from_dict(cls, data):
        masks = None
        if data.get("masks") is not None:
            masks = expr_from_dict(data["masks"])
        task_vectors = []
        for tv in data.get("task_vectors", []):
            task_vectors.append(TaskVector(tv["id"], tv["weight"], tv["layers"], tv["modules"]))
        grafts = []
        for graft in data.get("grafts", []):
            grafts.append(GraftSpec.from_dict(graft))
        heal = None
        if data.get("heal") is not None:
            heal = HealSpec(**data["heal"])
        lora = None
        if data.get("lora") is not None:
            lora = LoraSpec.from_dict(data["lora"])
        taps = []
        for tap in data.get("taps", []):
            taps.append(TapSpec.from_dict(tap))
        tap_train = None
        if data.get("tap_train") is not None:
            tap_train = TapTrainSpec.from_dict(data["tap_train"])
        host = data.get("host")
        if host is None:
            host = data["base"]
        return cls(host=host, masks=masks, task_vectors=task_vectors, seed=data.get("seed", 0),
                   grafts=grafts, heal=heal, lora=lora, taps=taps, tap_train=tap_train)

    def compile(self, output_dir, catalog, **options):
        """Build the model and save it to output_dir. Return output_dir.

        options pass through to wsp.compiler.compile_program
        (align_texts, heal_texts, device, cache_dir, tap_pairs, heal_pairs).
        """
        from wsp.compiler import compile_program

        return compile_program(self, catalog, output_dir, **options)

    def __repr__(self):
        return (f"Program(host={self.host!r}, masks={self.masks!r}, "
                f"task_vectors={len(self.task_vectors)}, grafts={len(self.grafts)}, "
                f"lora={self.lora_spec is not None}, taps={len(self.taps)})")


def collect_ids(expr, ids):
    if isinstance(expr, Use):
        ids.append(expr.component_id)
    elif isinstance(expr, Restrict):
        collect_ids(expr.inner, ids)
    else:
        collect_ids(expr.a, ids)
        collect_ids(expr.b, ids)
