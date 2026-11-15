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
    """

    def __init__(self, component_id, at=None, control=None, control_seed=0, glue_rank=None):
        if control not in (None, "random"):
            raise ValueError(f"unknown control: {control!r}")
        self.component_id = component_id
        self.at = at
        self.control = control
        self.control_seed = control_seed
        self.glue_rank = glue_rank

    def to_dict(self):
        return {"id": self.component_id, "at": self.at, "control": self.control,
                "control_seed": self.control_seed, "glue_rank": self.glue_rank}


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


class Program:
    """A model described as a host plus components.

    Compile order: task vectors, then the mask, then grafts, then heal.
    `base` is an older name for `host` and still works.
    """

    def __init__(self, host=None, masks=None, task_vectors=None, seed=0, base=None, grafts=None, heal=None):
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
        self.seed = seed

    @property
    def base(self):
        return self.host

    def tv(self, component_id, weight=1.0, layers=None, modules=None):
        """Add a task vector. Returns self so calls can chain."""
        self.task_vectors.append(TaskVector(component_id, weight, layers, modules))
        return self

    def graft(self, component_id, at=None, control=None, control_seed=0, glue_rank=None):
        """Add a graft. Returns self so calls can chain."""
        self.grafts.append(GraftSpec(component_id, at, control, control_seed, glue_rank))
        return self

    def heal(self, steps=300, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384):
        """Set the heal pass. Returns self so calls can chain."""
        self.heal_spec = HealSpec(steps, data, lr, batch_size, seq_len)
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
            grafts.append(GraftSpec(graft["id"], graft.get("at"), graft.get("control"), graft.get("control_seed", 0),
                                   graft.get("glue_rank")))
        heal = None
        if data.get("heal") is not None:
            heal = HealSpec(**data["heal"])
        host = data.get("host")
        if host is None:
            host = data["base"]
        return cls(host=host, masks=masks, task_vectors=task_vectors, seed=data.get("seed", 0),
                   grafts=grafts, heal=heal)

    def compile(self, output_dir, catalog, **options):
        """Build the model and save it to output_dir. Return output_dir.

        options pass through to wsp.compiler.compile_program
        (align_texts, heal_texts, device, cache_dir).
        """
        from wsp.compiler import compile_program

        return compile_program(self, catalog, output_dir, **options)

    def __repr__(self):
        return (f"Program(host={self.host!r}, masks={self.masks!r}, "
                f"task_vectors={len(self.task_vectors)}, grafts={len(self.grafts)})")


def collect_ids(expr, ids):
    if isinstance(expr, Use):
        ids.append(expr.component_id)
    elif isinstance(expr, Restrict):
        collect_ids(expr.inner, ids)
    else:
        collect_ids(expr.a, ids)
        collect_ids(expr.b, ids)
