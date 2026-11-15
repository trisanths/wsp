"""Docking ports toy: can Net2Net copies, trained toward a donor, carry the donor's skill into a host?

Two MLPs, 784-512-512-20. A is trained on MNIST (labels 0..9) and B on Fashion-MNIST
(labels 10..19), so both share one 20-way output space and a merged network can answer
both tasks with no task flag. Only the stitch arm uses a flag (a router by task, which is
part of that network). Pairs: "diff" (A and B start from different random inits, the hard
case) and "same" (both start from one init, the easy case).

Every arm produces ONE network, scored on MNIST test and Fashion-MNIST test. Training
data is the mix of both TRAIN splits. The test split is never used to train, fit, or
choose anything. Every trained arm uses STEPS steps of batch BATCH.

Run:
  modal run --detach modal_docking.py --smoke
  modal run --detach modal_docking.py
Rows go to results/docking_rows.json (results/docking_smoke_rows.json for --smoke).
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import modal
except ImportError:  # CPU tests may run without modal
    modal = None

HID = 512
OUT = 20
PRETRAIN_STEPS = 1500
STEPS = 500
BATCH = 256
LR = 1e-3
SEEDS = [0, 1, 2]
PAIRS = ["diff", "same"]
KS = [2, 4]
SIGMAS = [0.01, 0.1]
HIDDEN_WEIGHT = 1.0


# ---------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, d_in=784, h1=HID, h2=HID, d_out=OUT):
        super().__init__()
        self.l1 = nn.Linear(d_in, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, d_out)

    def hidden(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return h1, h2, self.l3(h2)

    def forward(self, x):
        return self.hidden(x)[2]


def clone(model):
    other = MLP(model.l1.in_features, model.l1.out_features, model.l2.out_features, model.l3.out_features)
    other.load_state_dict(model.state_dict())
    return other.to(model.l1.weight.dtype)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


class ConcatNet(nn.Module):
    """A and B side by side: block-diagonal hidden layers, outputs summed with learned scalars."""

    def __init__(self, a, b):
        super().__init__()
        h1a, h1b = a.l1.out_features, b.l1.out_features
        h2a, h2b = a.l2.out_features, b.l2.out_features
        self.h2a = h2a
        self.l1 = nn.Linear(a.l1.in_features, h1a + h1b)
        self.l2 = nn.Linear(h1a + h1b, h2a + h2b)
        self.l3 = nn.Linear(h2a + h2b, a.l3.out_features)
        self.scale = nn.Parameter(torch.ones(2))
        with torch.no_grad():
            self.l1.weight.copy_(torch.cat([a.l1.weight, b.l1.weight]))
            self.l1.bias.copy_(torch.cat([a.l1.bias, b.l1.bias]))
            self.l2.weight.zero_()
            self.l2.weight[:h2a, :h1a] = a.l2.weight
            self.l2.weight[h2a:, h1a:] = b.l2.weight
            self.l2.bias.copy_(torch.cat([a.l2.bias, b.l2.bias]))
            self.l3.weight.copy_(torch.cat([a.l3.weight, b.l3.weight], dim=1))
            self.l3.bias.copy_(a.l3.bias + b.l3.bias)

    def forward(self, x):
        h = F.relu(self.l2(F.relu(self.l1(x))))
        h = torch.cat([self.scale[0] * h[:, :self.h2a], self.scale[1] * h[:, self.h2a:]], dim=1)
        return self.l3(h)


# ---------------------------------------------------------------- Net2Wider

def net2wider(model, k, sigma=0.0, generator=None):
    """Widen both hidden layers by factor k. Unit j of the new layer copies unit j % n.

    Incoming weights are copied, outgoing weights are divided by k, so the function is
    exactly the same at sigma = 0. Noise of sigma * std(weight matrix) goes on every entry
    that touches a copy. Returns (wide model, masks); masks mark the entries a docking
    run may train (everything except the original-to-original block).
    """
    n1, n2 = model.l1.out_features, model.l2.out_features
    g1 = torch.arange(n1 * k) % n1
    g2 = torch.arange(n2 * k) % n2
    wide = MLP(model.l1.in_features, n1 * k, n2 * k, model.l3.out_features).to(model.l1.weight.dtype)
    dtype = model.l1.weight.dtype
    c1 = torch.arange(n1 * k) >= n1
    c2 = torch.arange(n2 * k) >= n2

    def noise(shape, std):
        z = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
        return z * (sigma * std)

    with torch.no_grad():
        w1 = model.l1.weight[g1].clone()
        w1[c1] += noise(w1[c1].shape, model.l1.weight.std().item())
        wide.l1.weight.copy_(w1)
        wide.l1.bias.copy_(model.l1.bias[g1])
        w2 = model.l2.weight[g2][:, g1] / k
        extra = noise(w2.shape, model.l2.weight.std().item() / k)
        touch = c2[:, None] | c1[None, :]
        w2 = w2 + extra * touch
        wide.l2.weight.copy_(w2)
        wide.l2.bias.copy_(model.l2.bias[g2])
        w3 = model.l3.weight[:, g2] / k
        e3 = noise(w3.shape, model.l3.weight.std().item() / k)
        w3 = w3 + e3 * c2[None, :]
        wide.l3.weight.copy_(w3)
        wide.l3.bias.copy_(model.l3.bias)
    masks = {
        "l1.weight": c1[:, None].expand_as(wide.l1.weight).clone(),
        "l1.bias": c1.clone(),
        "l2.weight": (c2[:, None] | c1[None, :]).clone(),
        "l2.bias": c2.clone(),
        "l3.weight": c2[None, :].expand_as(wide.l3.weight).clone(),
        "l3.bias": torch.zeros_like(wide.l3.bias, dtype=torch.bool),
    }
    return wide, {"masks": masks, "g1": g1, "g2": g2, "c1": c1, "c2": c2}


# ---------------------------------------------------------------- Re-Basin (weight matching)

def hungarian(cost):
    """Minimum cost assignment of a square matrix. Returns col index for each row."""
    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = np.zeros(n, dtype=np.int64)
    for j in range(1, n + 1):
        result[p[j] - 1] = j - 1
    return result


def permute_model(model, p1, p2):
    """Reorder hidden units of a model: new unit i is old unit p[i]. Function is unchanged."""
    out = clone(model)
    with torch.no_grad():
        out.l1.weight.copy_(model.l1.weight[p1])
        out.l1.bias.copy_(model.l1.bias[p1])
        out.l2.weight.copy_(model.l2.weight[p2][:, p1])
        out.l2.bias.copy_(model.l2.bias[p2])
        out.l3.weight.copy_(model.l3.weight[:, p2])
    return out


@torch.no_grad()
def rebasin_match(a, b, sweeps=20):
    """Weight matching (Git Re-Basin): permutations of b's hidden units that best match a."""
    n1, n2 = b.l1.out_features, b.l2.out_features
    p1 = torch.arange(n1)
    p2 = torch.arange(n2)
    wa1, wb1 = a.l1.weight.double(), b.l1.weight.double()
    wa2, wb2 = a.l2.weight.double(), b.l2.weight.double()
    wa3, wb3 = a.l3.weight.double(), b.l3.weight.double()
    ba1, bb1 = a.l1.bias.double(), b.l1.bias.double()
    ba2, bb2 = a.l2.bias.double(), b.l2.bias.double()
    for _ in range(sweeps):
        old = (p1.clone(), p2.clone())
        score = wa1 @ wb1.T + wa2.T @ wb2[p2].T + ba1[:, None] * bb1[None, :]
        p1 = torch.from_numpy(hungarian(-score.numpy())).long()
        # match[i] = j means a's unit i takes b's unit j
        score = wa2[:, :] @ wb2[:, p1].T + wa3.T @ wb3 + ba2[:, None] * bb2[None, :]
        p2 = torch.from_numpy(hungarian(-score.numpy())).long()
        if torch.equal(old[0], p1) and torch.equal(old[1], p2):
            break
    return p1, p2


def rebasin(a, b):
    p1, p2 = rebasin_match(a, b)
    return permute_model(b, p1, p2), (p1, p2)


def average(a, b):
    out = clone(a)
    with torch.no_grad():
        for pa, pb, po in zip(a.parameters(), b.parameters(), out.parameters()):
            po.copy_((pa + pb) / 2)
    return out


# ---------------------------------------------------------------- stitch

def stitch_forward(a, b, stitch, x, task):
    """Skill top in miniature. MNIST: A all the way. Fashion: A's bottom, stitch, B's top and head.

    task is a flag per row (0 MNIST, 1 Fashion): the router, which is part of the network.
    """
    h1 = F.relu(a.l1(x))
    out_a = a.l3(F.relu(a.l2(h1)))
    out_b = b.l3(F.relu(b.l2(stitch(h1))))
    return torch.where(task[:, None] == 1, out_b, out_a)


# ---------------------------------------------------------------- data

def load_arrays():
    from datasets import load_dataset

    def split(name):
        ds = load_dataset(name)
        out = []
        for part in ["train", "test"]:
            rows = ds[part]
            x = np.stack([np.asarray(im, dtype=np.uint8).reshape(-1) for im in rows["image"]])
            y = np.asarray(rows["label"], dtype=np.int64)
            out.append((torch.from_numpy(x).float() / 255.0, torch.from_numpy(y)))
        return out

    mn = split("ylecun/mnist")
    fa = split("zalando-datasets/fashion_mnist")
    return {"mnist": {"train": mn[0], "test": mn[1]}, "fashion": {"train": fa[0], "test": fa[1]}}


class Mix:
    """Both tasks' train splits as one pool. Labels: MNIST 0..9, Fashion 10..19. src: 0 or 1."""

    def __init__(self, data, generator):
        xm, ym = data["mnist"]["train"]
        xf, yf = data["fashion"]["train"]
        self.x = torch.cat([xm, xf])
        self.y = torch.cat([ym, yf + 10])
        self.src = torch.cat([torch.zeros(len(xm), dtype=torch.long), torch.ones(len(xf), dtype=torch.long)])
        self.g = generator

    def batch(self, size=BATCH, only=None):
        if only is None:
            idx = torch.randint(0, len(self.x), (size,), generator=self.g)
        else:
            pool = torch.nonzero(self.src == only).squeeze(1)
            idx = pool[torch.randint(0, len(pool), (size,), generator=self.g)]
        return self.x[idx], self.y[idx], self.src[idx]


@torch.no_grad()
def accuracy(predict, x, y_offset, y):
    right = 0
    for i in range(0, len(x), 2000):
        out = predict(x[i:i + 2000], i)
        right += (out.argmax(1) == (y[i:i + 2000] + y_offset)).sum().item()
    return right / len(x)


def score(predict, data):
    """predict(x, task_flag) -> logits. Test splits only."""
    xm, ym = data["mnist"]["test"]
    xf, yf = data["fashion"]["test"]
    m = accuracy(lambda x, i: predict(x, 0), xm, 0, ym)
    f = accuracy(lambda x, i: predict(x, 1), xf, 10, yf)
    return {"mnist": m, "fashion": f, "mean": (m + f) / 2}


def plain_predict(model):
    return lambda x, task: model(x)


# ---------------------------------------------------------------- training

def run_training(params, steps, loss_fn, masks=None, lr=LR):
    """Adam on params. masks: name -> bool tensor, applied to gradients (False entries stay frozen)."""
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        loss = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        if masks is not None:
            for p, m in masks:
                if p.grad is not None:
                    p.grad.mul_(m)
        opt.step()
    return float(loss.detach())


def train_ce(model, mix, steps, only=None, params=None, offset=0):
    def loss_fn(_):
        x, y, _s = mix.batch(only=only)
        return F.cross_entropy(model(x), y)
    run_training(params if params is not None else list(model.parameters()), steps, loss_fn)
    return model


def train_task(model, data, task, steps, generator):
    """Pretrain one model on one task alone (labels offset by 10 for Fashion)."""
    x, y = data[task]["train"]
    off = 10 if task == "fashion" else 0
    def loss_fn(_):
        idx = torch.randint(0, len(x), (BATCH,), generator=generator)
        return F.cross_entropy(model(x[idx]), y[idx] + off)
    run_training(list(model.parameters()), steps, loss_fn)
    return model


def kd_loss(student_logits, teacher_logits):
    return F.kl_div(F.log_softmax(student_logits, 1), F.softmax(teacher_logits, 1), reduction="batchmean")


def scramble(b, data, mix, generator, steps=STEPS):
    """B with entries of each hidden matrix shuffled at random, then the head re-fit on Fashion train."""
    out = clone(b)
    with torch.no_grad():
        for layer in [out.l1, out.l2]:
            flat = layer.weight.flatten()
            layer.weight.copy_(flat[torch.randperm(flat.numel(), generator=generator)].view_as(layer.weight))
            layer.bias.copy_(layer.bias[torch.randperm(layer.bias.numel(), generator=generator)])
    head = list(out.l3.parameters())
    def loss_fn(_):
        x, y, _s = mix.batch(only=1)
        return F.cross_entropy(out(x), y)
    run_training(head, steps, loss_fn)
    return out


# ---------------------------------------------------------------- docking

def dock(a, b, mix, k, sigma, mode, steps, generator, rebasin_perm=None):
    """Widen a by k with noise, freeze the original block, train the copies toward b.

    mode: "lin"   hidden targets are b's activations through a learned linear map from the copies
          "reb"   b is first permuted to a (Re-Basin); copy of unit i is pulled to b's unit i directly
          "scram" like "lin" but b is a scrambled b (control, no Fashion features)
          "plain" output distillation only, no hidden target (control)
    The output target on a mix row is a's logits for a MNIST row and b's logits for a Fashion row.
    Returns (wide model, info).
    """
    wide, meta = net2wider(a, k, sigma, generator)
    masks_by_name = dict(wide.named_parameters())
    mask_list = [(masks_by_name[name], m.to(wide.l1.weight.dtype)) for name, m in meta["masks"].items()]
    params = list(wide.parameters())
    n1, n2 = a.l1.out_features, a.l2.out_features
    c1, c2, g1, g2 = meta["c1"], meta["c2"], meta["g1"], meta["g2"]
    extra = []
    maps = None
    if mode == "reb":
        b_t = b if rebasin_perm is None else permute_model(b, *rebasin_perm)
    else:
        b_t = b
    if mode in ("lin", "scram"):
        m1 = nn.Linear(int(c1.sum()), n1)
        m2 = nn.Linear(int(c2.sum()), n2)
        maps = (m1, m2)
        extra = list(m1.parameters()) + list(m2.parameters())
    before = {n: p.detach().clone() for n, p in wide.named_parameters()}

    def loss_fn(_):
        x, _y, src = mix.batch()
        with torch.no_grad():
            ta = a(x)
            h1b, h2b, tb = b_t.hidden(x)
            teacher = torch.where(src[:, None] == 1, tb, ta)
        h1, h2, logits = wide.hidden(x)
        loss = kd_loss(logits, teacher)
        if mode in ("lin", "scram"):
            t1 = maps[0](h1[:, c1])
            t2 = maps[1](h2[:, c2])
            loss = loss + HIDDEN_WEIGHT * (F.mse_loss(t1, h1b) / (h1b.pow(2).mean() + 1e-6)
                                           + F.mse_loss(t2, h2b) / (h2b.pow(2).mean() + 1e-6))
        elif mode == "reb":
            t1 = h1b[:, g1[c1]]
            t2 = h2b[:, g2[c2]]
            loss = loss + HIDDEN_WEIGHT * (F.mse_loss(h1[:, c1], t1) / (h1b.pow(2).mean() + 1e-6)
                                           + F.mse_loss(h2[:, c2], t2) / (h2b.pow(2).mean() + 1e-6))
        return loss

    run_training(params + extra, steps, loss_fn, masks=mask_list + [(p, torch.ones_like(p)) for p in extra])
    frozen_ok = all(
        torch.equal(before[n][~meta["masks"][n]], dict(wide.named_parameters())[n].detach()[~meta["masks"][n]])
        for n in before)
    trainable = int(sum(m.sum() for m in meta["masks"].values()))
    return wide, {"frozen_intact": frozen_ok, "trainable": trainable}


# ---------------------------------------------------------------- arms

def arm_names():
    names = ["a_alone", "b_alone", "joint", "ft_full", "avg", "avg_heal", "rebasin", "rebasin_heal",
             "concat", "stitch"]
    for prefix in ["dock_lin", "dock_reb", "scram", "plain"]:
        for k in KS:
            for s in SIGMAS:
                names.append(f"{prefix}_k{k}_s{s}")
    return names


def parse_dock(name):
    prefix, k, s = name.rsplit("_", 2)
    return {"dock_lin": "lin", "dock_reb": "reb", "scram": "scram", "plain": "plain"}[prefix], int(k[1:]), float(s[1:])


def make_row(pair, arm, seed, sc, params, trainable, steps, router, note=""):
    return {"pair": pair, "arm": arm, "seed": seed, "mnist": sc["mnist"], "fashion": sc["fashion"],
            "mean": sc["mean"], "params": params, "trainable": trainable, "steps": steps,
            "needs_router": router, "note": note}


def run_unit(pair, seed, arms=None, data=None, log=print):
    """All arms for one (pair, seed). Returns rows."""
    data = data or load_arrays()
    wanted = arms or arm_names()
    torch.manual_seed(seed * 7 + (0 if pair == "diff" else 1))
    gen = torch.Generator().manual_seed(seed)
    a = MLP()
    if pair == "same":
        b = clone(a)
    else:
        torch.manual_seed(1000 + seed)
        b = MLP()
    train_task(a, data, "mnist", PRETRAIN_STEPS, torch.Generator().manual_seed(seed + 11))
    train_task(b, data, "fashion", PRETRAIN_STEPS, torch.Generator().manual_seed(seed + 12))
    a.eval(), b.eval()
    rows = []

    def mix_new(tag):
        return Mix(data, torch.Generator().manual_seed(seed * 100 + tag))

    def emit(arm, sc, params, trainable, steps, router=False, note=""):
        row = make_row(pair, arm, seed, sc, params, trainable, steps, router, note)
        rows.append(row)
        log(f"[{pair} {arm} seed{seed}] mnist={sc['mnist']:.4f} fashion={sc['fashion']:.4f} "
            f"params={params} trainable={trainable} steps={steps} {note}")

    def want(name):
        return name in wanted

    if want("a_alone"):
        emit("a_alone", score(plain_predict(a), data), n_params(a), 0, PRETRAIN_STEPS)
    if want("b_alone"):
        emit("b_alone", score(plain_predict(b), data), n_params(b), 0, PRETRAIN_STEPS)
    if want("joint"):
        torch.manual_seed(5000 + seed)
        m = train_ce(MLP(), mix_new(1), STEPS)
        emit("joint", score(plain_predict(m), data), n_params(m), n_params(m), STEPS)
    if want("ft_full"):
        m = train_ce(clone(a), mix_new(2), STEPS)
        emit("ft_full", score(plain_predict(m), data), n_params(m), n_params(m), STEPS)
    avg = average(a, b)
    if want("avg"):
        emit("avg", score(plain_predict(avg), data), n_params(avg), 0, 0)
    if want("avg_heal"):
        m = train_ce(clone(avg), mix_new(3), STEPS)
        emit("avg_heal", score(plain_predict(m), data), n_params(m), n_params(m), STEPS)
    need_rb = any(want(n) for n in ["rebasin", "rebasin_heal"]) or any(n.startswith("dock_reb") for n in wanted)
    perm = None
    if need_rb:
        b_perm, perm = rebasin(a, b)
        merged = average(a, b_perm)
        if want("rebasin"):
            emit("rebasin", score(plain_predict(merged), data), n_params(merged), 0, 0)
        if want("rebasin_heal"):
            m = train_ce(clone(merged), mix_new(4), STEPS)
            emit("rebasin_heal", score(plain_predict(m), data), n_params(m), n_params(m), STEPS)
    if want("concat"):
        net = ConcatNet(a, b)
        start = score(plain_predict(net), data)
        train_ce(net, mix_new(5), STEPS)
        emit("concat", score(plain_predict(net), data), n_params(net), n_params(net), STEPS,
             note=f"before_heal_mean={start['mean']:.4f}")
    if want("stitch"):
        emit_stitch(a, b, data, mix_new(6), seed, emit)
    scrambled = None
    for name in wanted:
        if not name.startswith(("dock_", "scram_", "plain_")):
            continue
        mode, k, sigma = parse_dock(name)
        mix = mix_new(7)
        g = torch.Generator().manual_seed(seed * 31 + k)
        if mode == "scram":
            if scrambled is None:
                scrambled = scramble(b, data, mix_new(8), torch.Generator().manual_seed(seed + 77))
            wide, info = dock(a, scrambled, mix, k, sigma, "scram", STEPS, g)
        else:
            wide, info = dock(a, b, mix, k, sigma, mode, STEPS, g, rebasin_perm=perm if mode == "reb" else None)
        emit(name, score(plain_predict(wide), data), n_params(wide), info["trainable"], STEPS,
             note=f"frozen_intact={info['frozen_intact']}")
    return rows


def emit_stitch(a, b, data, mix, seed, emit):
    stitch = nn.Linear(HID, HID)
    with torch.no_grad():
        x, _y, src = mix.batch(size=5000, only=1)
        h1a = F.relu(a.l1(x))
        h1b = F.relu(b.l1(x))
        design = torch.cat([h1a, torch.ones(len(h1a), 1)], 1)
        ridge = 1e-2 * torch.eye(design.shape[1])
        sol = torch.linalg.solve(design.T @ design + ridge, design.T @ h1b)
        stitch.weight.copy_(sol[:-1].T)
        stitch.bias.copy_(sol[-1])

    def loss_fn(_):
        x, _y, _s = mix.batch(only=1)
        with torch.no_grad():
            target = b(x)
        out = b.l3(F.relu(b.l2(stitch(F.relu(a.l1(x))))))
        return kd_loss(out, target)
    run_training(list(stitch.parameters()), STEPS, loss_fn)

    def predict(x, task):
        flag = torch.full((len(x),), task, dtype=torch.long)
        return stitch_forward(a, b, stitch, x, flag)
    params = n_params(a) + n_params(stitch) + n_params(b.l2) + n_params(b.l3)
    emit("stitch", score(predict, data), params, n_params(stitch), STEPS, router=True,
         note="router=task flag")


# ---------------------------------------------------------------- modal

if modal is not None:
    app = modal.App("wsp-docking")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.7.1", "torchvision", "numpy", "datasets")
        .env({"HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_XET": "1", "OMP_NUM_THREADS": "4"})
    )

    @app.function(image=image, cpu=4, memory=16384, timeout=3 * 3600)
    def unit(pair: str, seed: int, arms: str = ""):
        torch.set_num_threads(4)
        wanted = arms.split(",") if arms else None
        return run_unit(pair, seed, wanted, log=lambda s: print(s, flush=True))

    @app.local_entrypoint()
    def main(smoke: bool = False):
        if smoke:
            args = [("diff", 0, "ft_full,dock_lin_k2_s0.1")]
        else:
            args = [(p, s, "") for p in PAIRS for s in SEEDS]
        rows = []
        for out in unit.starmap(args, return_exceptions=True):
            if isinstance(out, Exception):
                print("unit failed:", repr(out)[:600])
                continue
            rows.extend(out)
            for r in out:
                print(f"[{r['pair']} {r['arm']} seed{r['seed']}] mnist={r['mnist']:.4f} fashion={r['fashion']:.4f}")
        os.makedirs("results", exist_ok=True)
        path = "results/docking_smoke_rows.json" if smoke else "results/docking_rows.json"
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
        print("wrote", path, len(rows), "rows")
        if not smoke:
            print_table(rows)


def summarize(rows):
    table = {}
    for r in rows:
        table.setdefault((r["pair"], r["arm"]), []).append(r)
    out = []
    for (pair, arm), rs in sorted(table.items()):
        def ms(key):
            v = np.array([r[key] for r in rs])
            return float(v.mean()), float(v.std())
        out.append({"pair": pair, "arm": arm, "n": len(rs), "mnist": ms("mnist"), "fashion": ms("fashion"),
                    "mean": ms("mean"), "params": rs[0]["params"], "trainable": rs[0]["trainable"],
                    "steps": rs[0]["steps"], "router": rs[0]["needs_router"]})
    return out


def print_table(rows):
    for s in summarize(rows):
        print(f"[{s['pair']} {s['arm']}] n={s['n']} mnist={s['mnist'][0]:.4f}+-{s['mnist'][1]:.4f} "
              f"fashion={s['fashion'][0]:.4f}+-{s['fashion'][1]:.4f} params={s['params']} steps={s['steps']}")
