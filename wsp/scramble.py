"""Scrambled weights: same shapes and statistics as a real component, no learned structure.

A control for "is the donor's pretrained knowledge used at all". Another
layer range of the same donor still holds learned structure. A scrambled
copy of the real block does not.

Modes:
  spectrum  W becomes U diag(s) V^T. s are the singular values of W. U and V
            are random orthonormal bases (Haar: QR of a seeded Gaussian, with
            the sign fix). Shape, Frobenius norm, and the full spectrum stay.
  permute   A seeded permutation of all entries of the tensor. The exact value
            histogram and the norm stay. The spectrum is destroyed.

1-D tensors (norm weights, biases) are always permuted. Embedding tables are
not touched unless include_embeddings=True. Tensors with more than two
dimensions are permuted.

Economy for very large matrices: only the singular values of W are computed
(torch.linalg.svdvals in float32, on the tensor's device), one tensor at a
time. A matrix with more entries than max_numel uses the fallback mode
instead (default "permute"). Each tensor reports its seconds.

All random draws come from a CPU generator seeded by (seed, tensor name or
position), so a result does not depend on the device.
"""

import time
import zlib

import torch
from torch import nn

MODES = ("spectrum", "permute")
EMBEDDING_HINTS = ("embed_tokens", "embedding", "wte", "wpe", "lm_head")


def tensor_seed(seed, key):
    """A stable 31-bit seed from the run seed and a tensor key (a name or a position)."""
    return (int(seed) * 1000003 + zlib.crc32(str(key).encode("utf-8"))) % (2 ** 31 - 1)


def haar_orthonormal(rows, cols, generator, device):
    """Return a (rows, cols) matrix with orthonormal columns, Haar distributed. Needs rows >= cols."""
    gauss = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(gauss.to(device))
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1.0
    return q * signs.unsqueeze(0)


def permute_tensor(tensor, seed):
    """Return a seeded permutation of all entries of tensor (same shape, dtype, device)."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(tensor.numel(), generator=generator).to(tensor.device)
    return tensor.reshape(-1)[order].reshape(tensor.shape).clone()


def spectrum_tensor(tensor, seed):
    """Return U diag(s) V^T with s the singular values of the 2-D tensor and U, V random orthonormal."""
    rows, cols = tensor.shape
    k = min(rows, cols)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    values = torch.linalg.svdvals(tensor.detach().float())
    u = haar_orthonormal(rows, k, generator, tensor.device)
    v = haar_orthonormal(cols, k, generator, tensor.device)
    out = (u * values.unsqueeze(0)) @ v.T
    return out.to(tensor.dtype)


def is_embedding_name(name):
    lowered = name.lower()
    for hint in EMBEDDING_HINTS:
        if hint in lowered:
            return True
    return False


def scramble_tensor(tensor, seed, mode="spectrum", max_numel=None, fallback="permute"):
    """Return (new_tensor, mode_used). The input is not changed."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if fallback not in MODES:
        raise ValueError(f"fallback must be one of {MODES}, got {fallback!r}")
    if tensor.dim() != 2:
        return permute_tensor(tensor, seed), "permute"
    used = mode
    if mode == "spectrum" and max_numel is not None and tensor.numel() > max_numel:
        used = fallback
    if used == "spectrum":
        return spectrum_tensor(tensor, seed), "spectrum"
    return permute_tensor(tensor, seed), "permute"


def fro(tensor):
    return float(tensor.detach().float().norm())


def cosine(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    denom = float(a.norm() * b.norm())
    if denom == 0.0:
        return 0.0
    return float(a @ b) / denom


def new_report(mode):
    return {"mode": mode, "tensors_changed": 0, "tensors_skipped": 0, "rel_fro_error": 0.0,
            "mean_abs_cosine": 0.0, "cosine_sum": 0.0, "cosine_count": 0, "seconds": 0.0, "modes_used": {}, "slowest": None}


def record(report, name, old, new, mode_used, seconds):
    """Add one tensor to the report. rel_fro_error is the largest |norm(new) - norm(old)| / norm(old)."""
    before = fro(old)
    after = fro(new)
    error = abs(after - before) / before if before > 0 else 0.0
    report["tensors_changed"] += 1
    report["rel_fro_error"] = max(report["rel_fro_error"], error)
    if old.dim() == 2:
        report["cosine_sum"] += abs(cosine(old, new))
        report["cosine_count"] += 1
    report["seconds"] += seconds
    report["modes_used"][mode_used] = report["modes_used"].get(mode_used, 0) + 1
    slowest = report["slowest"]
    if slowest is None or seconds > slowest["seconds"]:
        report["slowest"] = {"name": name, "shape": list(old.shape), "mode": mode_used, "seconds": round(seconds, 3)}


def finish(report):
    if report["cosine_count"] > 0:
        report["mean_abs_cosine"] = report["cosine_sum"] / report["cosine_count"]
    del report["cosine_sum"], report["cosine_count"]
    report["seconds"] = round(report["seconds"], 3)
    return report


def scramble_tensors(tensors, seed, mode="spectrum", max_numel=None, fallback="permute", include_embeddings=False):
    """Scramble a {name: tensor} dict in place. Return the report."""
    report = new_report(mode)
    for name in sorted(tensors):
        old = tensors[name]
        if not torch.is_floating_point(old) or (not include_embeddings and is_embedding_name(name)):
            report["tensors_skipped"] += 1
            continue
        t0 = time.time()
        with torch.no_grad():
            new, used = scramble_tensor(old, tensor_seed(seed, name), mode, max_numel, fallback)
        seconds = time.time() - t0
        record(report, name, old, new, used, seconds)
        tensors[name] = new
    return finish(report)


def scramble_module(module, seed, mode="spectrum", max_numel=None, fallback="permute", include_embeddings=False):
    """Rewrite every weight of module in place so no learned structure remains. Return a report.

    report keys: mode, tensors_changed, tensors_skipped, rel_fro_error (the
    largest relative change of a tensor's Frobenius norm, about 0),
    mean_abs_cosine (mean |cosine| between old and new over the 2-D tensors, about 0), seconds,
    modes_used, slowest.
    """
    report = new_report(mode)
    embedding_ids = set()
    if not include_embeddings:
        for sub in module.modules():
            if isinstance(sub, nn.Embedding):
                for p in sub.parameters(recurse=False):
                    embedding_ids.add(id(p))
    seen = set()
    for name, param in module.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if id(param) in embedding_ids or (not include_embeddings and is_embedding_name(name)):
            report["tensors_skipped"] += 1
            continue
        if not torch.is_floating_point(param):
            report["tensors_skipped"] += 1
            continue
        t0 = time.time()
        with torch.no_grad():
            old = param.detach()
            new, used = scramble_tensor(old, tensor_seed(seed, name), mode, max_numel, fallback)
            seconds = time.time() - t0
            record(report, name, old, new, used, seconds)
            param.copy_(new)
    return finish(report)
