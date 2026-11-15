"""Residual-stream rotation for Qwen2-family models.

Two models that were trained apart may keep the same function in residual
coordinates that differ by an orthogonal map. This file holds the tools to
test that and to apply the map without changing the function.

RMSNorm with a per-dimension weight does not commute with a rotation, so
fold_norms first moves every norm weight into the linear layers that read the
normed state. The norm then has weight 1 and commutes with any orthogonal map.
rotate_residual then rotates every writer of the residual stream by R and
every reader by R transposed.

Convention: a hidden state is a row vector x. A rotated state is x R.
Embedding rows E become E R. A reader weight W (out, in) becomes W R. A writer
weight W (out, in) becomes R^T W.
"""

import torch


def _layers(model):
    return model.model.layers


def untie_head(model):
    """Give lm_head its own weight when it shares the embedding. Output is unchanged."""
    head = model.lm_head
    embed = model.model.embed_tokens
    if head.weight is embed.weight or head.weight.data_ptr() == embed.weight.data_ptr():
        head.weight = torch.nn.Parameter(embed.weight.detach().clone())
    model.config.tie_word_embeddings = False


def _scale_columns(linear, weight):
    """Replace W by W diag(weight): the layer now reads an unweighted normed state."""
    w = linear.weight.data
    scaled = w.double() * weight.double()[None, :]
    linear.weight.data = scaled.to(w.dtype)


@torch.no_grad()
def fold_norms(model):
    """Fold each RMSNorm weight into the linear layers that read the normed state.

    Input norm goes into q, k, v. Post-attention norm goes into gate and up. The
    final norm goes into lm_head (untied first). Every norm weight becomes 1.
    Biases stay as they are, because they act after the matrix. The outputs of
    the model do not change.
    """
    untie_head(model)
    for layer in _layers(model):
        w = layer.input_layernorm.weight.data
        for name in ["q_proj", "k_proj", "v_proj"]:
            _scale_columns(getattr(layer.self_attn, name), w)
        layer.input_layernorm.weight.data = torch.ones_like(w)
        w = layer.post_attention_layernorm.weight.data
        for name in ["gate_proj", "up_proj"]:
            _scale_columns(getattr(layer.mlp, name), w)
        layer.post_attention_layernorm.weight.data = torch.ones_like(w)
    w = model.model.norm.weight.data
    _scale_columns(model.lm_head, w)
    model.model.norm.weight.data = torch.ones_like(w)
    return model


def procrustes(A, B):
    """Return the orthogonal R that minimizes ||A R - B||_F. Computed in float64.

    With A^T B = U S V^T, R = U V^T.
    """
    A = A.double()
    B = B.double()
    U, _, Vh = torch.linalg.svd(A.T @ B, full_matrices=False)
    return U @ Vh


def random_orthogonal(n, seed=0):
    """A Haar-random orthogonal matrix in float64."""
    gen = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))[None, :]


def shuffled_procrustes(A, B, seed=0):
    """Fit R after a random permutation of the token rows of A. Return (R, permutation).

    This is the control: the fit has all the freedom of a d x d rotation but no
    token correspondence.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(A.shape[0], generator=gen)
    return procrustes(A[perm], B), perm


def _relative(A, B):
    return float(torch.linalg.norm(A.double() - B.double()) / torch.linalg.norm(B.double()))


def residual_report(E_m, E_i, R, seed=0):
    """Relative residuals of a rotation fit.

    rotated: ||E_m R - E_i|| / ||E_i|| with the given R.
    unrotated: the same with R = identity.
    shuffled: the token rows of E_m are permuted, a new Procrustes R is fit
    against E_i, and the residual is taken with that R. A real match must beat
    this, or the fit only reflects the freedom of a d x d rotation.
    random: a random orthogonal R.
    """
    E_m = E_m.double()
    E_i = E_i.double()
    R_shuffled, perm = shuffled_procrustes(E_m, E_i, seed)
    shuffled = E_m[perm]
    return {
        "rotated": _relative(E_m @ R.double(), E_i),
        "unrotated": _relative(E_m, E_i),
        "shuffled": _relative(shuffled @ R_shuffled, E_i),
        "random": _relative(E_m @ random_orthogonal(E_m.shape[1], seed), E_i),
    }


def _check_folded(model):
    for layer in _layers(model):
        for norm in [layer.input_layernorm, layer.post_attention_layernorm]:
            if not torch.all(norm.weight == 1):
                raise ValueError("norms are not folded: call fold_norms first")
    if not torch.all(model.model.norm.weight == 1):
        raise ValueError("final norm is not folded: call fold_norms first")
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise ValueError("lm_head is tied to the embedding: call fold_norms first")


def _right(linear, R):
    w = linear.weight.data
    linear.weight.data = (w.double() @ R).to(w.dtype)


def _left(linear, R):
    w = linear.weight.data
    linear.weight.data = (R.T @ w.double()).to(w.dtype)
    if linear.bias is not None:
        b = linear.bias.data
        linear.bias.data = (R.T @ b.double()).to(b.dtype)


@torch.no_grad()
def rotate_residual(model, R):
    """Rotate the residual stream of a folded model by R. The function is unchanged.

    Writers: embedding rows, o_proj, down_proj. Readers: q, k, v, gate, up, and
    the head. R must be orthogonal (d x d).
    """
    _check_folded(model)
    R = R.double()
    d = model.config.hidden_size
    if R.shape != (d, d):
        raise ValueError(f"R must be {d}x{d}, got {tuple(R.shape)}")
    emb = model.model.embed_tokens.weight.data
    model.model.embed_tokens.weight.data = (emb.double() @ R).to(emb.dtype)
    for layer in _layers(model):
        for name in ["q_proj", "k_proj", "v_proj"]:
            _right(getattr(layer.self_attn, name), R)
        _left(layer.self_attn.o_proj, R)
        for name in ["gate_proj", "up_proj"]:
            _right(getattr(layer.mlp, name), R)
        _left(layer.mlp.down_proj, R)
    _right(model.lm_head, R)
    return model
