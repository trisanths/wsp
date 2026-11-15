# A tiny decoder for the CPU tests of mine/adapter.py and the skill maps. It has every layer shape
# the adapter knows: attention with a dense MLP, fused experts only, a layer with no attention, a
# shared expert next to fused experts, and an older MoE block with a ModuleList of experts.
# Each layer can leave out its attention or its feed-forward module (layer.skip), which gives the
# tests a reference that does not use hooks.
import types

import torch

D = 8
HEADS = 2
WIDTH = 12
N_EXPERTS = 4
TOP_K = 2
VOCAB = 32
# layer -> what full_module_parts must return for it (part kinds for "attn" and for "mlp")
EXPECTED_KINDS = [
    {"attn": ["attn_heads"], "mlp": ["mlp_cluster"]},
    {"attn": ["attn_heads"], "mlp": ["expert"]},
    {"attn": [], "mlp": ["mlp_cluster"]},
    {"attn": ["attn_heads"], "mlp": ["mlp_cluster", "expert"]},
    {"attn": ["attn_heads"], "mlp": ["expert"]},
    {"attn": ["attn_heads"], "mlp": ["mlp_cluster"]},
]


class Attention(torch.nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.v_proj = torch.nn.Linear(D, D)
        self.o_proj = torch.nn.Linear(D, D, bias=bias)

    def forward(self, x):
        # causal running mean of the values: position t sees positions 0..t
        v = torch.tanh(self.v_proj(x))
        steps = torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1, 1)
        return self.o_proj(v.cumsum(dim=1) / steps)


class DenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(D, WIDTH)
        self.down_proj = torch.nn.Linear(WIDTH, D, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class FusedExperts(torch.nn.Module):
    # transformers-5 style: 3D parameters, called as experts(hidden_states, top_k_index, top_k_weights)
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(N_EXPERTS, D, WIDTH) * 0.5)
        self.down_proj = torch.nn.Parameter(torch.randn(N_EXPERTS, WIDTH, D) * 0.5)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for k in range(top_k_index.shape[-1]):
            e = top_k_index[:, k]
            h = torch.tanh(torch.einsum("nd,ndw->nw", hidden_states, self.gate_up_proj[e]))
            out = out + top_k_weights[:, k:k + 1] * torch.einsum("nw,nwd->nd", h, self.down_proj[e])
        return out


class FusedMoE(torch.nn.Module):
    def __init__(self, shared, keywords):
        super().__init__()
        self.router = torch.nn.Linear(D, N_EXPERTS)
        self.experts = FusedExperts()
        if shared:
            self.shared_expert = DenseMLP()
        self.keywords = keywords  # call the experts with keyword arguments (the adapter handles both)

    def forward(self, x):
        flat = x.reshape(-1, D)
        weights, index = torch.softmax(self.router(flat), dim=-1).topk(TOP_K, dim=-1)
        if self.keywords:
            out = self.experts(hidden_states=flat, top_k_index=index, top_k_weights=weights)
        else:
            out = self.experts(flat, index, weights)
        if hasattr(self, "shared_expert"):
            out = out + self.shared_expert(flat)
        return out.reshape(x.shape)


class ListMoE(torch.nn.Module):
    # older style: a ModuleList of per-expert MLPs, each output scaled by its routing weight
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(D, N_EXPERTS)
        self.experts = torch.nn.ModuleList([DenseMLP() for _ in range(N_EXPERTS)])

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=-1)
        out = torch.zeros_like(x)
        for i in range(N_EXPERTS):
            out = out + weights[..., i:i + 1] * self.experts[i](x)
        return out


class Layer(torch.nn.Module):
    def __init__(self, attention, ff_name, ff):
        super().__init__()
        if attention is not None:
            self.self_attn = attention
        else:
            self.conv = torch.nn.Linear(D, D)  # a layer with no attention, like a conv block
        setattr(self, ff_name, ff)
        self.ff_name = ff_name
        # "attn" or "mlp": leave that module out; "all": skip the layer (the reference without hooks)
        self.skip = None

    def attn_module(self):
        return getattr(self, "self_attn", None)

    def ff_module(self):
        return getattr(self, self.ff_name)

    def forward(self, hidden_states):
        x = hidden_states
        if self.skip == "all":
            return x
        if self.attn_module() is None:
            x = x + torch.tanh(self.conv(x))
        elif self.skip != "attn":
            x = x + self.self_attn(x)
        if self.skip != "mlp":
            x = x + self.ff_module()(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = Decoder(layers)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)

    def get_decoder(self):
        return self.model

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return types.SimpleNamespace(logits=self.lm_head(self.model(input_ids)))


class Config:
    hidden_size = D
    num_attention_heads = HEADS


def build(seed=0):
    torch.manual_seed(seed)
    layers = [
        Layer(Attention(), "mlp", DenseMLP()),
        Layer(Attention(), "mlp", FusedMoE(shared=False, keywords=False)),
        Layer(None, "feed_forward", DenseMLP()),
        Layer(Attention(), "mlp", FusedMoE(shared=True, keywords=True)),
        Layer(Attention(), "moe", ListMoE()),
        Layer(Attention(), "mlp", DenseMLP()),
    ]
    model = Model(layers)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, Config()
