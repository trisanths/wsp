# Test of the engine's forward-hook scans on a model whose layers are on two devices, the way
# device_map="auto" places a large model on a multi-GPU tier. Layer 0 and the inputs are on the
# home device (it stands for cuda:0), every other layer is on a second device. The padding mask
# is made next to the inputs, so each hook that uses it in another layer must move it.
#   .venv/bin/python mine/tests/test_two_devices.py
# The test needs two real devices. It uses mps as the home device and the CPU as the second one,
# and it prints "skipped" on a machine that has no mps. Nothing here talks to Modal.
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import adapter  # noqa: E402
import toy_model  # noqa: E402
import test_skill_maps_engine as base  # noqa: E402

OTHER = torch.device("cpu")


def place(module, device):
    # what accelerate does for a module of a spread model: the input moves to the module's device
    module.to(device)
    inner = module.forward

    def forward(x, *args, **kwargs):
        return inner(x.to(device), *args, **kwargs)
    module.forward = forward


class SpreadEngine(base.CpuEngine):
    def __init__(self, home):
        base.CpuEngine.__init__(self)
        self.home = home
        self.model.model.embed.to(home)
        for i in range(len(self.layers)):
            place(self.layers[i], home if i == 0 else OTHER)
        place(self.model.lm_head, OTHER)

    def _pad(self, rows, batch):
        ids, mask = base.CpuEngine._pad(self, rows, batch)
        return ids.to(self.home), mask.to(self.home)

    def _score_rows(self, rows):
        real = self.model.forward

        def forward(input_ids, attention_mask=None, use_cache=False):
            out = real(input_ids.to(self.home))
            out.logits = out.logits.cpu()
            return out
        self.model.forward = forward
        try:
            return base.CpuEngine._score_rows(self, rows)
        finally:
            self.model.forward = real


def raw(name):
    method = base.engine_app.EngineBase.__dict__[name]
    return method.raw_f if hasattr(method, "raw_f") else method._get_raw_f()


def close(a, b, where="result"):
    # same structure and the same numbers, within the float difference of two devices
    if isinstance(a, dict):
        assert sorted(a.keys()) == sorted(b.keys()), where
        for k in a:
            close(a[k], b[k], where + "." + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i in range(len(a)):
            close(a[i], b[i], where + "[" + str(i) + "]")
    elif isinstance(a, float) or isinstance(b, float):
        assert abs(a - b) < 2e-3, (where, a, b)
    else:
        assert a == b, (where, a, b)


def scan(engine, kind, layers):
    # the per-unit statistic of activation_scan, before the ranking (a ranking can swap two units
    # that differ by a float error)
    out = {}
    for skill in ["law", "general_text"]:
        means, n_tokens = engine._activation_means(base.payload(), kind, skill, layers, 32)
        out[skill] = {str(l): [float(v) for v in means[l]] for l in means}
        out[skill]["n_tokens"] = n_tokens
    return out


def test_two_devices():
    if not torch.backends.mps.is_available():
        print("skipped: this machine has no second device (mps)")
        return
    one = SpreadEngine(torch.device("cpu"))
    two = SpreadEngine(torch.device("mps"))
    assert two.model.model.embed.weight.device.type == "mps"
    assert two.layers[1].mlp.router.weight.device.type == "cpu"
    payload = base.payload()

    # skill_maps, plain call: the whole-layer passes, then the router counts of the fused-expert
    # layers 1 and 3, which are not on the home device
    want = one.skill_maps(payload)
    got = two.skill_maps(payload)
    assert sorted(got["expert_freq"]["law|search"].keys()) == ["1", "3"]
    for name in ["base", "layer_lp", "expert_freq"]:
        close(got[name], want[name], name)
    for name in got["expert_freq"]:
        for layer in ["1", "3"]:
            assert abs(sum(got["expert_freq"][name][layer]) - toy_model.TOP_K) < 1e-4

    # activation_scan for every kind: the call returns, and the statistic it ranks is the same
    scan_fn = raw("activation_scan")
    for kind, layers in [("mlp_cluster", [0, 2, 5]), ("attn_heads", [0, 1, 3]), ("expert", [1, 3])]:
        res = scan_fn(two, payload, kind, ["law", "general_text"], layers)
        assert sorted(res["stats"].keys()) == ["general_text", "law"], kind
        assert res["stats"]["law"], kind
        close(scan(two, kind, layers), scan(one, kind, layers), kind)

    # no hook stays on the model
    for engine in [one, two]:
        for module in engine.model.modules():
            assert not module._forward_pre_hooks and not module._forward_hooks


if __name__ == "__main__":
    test_two_devices()
    print("ok")
