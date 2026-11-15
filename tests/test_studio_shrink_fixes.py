from tests.test_studio_shrink import client, make_spec  # noqa: F401


def test_moe_shrink_block_has_the_real_expert_count(client):
    spec = make_spec(client, "keep only the math experts of a 30B MoE")
    sh = spec["shrink"]
    assert sh["method"] == "experts"
    assert isinstance(sh["experts_total"], int) and sh["experts_total"] > 1
    assert sh["keep_experts"] <= sh["experts_total"]


def test_dense_shrink_block_has_no_expert_count(client):
    spec = make_spec(client, "a coding-only model: drop everything else from Qwen2.5-Coder-7B and get it under 3B")
    assert spec["shrink"]["experts_total"] is None
