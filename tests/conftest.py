"""Skip the tests that need the Transformers 5 model layouts when 4.x runs.

Transformers 5 holds MoE experts as fused 3-D parameters and adds model
families that 4.57 does not have (Lfm2Moe, Gemma 4, Qwen 3.5). The tests
below cover those layouts and the 5.x part code in wsp/parts.py. On 4.x,
wsp/parts.py serves the older code in wsp/parts_tf4.py, which the rest of the
suite covers.
"""

import pytest
import transformers

TRANSFORMERS_5 = int(transformers.__version__.split(".")[0]) >= 5

NEEDS_TRANSFORMERS_5 = (
    "tests/test_lfm2.py",
    "tests/test_new_families.py",
    "test_block.py::test_block_equals_donor_layers[lfm3",
    "test_block.py::test_block_with_left_padding[lfm3",
    "test_part_cache.py::test_cached_generate_equals_uncached[lfm",
    "test_part_cache.py::test_cached_step_logits_match_full_forward[lfm",
    "test_part_cache.py::test_generate_twice_resets_part_cache[lfm",
    "test_part_cache.py::test_beam_search_cached_equals_uncached[lfm",
    "test_part_cache.py::test_chunked_prefill_equals_uncached[lfm",
    "test_expert.py::test_single_expert_matches_donor[qwen3_moe",
    "test_expert.py::test_single_expert_matches_donor[mixtral",
    "test_expert.py::test_two_experts_sum[qwen3_moe",
    "test_expert.py::test_two_experts_sum[mixtral",
    "test_expert.py::test_fused_checkpoint_experts_are_sliced",
    "test_expert.py::test_lfm2_moe",
    "test_expert.py::test_block_over_moe_layers_reads_per_expert_checkpoint[tiny_lfm2_moe",
    "test_expert.py::test_moe_block_graft_with_part_lora_compiles_heals_and_reloads",
    "test_sink.py::test_part_lora_wraps_fused_experts",
    "test_sink.py::test_part_lora_on_a_fused_expert_graft_heals_and_reloads",
    "test_sink.py::test_part_lora_wraps_lfm2_part_linears",
    "test_extraction_and_claims.py::test_expert_ablation_changes_the_output_and_removes_cleanly",
)


def pytest_collection_modifyitems(config, items):
    if TRANSFORMERS_5:
        return
    skip = pytest.mark.skip(reason="needs the Transformers 5 model layouts")
    for item in items:
        if any(mark in item.nodeid for mark in NEEDS_TRANSFORMERS_5):
            item.add_marker(skip)
