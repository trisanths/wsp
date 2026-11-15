import torch

from tests.test_align import tiny_model, tiny_tokenizer
from wsp.extract import attribution_neurons, contrast_neurons, mean_abs_activation


def test_contrast_prefers_task_specific_neurons():
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    mlp = model.model.layers[0].mlp
    task = ["the cat sat on a mat"] * 4
    generic = ["dog ran to big red hat"] * 4
    picked = contrast_neurons(model, tokenizer, task, generic, mlp, 5)
    task_act = mean_abs_activation(model, tokenizer, task, mlp)
    generic_act = mean_abs_activation(model, tokenizer, generic, mlp)
    expected = torch.sort(torch.topk(task_act - generic_act, 5).indices).values
    assert torch.equal(picked, expected)


def test_attribution_returns_sorted_distinct_ids():
    model = tiny_model()
    tokenizer = tiny_tokenizer()
    picked = attribution_neurons(model, tokenizer, ["the cat sat on a mat"] * 3, model.model.layers[1].mlp, 6)
    assert len(picked) == 6
    assert len(set(picked.tolist())) == 6
    assert torch.equal(picked, torch.sort(picked).values)
    for param in model.parameters():
        assert param.grad is None
