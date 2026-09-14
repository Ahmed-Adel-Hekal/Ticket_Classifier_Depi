import pytest


def torch_or_skip():
    return pytest.importorskip("torch")


def test_logit_bias_applies_to_vocabulary_dimension():
    torch = torch_or_skip()
    from src.inference.decoder import apply_logit_bias

    logits = torch.zeros((2, 5))
    result = apply_logit_bias(logits, {3: 2.5})
    assert result[:, 3].tolist() == [2.5, 2.5]
    assert result[:, 0].tolist() == [0.0, 0.0]


def test_top_k_keeps_k_values_per_batch():
    torch = torch_or_skip()
    from src.inference.decoder import apply_top_k

    result = apply_top_k(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), 2)
    assert torch.isfinite(result).sum().item() == 2


def test_invalid_top_p_is_rejected():
    torch_or_skip()
    from src.inference.decoder import apply_top_p

    with pytest.raises(ValueError):
        apply_top_p(__import__("torch").zeros((1, 3)), 0)
