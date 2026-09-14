import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    k = min(top_k, logits.shape[-1])
    threshold = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    indices_to_remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def select_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def apply_logit_bias(logits: torch.Tensor, token_bias: dict[int, float]) -> torch.Tensor:
    result = logits.clone()
    vocabulary_size = result.shape[-1]
    for token_id, bias in token_bias.items():
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(f"token_id out of range: {token_id}")
        result[..., token_id] += bias
    return result
