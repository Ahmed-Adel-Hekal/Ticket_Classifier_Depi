import torch

from .decoder import apply_logit_bias, apply_temperature, apply_top_k, apply_top_p, select_next_token
from .model import get_model_device
from .stopping import StopSequenceDetector
from .tokenizer import tokenize


@torch.no_grad()
def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_sequences: list[str] | None = None,
    token_bias: dict[int, float] | None = None,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")

    device = get_model_device(model)
    inputs = tokenize(tokenizer, prompt, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = input_ids.shape[1]

    stop_token_sequences = [
        tokenizer.encode(sequence, add_special_tokens=False)
        for sequence in (stop_sequences or [])
    ]
    stop_token_sequences = [sequence for sequence in stop_token_sequences if sequence]
    detector = StopSequenceDetector(stop_token_sequences)
    past_key_values = None

    for _ in range(max_new_tokens):
        kwargs = {
            "input_ids": input_ids if past_key_values is None else input_ids[:, -1:],
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        outputs = model(**kwargs)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        if token_bias:
            logits = apply_logit_bias(logits, token_bias)
        logits = apply_temperature(logits, temperature)
        logits = apply_top_k(logits, top_k)
        logits = apply_top_p(logits, top_p)
        next_token = select_next_token(logits, temperature)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
            dim=-1,
        )

        if tokenizer.eos_token_id is not None and torch.all(next_token == tokenizer.eos_token_id):
            break
        generated = input_ids[:, prompt_length:]
        if generated.shape[0] == 1 and detector.match(generated[0].tolist()):
            break

    generated_ids = input_ids[:, prompt_length:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
