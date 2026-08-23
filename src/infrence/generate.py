import torch

from .model import get_model_device
from .decoder import (
    apply_logit_bias,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    select_next_token,
)
from .tokinezer import tokenize
from .stopping import StopSequenceDetector

@torch.no_grad()
def generate_text(
    tokenizer,
    model,
    prompt,
    max_new_token=50,
    temperature=1.0,
    top_k=None,
    top_p=None,
    stop_sequence=None
):


    device = get_model_device(model)

    inputs = tokenize(
        tokenizer,
        prompt,
        device,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_length = input_ids.shape[1]

    past_key_values = None

    for _ in range(max_new_token):

        # First step:
        # Send the complete prompt to the model.
        if past_key_values is None:

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        else:

            outputs = model(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        # [batch_size, sequence_length, vocabulary_size]
        next_token_logits = outputs.logits[:, -1, :]

        # Update KV cache
        past_key_values = outputs.past_key_values

        # # stop_seq ---> token id 
        stop_sequence_token = []

        for s in stop_sequence :
            token_id = tokenizer.encode(
                s,
                add_special_tokens=False
            )

            if token_id :
                stop_sequence_token.append(token_id)

        stop_detector= StopSequenceDetector(stop_sequence_token)

        # Apply decoding strategies
        next_token_logits = apply_temperature(
            next_token_logits,
            temperature,
        )

        next_token_logits = apply_top_k(
            next_token_logits,
            top_k,
        )

        next_token_logits = apply_top_p(
            next_token_logits,
            top_p,
        )

        # Select next token
        next_token_id = select_next_token(
            next_token_logits,
            temperature,
        )

        # Stop when EOS is generated
        if (
            tokenizer.eos_token_id is not None
            and next_token_id.item()
            == tokenizer.eos_token_id
        ):
            break

        # Append generated token to sequence
        input_ids = torch.cat(
            [
                input_ids,
                next_token_id,
            ],
            dim=-1,
        )

        # Extend attention mask for the new token
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=-1,
        )

        # generated_idx = input_ids[0,prompt_length:,]

        # stop_flag = stop_sequence.match(generated_idx)

        # if stop_flag > 0 : 
        #     break 


    generated_ids = input_ids[:, prompt_length:]

    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()