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
    stop_sequence=None,
    token_bias=None,
):
    """
    Generate text autoregressively from a Hugging Face causal language model.

    Generation flow:

        Prompt
          ↓
        Tokenization
          ↓
        Model
          ↓
        Logits
          ↓
        Logit Bias
          ↓
        Temperature
          ↓
        Top-K
          ↓
        Top-P
          ↓
        Token Selection
          ↓
        Stop Sequence / EOS
          ↓
        Generated Text
    """

    if max_new_token < 0:
        raise ValueError("max_new_token must be >= 0.")

    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be >= 0.")

    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1.")

    device = get_model_device(model)

    # ---------------------------------------------------------
    # 1. Tokenize prompt
    # ---------------------------------------------------------

    inputs = tokenize(
        tokenizer,
        prompt,
        device,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    prompt_length = input_ids.shape[1]

    # ---------------------------------------------------------
    # 2. Prepare stop sequences ONCE
    # ---------------------------------------------------------

    stop_token_sequences = []

    for sequence in stop_sequence or []:

        token_ids = tokenizer.encode(
            sequence,
            add_special_tokens=False,
        )

        if token_ids:
            stop_token_sequences.append(token_ids)

    stop_detector = StopSequenceDetector(
        stop_token_sequences
    )

    # ---------------------------------------------------------
    # 3. Generation state
    # ---------------------------------------------------------

    past_key_values = None
    matched_stop_length = 0

    # ---------------------------------------------------------
    # 4. Autoregressive generation
    # ---------------------------------------------------------

    for _ in range(max_new_token):

        # -----------------------------------------------------
        # First generation step
        # Send the complete prompt.
        # -----------------------------------------------------

        if past_key_values is None:

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        # -----------------------------------------------------
        # Subsequent steps
        # Only send the newest token because of KV cache.
        # -----------------------------------------------------

        else:

            outputs = model(
                input_ids=input_ids[:, -1:],
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        # -----------------------------------------------------
        # Extract logits for the next token
        #
        # outputs.logits:
        # [batch_size, sequence_length, vocabulary_size]
        #
        # We only need the last position.
        # -----------------------------------------------------

        next_token_logits = outputs.logits[:, -1, :]

        # -----------------------------------------------------
        # Update KV cache
        # -----------------------------------------------------

        past_key_values = outputs.past_key_values

        # -----------------------------------------------------
        # 5. Apply logit bias
        # -----------------------------------------------------

        if token_bias:
            next_token_logits = apply_logit_bias(
                next_token_logits,
                token_bias,
            )

        # -----------------------------------------------------
        # 6. Apply temperature
        #
        # temperature == 0 is handled by select_next_token()
        # as greedy decoding.
        # -----------------------------------------------------

        next_token_logits = apply_temperature(
            next_token_logits,
            temperature,
        )

        # -----------------------------------------------------
        # 7. Apply Top-K
        # -----------------------------------------------------

        next_token_logits = apply_top_k(
            next_token_logits,
            top_k,
        )

        # -----------------------------------------------------
        # 8. Apply Top-P
        # -----------------------------------------------------

        next_token_logits = apply_top_p(
            next_token_logits,
            top_p,
        )

        # -----------------------------------------------------
        # 9. Select next token
        # -----------------------------------------------------

        next_token_id = select_next_token(
            next_token_logits,
            temperature,
        )

        # -----------------------------------------------------
        # 10. Append generated token
        # -----------------------------------------------------

        input_ids = torch.cat(
            [
                input_ids,
                next_token_id,
            ],
            dim=-1,
        )

        # -----------------------------------------------------
        # 11. Extend attention mask
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 12. Check EOS
        # -----------------------------------------------------

        if (
            tokenizer.eos_token_id is not None
            and next_token_id.item()
            == tokenizer.eos_token_id
        ):
            break

        # -----------------------------------------------------
        # 13. Check stop sequences
        #
        # Only inspect generated tokens, not the prompt.
        # -----------------------------------------------------

        generated_tokens = input_ids[
            0,
            prompt_length:,
        ].tolist()

        stop_length = stop_detector.match(
            generated_tokens
        )

        if stop_length > 0:
            matched_stop_length = stop_length
            break

    # ---------------------------------------------------------
    # 14. Extract generated tokens
    # ---------------------------------------------------------

    generated_ids = input_ids[
        0,
        prompt_length:,
    ]

    # ---------------------------------------------------------
    # 15. Remove matched stop sequence from final output
    # ---------------------------------------------------------

    if matched_stop_length > 0:
        generated_ids = generated_ids[
            :-matched_stop_length
        ]

    # ---------------------------------------------------------
    # 16. Decode
    # ---------------------------------------------------------

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()