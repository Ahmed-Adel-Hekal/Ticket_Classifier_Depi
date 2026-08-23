import outlines


def create_structured_model(model, tokenizer):

    return outlines.from_transformers(
        model,
        tokenizer,
    )