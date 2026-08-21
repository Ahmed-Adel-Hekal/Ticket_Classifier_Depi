import outlines

from schemas import TicketOutput


def create_structured_model(model, tokenizer):

    return outlines.from_transformers(
        model,
        tokenizer,
    )