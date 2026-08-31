from .base import build_prompt

def build_ticket_classifier(
        user_query
):
    return build_prompt(
        role="You are a precise ticket classification assistant.",

        task="Classify the customer ticket.",

        user_input=user_query,

        constraints=[
            "Return only the requested result.",
            "Do not provide explanations.",
            "Do not use markdown.",
            "summary must be one sentence as max"
        ],
    )