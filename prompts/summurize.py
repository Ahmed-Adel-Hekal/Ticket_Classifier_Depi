from .base import build_prompt

def build_ticket_classifier(
        user_query
):
    return build_prompt(
        role="You are Text summurizer.",

        task="Summurize customer ticket.",

        user_input=user_query,

        constraints=[
            "Return only the requested result.",
            "Do not provide explanations.",
            "Do not use markdown.",
        ],
    )