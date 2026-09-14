from .base import build_prompt


def build_ticket_classification_prompt(ticket: str) -> str:
    return build_prompt(
        role="You are a precise customer-support ticket classification assistant.",
        task=(
            "Classify the ticket and produce JSON with exactly these fields: "
            "category, sentiment, urgency, summary."
        ),
        user_input=ticket,
        constraints=[
            "category must be one of: technical, account, delivery, billing, subscription",
            "sentiment must be one of: positive, negative, neutral",
            "urgency must be one of: low, medium, high",
            "summary must be concise and factual",
            "Return JSON only; do not use markdown or explanations",
        ],
    )


# Backward-compatible alias for the original project name.
build_ticket_classifier = build_ticket_classification_prompt
