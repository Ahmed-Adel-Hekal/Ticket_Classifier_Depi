from .base import build_prompt


def build_ticket_summary_prompt(ticket: str) -> str:
    return build_prompt(
        role="You are a concise customer-support ticket summarizer.",
        task="Summarize the customer ticket in one or two factual sentences.",
        user_input=ticket,
        constraints=[
            "Include the main issue and relevant requested action",
            "Do not invent details",
            "Return plain text only",
        ],
    )


# Backward-compatible alias for the original project name.
build_ticket_classifier = build_ticket_summary_prompt
