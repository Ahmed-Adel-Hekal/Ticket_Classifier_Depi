from prompts.classifier import build_ticket_classification_prompt


def test_classifier_prompt_contains_ticket_and_constraints():
    prompt = build_ticket_classification_prompt("I was charged twice")
    assert "I was charged twice" in prompt
    assert "category" in prompt
    assert "Return JSON only" in prompt
