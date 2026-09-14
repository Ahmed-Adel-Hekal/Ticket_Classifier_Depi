def build_prompt(
    role: str,
    task: str,
    user_input: str,
    constraints: list[str] | None = None,
) -> str:
    constraints_text = "\n".join(f"- {item}" for item in (constraints or []))
    return (
        f"Role:\n{role}\n\n"
        f"Constraints:\n{constraints_text}\n\n"
        f"Task:\n{task}\n\n"
        f"Input:\n<input>\n{user_input}\n</input>"
    )
