def build_prompt(
    role: str,
    task: str,
    user_input: str,
    constraints: list[str] | None = None,
) -> str:

    constraints_text = ""

    if constraints:
        constraints_text = "\n".join(
            f"- {item}"
            for item in constraints
        )

    return f"""
            Role:
            {role}

            Constraints:
            {constraints_text}

            Task:
            {task}

            Input:
            <input>
            {user_input}
            </input>
                """.strip()