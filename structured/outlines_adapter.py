def create_structured_model(model, tokenizer):
    """Create an Outlines model lazily so the core package has no hard dependency on it."""
    try:
        import outlines
    except ImportError as exc:
        raise RuntimeError(
            "Outlines is optional. Install the local structured-generation dependencies first."
        ) from exc
    return outlines.from_transformers(model, tokenizer)
