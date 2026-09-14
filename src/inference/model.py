def load_model(model_name: str | None = None):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the optional local dependencies to load a model") from exc

    from .config import MODEL_NAME
    name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return tokenizer, model


def get_model_device(model):
    return model.get_input_embeddings().weight.device
