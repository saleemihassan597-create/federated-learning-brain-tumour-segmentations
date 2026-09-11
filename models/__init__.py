from .registry import MODEL_REGISTRY


def create_model(model_name: str):
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    return model_class().build()