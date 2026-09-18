from .ML_model import build_model
from .registry import MODEL_REGISTRY


def create_model(model_name: str = "unet"):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]().build()
    return build_model()