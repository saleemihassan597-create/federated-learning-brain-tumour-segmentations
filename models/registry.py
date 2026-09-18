from .ML_model import UNetModel

MODEL_REGISTRY = {
    "ml_model": UNetModel,
    "unet": UNetModel,
    "default": UNetModel,
}