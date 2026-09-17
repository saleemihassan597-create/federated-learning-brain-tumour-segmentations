from .base import BaseModel
from .mobilenet import MobileNetModel
from .resnet import ResNetModel
from .unet import UNetModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "unet": UNetModel,
    "mobilenet": MobileNetModel,
    "resnet": ResNetModel,
}