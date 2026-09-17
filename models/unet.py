import torch.nn as nn
from monai.networks.nets import UNet

from .base import BaseModel


class UNetModel(BaseModel):
    """3D MONAI UNet model for FeTS 2022 segmentation."""

    def build(self) -> nn.Module:
        return UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
