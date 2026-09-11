from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(ABC):
    """Common interface for all federated learning models."""

    @abstractmethod
    def build(self) -> nn.Module:
        """Create and return the PyTorch model."""
        raise NotImplementedError