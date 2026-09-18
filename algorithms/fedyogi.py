from .base import BaseTrainer


class FedYogiTrainer(BaseTrainer):
    """FedYogi local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
