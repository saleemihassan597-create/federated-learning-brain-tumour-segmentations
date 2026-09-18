from .base import BaseTrainer


class FedMedianTrainer(BaseTrainer):
    """FedMedian local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
