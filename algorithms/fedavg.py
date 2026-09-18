from .base import BaseTrainer


class FedAvgTrainer(BaseTrainer):
    """FedAvg local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)