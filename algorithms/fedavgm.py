from .base import BaseTrainer


class FedAvgMTrainer(BaseTrainer):
    """FedAvgM (Server Momentum) local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
