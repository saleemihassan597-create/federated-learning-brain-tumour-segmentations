from .base import BaseTrainer


class FedTrimmedAvgTrainer(BaseTrainer):
    """FedTrimmedAvg local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
