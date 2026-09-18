from .base import BaseTrainer


class FedAdamTrainer(BaseTrainer):
    """FedAdam local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
