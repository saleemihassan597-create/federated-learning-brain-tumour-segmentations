from .base import BaseTrainer


class FedAdagradTrainer(BaseTrainer):
    """FedAdagrad local trainer."""

    def compute_loss(self, model, outputs, labels, criterion):
        return criterion(outputs, labels)
