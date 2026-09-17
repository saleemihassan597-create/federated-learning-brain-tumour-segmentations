import torch
from .base import BaseTrainer


class FedProxTrainer(BaseTrainer):
    """FedAvg + proximal term penalizing drift from the global model."""

    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = float(proximal_mu)
        self.global_params = None

    def on_train_start(self, model):
        # Snapshot global params BEFORE local training mutates them
        self.global_params = [p.detach().clone() for p in model.parameters()]

    def compute_loss(self, model, outputs, labels, criterion):
        loss = criterion(outputs, labels)
        if self.proximal_mu > 0.0 and self.global_params is not None:
            proximal_term = sum(
                torch.sum((local_p - global_p) ** 2)
                for local_p, global_p in zip(model.parameters(), self.global_params)
            )
            loss = loss + 0.5 * self.proximal_mu * proximal_term
        return loss