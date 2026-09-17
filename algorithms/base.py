from abc import ABC, abstractmethod
import torch


class BaseTrainer(ABC):
    """Common interface every algorithm trainer must implement."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def compute_loss(self, model, outputs, labels, criterion) -> torch.Tensor:
        """Return the (possibly augmented) loss for a single batch."""
        raise NotImplementedError

    def on_train_start(self, model):
        """Hook called once before local training begins.
        Override if the algorithm needs to snapshot state (e.g. global params)."""
        pass

    def train(self, model, trainloader, epochs: int, lr: float, device):
        """Shared training loop matching maam's structure and user's AdamW/FeTS training flow."""
        self.on_train_start(model)

        model.to(device)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for batch in trainloader:
                images = (batch["image"] if "image" in batch else batch["img"]).to(device)
                labels = batch["label"].to(device).long()
                if labels.ndim == 5:
                    labels = labels.squeeze(1)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = self.compute_loss(model, outputs, labels, criterion)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

        return total_loss / max(steps, 1)