from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference

from datasets_loaders.fets_dataset import (
    client_records,
    fets_region_metrics,
    make_loaders,
    read_partitioning,
)


def load_data(partition_id: int, config: dict):
    """Load train and validation data loaders for a specific client partition."""
    data_root = config["data-root"]
    partition_csv = config["partition-csv"]
    batch_size = int(config.get("batch-size", 1))
    cache_rate = float(config.get("cache-rate", 0.0))
    seed = int(config.get("seed", 42)) + int(partition_id)
    num_workers = int(config.get("num-workers", 0))

    records = client_records(data_root, partition_csv, int(partition_id))
    return make_loaders(
        records,
        batch_size=batch_size,
        cache_rate=cache_rate,
        seed=seed,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_centralized_dataset(config: dict):
    """Load pooled dataset loaders for centralized baseline evaluation."""
    data_root = config["data-root"]
    partition_csv = config["partition-csv"]
    batch_size = int(config.get("batch-size", 1))
    cache_rate = float(config.get("cache-rate", 0.0))
    seed = int(config.get("seed", 42))
    num_workers = int(config.get("num-workers", 0))

    all_groups = read_partitioning(data_root, partition_csv)
    records = [rec for _, group in all_groups for rec in group]
    return make_loaders(
        records,
        batch_size=batch_size,
        cache_rate=cache_rate,
        seed=seed,
        num_workers=num_workers,
        pin_memory=False,
    )


def test(model, valloader, device: torch.device):
    """Evaluate model on validation loader using sliding window inference."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    totals = {
        "loss": 0.0,
        "dice_et": 0.0, "dice_tc": 0.0, "dice_wt": 0.0,
        "hd95_et": 0.0, "hd95_tc": 0.0, "hd95_wt": 0.0,
    }
    with torch.no_grad():
        for batch in valloader:
            images = (batch["image"] if "image" in batch else batch["img"]).to(device)
            labels = batch["label"].to(device).long()
            logits = sliding_window_inference(
                images, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model
            )
            loss_val = criterion(logits, labels.squeeze(1) if labels.ndim == 5 else labels).item()
            totals["loss"] += loss_val
            metrics = fets_region_metrics(logits, labels)
            for key, val in metrics.items():
                totals[key] += val

    count = max(len(valloader), 1)
    results = {k: v / count for k, v in totals.items()}
    return results["loss"], results
