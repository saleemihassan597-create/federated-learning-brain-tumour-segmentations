from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from algorithms import get_trainer
from models import create_model
from task import load_data, test

app = ClientApp()


def _append_client_csv(context: Context, row: dict) -> None:
    output_dir = Path(context.run_config.get("output-dir", "artifacts"))
    algorithm = str(context.run_config.get("algorithm", "federated")).lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / f"{algorithm}_fets2022_client_history.csv"
    df = pd.DataFrame([row])
    if not csv_file.exists():
        df.to_csv(csv_file, index=False, mode="w")
    else:
        df.to_csv(csv_file, index=False, mode="a", header=False)


def _get_device(context: Context) -> torch.device:
    requested = str(context.run_config.get("device", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train global model on local client partition using configured algorithm."""
    start_time = time.time()
    partition_id = int(context.node_config["partition-id"])
    device = _get_device(context)

    # Load model and set weights
    model_name = context.run_config.get("model_name", "unet")
    model = create_model(model_name).to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load data loader
    trainloader, _ = load_data(partition_id, context.run_config)

    # Get algorithm trainer (e.g. fedavg or fedprox)
    algorithm = context.run_config.get("algorithm", "fedavg")
    proximal_mu = float(msg.content["config"].get("proximal_mu", context.run_config.get("proximal-mu", 0.0)))
    trainer = get_trainer(algorithm, proximal_mu=proximal_mu)

    # Train
    local_epochs = int(context.run_config.get("local-epochs", 1))
    lr = float(msg.content["config"].get("lr", context.run_config.get("learning-rate", 1e-4)))
    loss = trainer.train(model, trainloader, local_epochs, lr, device)

    elapsed = time.time() - start_time
    print(
        f"[Institution {partition_id:02d}] Training Finished | "
        f"Loss: {loss:.4f} | Time: {elapsed:.2f}s | "
        f"Examples: {len(trainloader.dataset)}"
    )

    _append_client_csv(context, {
        "institution_id": f"{partition_id:02d}",
        "phase": "train",
        "loss": loss,
        "time_sec": elapsed,
        "examples": len(trainloader.dataset),
    })

    metrics = MetricRecord({
        "train_loss": loss,
        "train_time_sec": elapsed,
        "num-examples": len(trainloader.dataset),
    })
    return Message(
        content=RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate global model on client held-out partition."""
    start_time = time.time()
    partition_id = int(context.node_config["partition-id"])
    device = _get_device(context)

    # Load model and weights
    model_name = context.run_config.get("model_name", "unet")
    model = create_model(model_name).to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # Load data loader
    _, valloader = load_data(partition_id, context.run_config)

    # Evaluate
    eval_loss, region_metrics = test(model, valloader, device)

    elapsed = time.time() - start_time
    dice_et = region_metrics.get("dice_et", 0.0)
    dice_tc = region_metrics.get("dice_tc", 0.0)
    dice_wt = region_metrics.get("dice_wt", 0.0)
    hd95_et = region_metrics.get("hd95_et", 0.0)
    hd95_tc = region_metrics.get("hd95_tc", 0.0)
    hd95_wt = region_metrics.get("hd95_wt", 0.0)

    print(
        f"[Institution {partition_id:02d}] Evaluation Finished | "
        f"Loss: {eval_loss:.4f} | "
        f"Dice (ET/TC/WT): {dice_et:.4f} / {dice_tc:.4f} / {dice_wt:.4f} | "
        f"HD95 (ET/TC/WT): {hd95_et:.2f} / {hd95_tc:.2f} / {hd95_wt:.2f} | "
        f"Time: {elapsed:.2f}s | Examples: {len(valloader.dataset)}"
    )

    _append_client_csv(context, {
        "institution_id": f"{partition_id:02d}",
        "phase": "evaluate",
        "loss": eval_loss,
        "dice_et": dice_et,
        "dice_tc": dice_tc,
        "dice_wt": dice_wt,
        "hd95_et": hd95_et,
        "hd95_tc": hd95_tc,
        "hd95_wt": hd95_wt,
        "time_sec": elapsed,
        "examples": len(valloader.dataset),
    })

    metrics = MetricRecord({
        "eval_loss": eval_loss,
        "eval_dice_et": dice_et,
        "eval_dice_tc": dice_tc,
        "eval_dice_wt": dice_wt,
        "eval_hd95_et": hd95_et,
        "eval_hd95_tc": hd95_tc,
        "eval_hd95_wt": hd95_wt,
        "eval_time_sec": elapsed,
        "num-examples": len(valloader.dataset),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
