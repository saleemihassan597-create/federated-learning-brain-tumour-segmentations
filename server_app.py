from pathlib import Path

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from algorithms.server_strategies import get_strategy
from models import create_model
from task import load_centralized_dataset, test

app = ServerApp()
model_name_global = "unet"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for Flower ServerApp."""
    global model_name_global
    config = context.run_config
    algorithm = str(config.get("algorithm", config.get("strategy", "fedavg"))).lower()
    model_name_global = str(config.get("model_name", "unet"))
    num_clients = int(config.get("num-clients", config.get("min-available-clients", 2)))
    num_rounds = int(config.get("num-server-rounds", 2))
    lr = float(config.get("learning-rate", 1e-4))

    # Load initial global model
    global_model = create_model(model_name_global)
    initial_arrays = ArrayRecord(global_model.state_dict())

    # Build strategy arguments
    strategy_kwargs = {
        "fraction_train": float(config.get("fraction-train", 1.0)),
        "fraction_evaluate": float(config.get("fraction-evaluate", 1.0)),
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
        "weighted_by_key": "num-examples",
    }

    if algorithm == "fedprox":
        strategy_kwargs["proximal_mu"] = float(config.get("proximal-mu", 0.01))

    strategy = get_strategy(algorithm, **strategy_kwargs)

    # Run strategy across server rounds
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr, "proximal_mu": strategy_kwargs.get("proximal_mu", 0.0)}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate if config.get("enable-global-eval", False) else None,
    )

    output_dir = Path(config.get("output-dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save final global model checkpoint
    checkpoint_name = "final_model.pt" if config.get("save-model", False) else f"{algorithm}_fets2022_final.pt"
    model_checkpoint_path = output_dir / checkpoint_name
    torch.save(result.arrays.to_torch_state_dict(), model_checkpoint_path)
    print(f"\n[Server] Model checkpoint saved to: {model_checkpoint_path}")

    # Build and save round-by-round metrics to CSV
    rows = []
    rounds = sorted(set(list(result.train_metrics_clientapp.keys()) + list(result.evaluate_metrics_clientapp.keys())))
    for r in rounds:
        row = {"round": r, "strategy": algorithm}
        if r in result.train_metrics_clientapp:
            for k, v in dict(result.train_metrics_clientapp[r]).items():
                row[k] = v
        if r in result.evaluate_metrics_clientapp:
            for k, v in dict(result.evaluate_metrics_clientapp[r]).items():
                row[k] = v
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{algorithm}_fets2022_metrics.csv"
        df.to_csv(csv_path, index=False)
        print("\n" + "=" * 78)
        print(f"        FEDERATED LEARNING RESULTS SUMMARY ({algorithm.upper()})")
        print("=" * 78)
        print(df.to_string(index=False))
        print("=" * 78)
        print(f"[Server] Round results saved to CSV: {csv_path}\n")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global model on pooled dataset if enabled."""
    model = create_model(model_name_global)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = {"data-root": "E:/rnds/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData",
              "partition-csv": "E:/rnds/MICCAI_FeTS2022_TrainingData/MICCAI_FeTS2022_TrainingData/partitioning_1.csv"}
    _, valloader = load_centralized_dataset(config)
    eval_loss, region_metrics = test(model, valloader, device)
    metrics_dict = {"loss": eval_loss}
    metrics_dict.update(region_metrics)
    return MetricRecord(metrics_dict)
