"""Generate client data-distribution CSV report matching terminal output."""

from __future__ import annotations

import tomllib
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

from dataset import _global_test_split, read_partitioning, split_records


def _count_classes_in_records(records: list[dict]) -> dict[int, int]:
    """Count number of cases containing each class label {0, 1, 2, 3}."""
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for rec in records:
        label_path = rec.get("label")
        if label_path and Path(label_path).exists():
            try:
                data = nib.load(label_path).get_fdata()
                uniques = set(np.unique(data))
                mapped = {3 if x == 4 else int(x) for x in uniques}
                for c in (0, 1, 2, 3):
                    if c in mapped:
                        counts[c] += 1
            except Exception:
                for c in (0, 1, 2, 3):
                    counts[c] += 1
        else:
            for c in (0, 1, 2, 3):
                counts[c] += 1
    return counts


def generate_client_data_distribution(
    pyproject_path: str | Path = "pyproject.toml",
    requested_clients: int | None = None,
) -> Path:
    """Calculate and save client-wise class distribution matching terminal output."""
    config_path = Path(pyproject_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]

    root = Path(config["data-root"])
    csv_path = Path(config["partition-csv"])
    global_test_fraction = float(config.get("global-test-fraction", 0.15))
    seed = int(config.get("seed", 42))
    output_dir = Path(config.get("output-dir", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = read_partitioning(root, csv_path)
    num_clients = int(config["num-clients"]) if requested_clients is None else requested_clients
    if not 1 <= num_clients <= len(groups):
        raise ValueError(f"Requested {num_clients} clients; partitioning CSV contains {len(groups)} institutions")

    client_rows = []

    print("\n" + "=" * 65)
    print("   CLIENT DATA DISTRIBUTION SUMMARY")
    print("=" * 65)

    for i in range(num_clients):
        partition_id, records = groups[i]
        total_cases = len(records)

        # Re-use project's _global_test_split logic
        trainval, test = _global_test_split(
            records,
            global_test_fraction=global_test_fraction,
            seed=seed + i,
        )

        global_test_cases = len(test)
        client_cases = len(trainval)

        # Re-use project's split_records logic
        train_recs, val_recs = split_records(trainval, validation_fraction=0.15, seed=seed + i)
        train_cases = len(train_recs)
        validation_cases = len(val_recs)

        # Calculate class counts across client cases
        class_counts = _count_classes_in_records(trainval)

        client_rows.append({
            "client_id": f"{i:02d}",
            "partition_id": str(partition_id),
            "total_cases": total_cases,
            "global_test_cases": global_test_cases,
            "client_cases": client_cases,
            "train_cases": train_cases,
            "validation_cases": validation_cases,
            "class_0": class_counts[0],
            "class_1": class_counts[1],
            "class_2": class_counts[2],
            "class_3": class_counts[3],
        })

        print(
            f"Institution {i:02d} (Partition ID {partition_id}): Total={total_cases} | "
            f"Global Test={global_test_cases} | Train={train_cases} | Val={validation_cases} | "
            f"Class 0={class_counts[0]}, Class 1={class_counts[1]}, Class 2={class_counts[2]}, Class 3={class_counts[3]}"
        )

    print("=" * 65 + "\n")

    # Save single CSV report file
    out_file = output_dir / "client_data_distribution.csv"
    pd.DataFrame(client_rows).to_csv(out_file, index=False)
    print(f"[Data Distribution] Saved CSV report to: {out_file}\n")

    return out_file


# Alias for compatibility
generate_reports = generate_client_data_distribution

if __name__ == "__main__":
    generate_client_data_distribution()
