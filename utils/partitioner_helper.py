import tomllib
from pathlib import Path

from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)

PARTITIONERS = {
    "iid": lambda cfg, num_partitions: IidPartitioner(num_partitions=num_partitions),
    "dirichlet": lambda cfg, num_partitions: DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=cfg["alpha"],
    ),
    "shard": lambda cfg, num_partitions: ShardPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_shards_per_partition=cfg["num_shards"],
    ),
}

def get_partitioner(num_partitions):
    
    config = {}

    BASE_DIR = Path(__file__).resolve().parent.parent 
    CONFIG_PATH = BASE_DIR / "pyproject.toml"

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    cfg = config["partition"]

    try:
        return PARTITIONERS[cfg["type"].lower()](cfg, num_partitions)
    except KeyError:
        raise ValueError(f"Unsupported partition type: {cfg['type']}")
        