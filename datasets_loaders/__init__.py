from .registry import DATASET_REGISTRY
from .dataset import (
    PATCH_SIZE,
    RemapFetsLabeld,
    client_records,
    fets_region_metrics,
    make_loaders,
    read_partitioning,
)


def create_dataset(name: str = "fets2022", root_dir: str = "", **kwargs):
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name](root_dir, **kwargs)
    return DATASET_REGISTRY["default"](root_dir, **kwargs)