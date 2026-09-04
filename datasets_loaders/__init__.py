from .registry import DATASET_REGISTRY


def create_dataset(name: str, root_dir: str):

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    return DATASET_REGISTRY[name](root_dir)