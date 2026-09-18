from .dataset import FetsDatasetLoader

DATASET_REGISTRY = {
    "dataset": FetsDatasetLoader,
    "fets2022": FetsDatasetLoader,
    "default": FetsDatasetLoader,
}