from .brain_tumor_dataset import BrainTumorLoader
from .fets_dataset import FetsDatasetLoader

DATASET_REGISTRY = {
    "fets2022": FetsDatasetLoader,
    "brain_tumor": BrainTumorLoader,
}