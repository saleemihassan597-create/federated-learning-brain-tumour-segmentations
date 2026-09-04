from datasets import Dataset, DatasetDict

from datasets_loaders.brain_tumor_dataset import BrainTumorLoader


def test_dataset_length():
    loader = BrainTumorLoader("../data/Brain-tumor-multimodal-image")
    dataset = loader.load()

    assert len(dataset) == 9618