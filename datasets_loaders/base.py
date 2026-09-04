# datasets/base.py

from abc import ABC, abstractmethod
from datasets import Dataset


class BaseDatasetLoader(ABC):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def load(self) -> Dataset:
        """Return a HuggingFace Dataset"""
        pass