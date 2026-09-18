from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @abstractmethod
    def load(self):
        """Load dataset partitions/records."""
        pass