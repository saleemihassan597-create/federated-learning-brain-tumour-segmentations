from pathlib import Path
from datasets import Dataset, Image
from .base import BaseDatasetLoader


class BrainTumorLoader(BaseDatasetLoader):

    def load(self):
        dataset_root = Path(self.root_dir)
        records = []
        modalities = {
            "Brain Tumor CT scan Images": "CT",
            "Brain Tumor MRI images": "MRI",
        }
        labels = {
            "Healthy": 0,
            "Tumor": 1,
        }

        for folder_name, modality in modalities.items():
            modality_path = dataset_root / folder_name

            for class_name, label in labels.items():
                class_path = modality_path / class_name

                for img_path in class_path.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                        records.append(
                            {
                                "img": str(img_path),
                                "label": label,
                                "modality": modality,
                            }
                        )
        
        dataset = Dataset.from_list(records)
        dataset = dataset.cast_column("img", Image())

        brain_dataset = dataset.train_test_split(
            test_size=0.2,
            seed=42,
        )
        
        return brain_dataset