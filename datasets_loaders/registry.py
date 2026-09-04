from .brain_tumor_dataset import BrainTumorLoader
#from .kaggle_mri_dataset import KaggleMRILoader


DATASET_REGISTRY = {
    "brain_tumor": BrainTumorLoader,
#    "kaggle_mri": KaggleMRILoader,
}