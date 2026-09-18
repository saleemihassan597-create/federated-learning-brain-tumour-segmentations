from .fedavg import FedAvgTrainer
from .fedprox import FedProxTrainer
from .fedavgm import FedAvgMTrainer
from .fedadagrad import FedAdagradTrainer
from .fedadam import FedAdamTrainer
from .fedyogi import FedYogiTrainer
from .qfedavg import QFedAvgTrainer
from .fedmedian import FedMedianTrainer
from .fedtrimmedavg import FedTrimmedAvgTrainer

TRAINER_REGISTRY = {
    "fedavg": FedAvgTrainer,
    "fedprox": FedProxTrainer,
    "fedavgm": FedAvgMTrainer,
    "fedadagrad": FedAdagradTrainer,
    "fedadam": FedAdamTrainer,
    "fedyogi": FedYogiTrainer,
    "qfedavg": QFedAvgTrainer,
    "fedmedian": FedMedianTrainer,
    "fedtrimmedavg": FedTrimmedAvgTrainer,
}


def get_trainer(algorithm: str, **kwargs):
    alg = algorithm.lower()
    if alg not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(TRAINER_REGISTRY)}")
    return TRAINER_REGISTRY[alg](**kwargs)