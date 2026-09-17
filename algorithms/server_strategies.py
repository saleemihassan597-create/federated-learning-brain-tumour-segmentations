from flwr.serverapp.strategy import FedAvg, FedProx

STRATEGY_REGISTRY = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
}


def get_strategy(algorithm: str, **kwargs):
    algorithm_key = algorithm.lower()
    if algorithm_key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    strategy_cls = STRATEGY_REGISTRY[algorithm_key]

    params = dict(kwargs)
    if "weighted_by_key" not in params:
        params["weighted_by_key"] = "num-examples"

    return strategy_cls(**params)