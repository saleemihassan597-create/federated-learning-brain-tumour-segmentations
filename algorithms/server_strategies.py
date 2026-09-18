from .FL_methods import build_strategy, make_fedavg, make_fedprox


def get_strategy(strategy_name: str, config: dict, num_clients: int):
    return build_strategy(strategy_name, config, num_clients)