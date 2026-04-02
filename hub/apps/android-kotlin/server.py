"""Start a Flower server.

Derived from Flower Android example.
"""

from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvgAndroid

PORT = 8080


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5,
    }
    return config


def main():
    strategy = FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
    )

    try:
        # Start Flower server for 10 rounds of federated learning
        start_server(
            server_address=f"0.0.0.0:{PORT}",
            config=ServerConfig(num_rounds=10),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
