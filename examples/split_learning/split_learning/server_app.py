"""ServerApp for the split learning demo."""

from __future__ import annotations

import torch
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from split_learning.strategy import SplitLearningStrategy
from split_learning.task import ServerNet


def server_fn(context: Context) -> ServerAppComponents:
    """Construct the ServerApp components."""
    run_config = context.run_config
    num_rounds = int(run_config.get("num-server-rounds", 3))
    server_lr = float(run_config.get("server-learning-rate", 0.05))
    clients_per_round = int(run_config.get("clients-per-round", 2))
    embedding_size = int(run_config.get("embedding-size", 128))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = ServerNet(embedding_size=embedding_size)
    optimizer = torch.optim.SGD(server_model.parameters(), lr=server_lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    strategy = SplitLearningStrategy(
        server_model=server_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        clients_per_round=clients_per_round,
        min_available_clients=clients_per_round,
        device=device,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
