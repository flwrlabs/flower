from logging import INFO

import torch
from datasets import load_dataset
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from vertical_fl.task import FEATURE_COLUMNS, ServerModel, evaluate_head_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    if sum(in_feature_dim_clientapp) != len(FEATURE_COLUMNS):
        raise ValueError(
            "The sum of feature splits must equal the total number of features "
            f"(got {sum(in_feature_dim_clientapp)} vs. {len(FEATURE_COLUMNS)})."
        )
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]

    # Get dataset
    dataset = load_dataset("julien-c/titanic-survival", split="train")
    labels = dataset["Survived"]
    labels = torch.tensor(labels).float().unsqueeze(1)

    # Serverapp model
    head = None
    optimizer = None
    criterion = torch.nn.BCELoss()

    # Track metrics
    eval_interval = 25  # Evaluate the model every 25 rounds
    accuracies: list[tuple[int, float]] = []
    losses: list[tuple[int, float]] = []

    for i in range(1, num_rounds + 1):
        log(INFO, "")
        log(INFO, f"--- ServerApp Round {i} / {num_rounds} ---")

        node_ids = list(grid.get_node_ids())

        if len(node_ids) != len(in_feature_dim_clientapp):
            raise ValueError(
                "The number of feature splits must equal the number of nodes "
                f"(got {len(in_feature_dim_clientapp)} vs. {len(node_ids)})."
            )

        if head is None:
            # The server model's input size is determined by the number of clients
            # and the output feature dimension of each embedding produced by the clients
            head = ServerModel(input_size=out_feature_dim_clientapp * len(node_ids))
            optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

        # 1. Get embeddings from all clients
        embeddings, node_pos_mapping = get_remote_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_nodes=len(labels),
            embedding_dim=out_feature_dim_clientapp,
        )

        # 2. Complete forward pass and compute loss
        optimizer.zero_grad()  # Clear gradients before backward pass
        output = head(embeddings)
        loss = criterion(output, labels)
        loss.backward()

        # 4. Extract gradients w.r.t. embeddings
        embeddings_grad = embeddings.grad.split(
            [out_feature_dim_clientapp] * len(node_ids), dim=1
        )

        # Update the head model
        optimizer.step()

        # 3. Compute accuracy using updated head model
        if i % eval_interval == 0 or i == num_rounds:
            accuracy = evaluate_head_model(head, embeddings, labels)
            log(INFO, f"Round {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            accuracies.append((i, accuracy))
            losses.append((i, loss.item()))

        # 5. Send gradients to clients
        send_gradients_to_clients(grid, node_pos_mapping, embeddings_grad)

    # Log final results
    log(INFO, "")
    log(INFO, "=== Final Results ===")
    for (round_num, accuracy), (_, loss_value) in zip(accuracies, losses):
        log(
            INFO,
            f"Round {round_num} -> Loss: {loss_value:.4f} | Accuracy: {accuracy:.2f}%",
        )


def send_gradients_to_clients(
    grid: Grid,
    node_pos_mapping: dict[int, int],
    embeddings_grad: list[torch.Tensor],
) -> None:
    """Send gradients to clients."""

    # Create messages that target method in ClientApp with
    # @app.train("apply_gradients") decorator
    messages = []
    for node_id, pos in node_pos_mapping.items():
        arrc = ArrayRecord({"local-gradients": Array(embeddings_grad[pos].numpy())})
        message = Message(
            content=RecordDict({"gradients": arrc}),
            message_type="train.apply_gradients",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages, but don't wait for results (no replies expected)
    log(INFO, "Sending gradients to %s nodes...", len(messages))
    grid.push_messages(messages)


def get_remote_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_nodes: int,
    embedding_dim: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    """Get embeddings from sampled remote nodes."""

    # Create messages that target method in ClientApp with
    # @app.query("generate_embeddings") decorator
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=RecordDict(),
            message_type="query.generate_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

    embeddings = torch.zeros((num_nodes, embedding_dim * len(node_ids)))

    # Convert all embeddings back to pytorch tensors
    # and place them in the corresponding feature segment
    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping
