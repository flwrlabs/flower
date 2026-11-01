from logging import INFO

import torch
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from vertical_fl.task import FEATURE_COLUMNS, ServerModel, load_labels

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
    reply_timeout = float(context.run_config["reply-timeout"])

    # Use the same shuffled row order as the clients' feature partitions.
    labels = load_labels(in_feature_dim_clientapp)
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
            timeout=reply_timeout,
        )

        # 2. Complete forward pass and compute loss
        optimizer.zero_grad()  # Clear gradients before backward pass
        output = head(embeddings)
        loss = criterion(output, labels)
        loss.backward()

        # 3. Extract gradients w.r.t. embeddings
        embeddings_grad = embeddings.grad.split(
            [out_feature_dim_clientapp] * len(node_ids), dim=1
        )

        # 4. Report training metrics from the same forward pass, before updates.
        if i % eval_interval == 0 or i == num_rounds:
            predicted = (output.detach() > 0.5).float()
            accuracy = (predicted == labels).float().mean().item() * 100
            log(INFO, f"Round {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            accuracies.append((i, accuracy))
            losses.append((i, loss.item()))

        # 5. Update every model before requesting the next round's embeddings.
        optimizer.step()
        send_gradients_to_clients(
            grid, node_pos_mapping, embeddings_grad, timeout=reply_timeout
        )

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
    embeddings_grad: tuple[torch.Tensor, ...],
    timeout: float,
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

    # Wait for every update to complete, including in the final round.
    log(INFO, "Sending gradients to %s nodes...", len(messages))
    replies = grid.send_and_receive(messages, timeout=timeout)
    acknowledged = set()
    for reply in replies:
        node_id = reply.metadata.src_node_id
        if reply.has_error():
            raise RuntimeError(
                f"Gradient update failed at node {node_id}: {reply.error}"
            )
        acknowledged.add(node_id)
    if acknowledged != set(node_pos_mapping):
        raise RuntimeError("Did not receive all gradient update acknowledgements")


def get_remote_embeddings(
    grid: Grid,
    node_ids: list[int],
    timeout: float,
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
    replies = list(grid.send_and_receive(messages, timeout=timeout))
    log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

    embeddings: dict[int, torch.Tensor] = {}

    # Convert all embeddings back to pytorch tensors
    # and place them in the corresponding feature segment
    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        node_id = reply.metadata.src_node_id
        if reply.has_error():
            raise RuntimeError(
                f"Embedding query failed at node {node_id}: {reply.error}"
            )
        pos = reply.content["config"]["pos"]
        embeddings[pos] = torch.from_numpy(reply.content["arrays"]["embedding"].numpy())
        node_pos_mapping[node_id] = pos

    # VFL needs every feature block; a missing party cannot be replaced by zeros.
    if set(embeddings) != set(range(len(node_ids))):
        raise RuntimeError("Did not receive embeddings from every feature party")
    ordered_embeddings = [embeddings[pos] for pos in range(len(node_ids))]
    return torch.cat(ordered_embeddings, dim=1).requires_grad_(), node_pos_mapping
