"""pandas_example: A Flower / Pandas app."""

import random
import time
from collections.abc import Iterable
from logging import INFO

import numpy as np
from flwr.app import Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """This `ServerApp` construct a histogram from partial-histograms reported by the
    `ClientApp`s."""

    num_rounds = context.run_config["num-server-rounds"]
    min_nodes = 2
    fraction_sample = context.run_config["fraction-sample"]

    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_nodes:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_nodes:
                # Sample nodes
                num_to_sample = int(len(all_node_ids) * fraction_sample)
                node_ids = random.sample(all_node_ids, num_to_sample)
                break
            log(INFO, "Waiting for nodes to connect...")
            time.sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(node_ids), len(all_node_ids))

        # Create messages
        recorddict = RecordDict()
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=recorddict,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
                group_id=str(server_round),
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Aggregate partial histograms
        aggregated_hist = aggregate_partial_histograms(replies)

        # Display aggregated histogram
        log(INFO, "Aggregated histogram: %s", aggregated_hist)


def aggregate_partial_histograms(messages: Iterable[Message]):
    """Aggregate partial histograms."""

    aggregated_hist = {}
    total_count = 0
    for rep in messages:
        if rep.has_error():
            continue
        query_results = rep.content["query_results"]
        # Sum metrics
        for k, v in query_results.items():
            if k in ["SepalLengthCm", "SepalWidthCm"]:
                if k in aggregated_hist:
                    aggregated_hist[k] += np.array(v)
                else:
                    aggregated_hist[k] = np.array(v)
            if "_count" in k:
                total_count += v

    # Verify aggregated histogram adds up to total reported count
    assert total_count == sum([sum(v) for v in aggregated_hist.values()])
    return aggregated_hist
