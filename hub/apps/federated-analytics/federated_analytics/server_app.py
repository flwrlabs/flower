"""federated_analytics: A Flower / Federated Analytics app."""

import json
import random
import time
from logging import INFO

from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp

from federated_analytics.task import aggregate_features

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """This `ServerApp` construct a histogram from partial-histograms reported by the
    `ClientApp`s."""

    min_nodes = 1
    fraction_sample = context.run_config["fraction-sample"]
    selected_features = str(context.run_config["selected-features"]).split(",")
    feature_aggregation = str(context.run_config["feature-aggregation"]).split(",")

    log(INFO, "")  # Add newline for log readability

    log(INFO, "=" * 60)
    log(INFO, "FEDERATED ANALYTICS CONFIGURATION".center(60))
    log(INFO, "=" * 60)
    log(INFO, "Selected features:")
    for i, feature in enumerate(selected_features, 1):
        log(INFO, "  %d. %s", i, feature.strip())
    log(INFO, "Feature aggregation methods: %s", ", ".join(feature_aggregation))
    log(INFO, "=" * 60)

    log(INFO, "")  # Add newline for log readability

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
    config = ConfigRecord(
        {
            "selected_features": selected_features,
            "feature_aggregation": feature_aggregation,
        }
    )
    recorddict = RecordDict({"config": config})
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=recorddict,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    log(INFO, "Received %s/%s results", len(list(replies)), len(messages))

    aggregated_stats = aggregate_features(
        replies, selected_features, feature_aggregation
    )

    log(INFO, "Saving aggregated statistics to 'aggregated_stats.json'...")
    # Save aggregated stats to JSON file
    with open("aggregated_stats.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_stats, f, indent=2)

    # Display final aggregated stats
    print("\n" + "=" * 40)
    print("FINAL AGGREGATED STATISTICS".center(40))
    print("=" * 40)
    print(json.dumps(aggregated_stats, indent=2))
    print("=" * 40 + "\n")
