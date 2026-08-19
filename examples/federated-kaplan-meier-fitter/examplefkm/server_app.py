"""examplefkm: A Flower / Lifelines app."""

from logging import INFO
from time import sleep

import numpy as np
from flwr.app import Context, Message, MessageType, RecordDict
from flwr.common.logger import log
from flwr.serverapp import Grid, ServerApp
from lifelines import KaplanMeierFitter

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    min_num_clients = context.run_config["min-num-clients"]
    num_rounds = context.run_config["num-server-rounds"]

    # Define the filter
    fitter = KaplanMeierFitter()  # You can choose other method that work on E, T data

    # Define FL round
    for server_round in range(num_rounds):
        log(INFO, "")  # Add newline for log readability
        log(INFO, "Starting round %s/%s", server_round + 1, num_rounds)

        # Loop and wait until enough nodes are available.
        all_node_ids: list[int] = []
        while len(all_node_ids) < min_num_clients:
            all_node_ids = list(grid.get_node_ids())
            if len(all_node_ids) >= min_num_clients:
                break
            log(INFO, "Waiting for nodes to connect...")
            sleep(2)

        log(INFO, "Sampled %s nodes (out of %s)", len(all_node_ids), len(all_node_ids))

        # Create messages
        recorddict = RecordDict()
        messages = []
        for node_id in all_node_ids:  # one message for each node
            message = Message(
                content=recorddict,
                message_type=MessageType.QUERY,  # target `query` method in ClientApp
                dst_node_id=node_id,
            )
            messages.append(message)

        # Send messages and wait for all results
        replies = grid.send_and_receive(messages)
        log(INFO, "Received %s/%s results", len(replies), len(messages))

        # Apply the fitter
        apply_fitter(fitter, replies)


def apply_fitter(fitter: KaplanMeierFitter, replies: list[Message]) -> None:
    """Apply the lifelines fitter to the received data."""
    # Use lifelines filter to fit the data
    remote_times = []
    remote_events = []
    for reply in replies:
        if reply.has_error():
            continue

        # Both `T` and `E` are `Array` objects, let's convert them to numpy
        remote_times.append(reply.content["survival-data"]["T"].numpy())
        remote_events.append(reply.content["survival-data"]["E"].numpy())

    combined_times = remote_times[0]
    combined_events = remote_events[0]

    for t, e in zip(remote_times[1:], remote_events[1:]):
        combined_times = np.concatenate((combined_times, t))
        combined_events = np.concatenate((combined_events, e))

    args_sorted = np.argsort(combined_times)
    sorted_times = combined_times[args_sorted]
    sorted_events = combined_events[args_sorted]
    fitter.fit(sorted_times, sorted_events)
    print("Survival function:")
    print(fitter.survival_function_)
    print("Mean survival time:")
    print(fitter.median_survival_time_)
