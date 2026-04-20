"""objecttest: A Flower / NumPy app."""

import numpy as np
from flwr.app import ArrayRecord, Context, Message, RecordDict, MessageType
from flwr.serverapp import Grid, ServerApp


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Load global model
    model = [np.ones((1, 1))]
    arrays = ArrayRecord(model)
    record = RecordDict({"arrays": arrays})
    
    # Get all SuperNode IDs in the Grid
    node_ids = grid.get_node_ids()

    # Construct one message for each SuperNode
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=record,
            message_type=MessageType.TRAIN,
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send TRAIN messages to all SuperNodes and wait for replies
    train_replies = grid.send_and_receive(
        messages=messages,
        timeout=30,
    )

    # Save final model to disk
    print("\nNumber of TRAIN replies received:", len(train_replies))
