"""examplefkm: A Flower / Lifelines app."""

from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from examplefkm.task import load_partition

# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):
    """Query time to event data."""

    # Load the data
    partition_id = context.node_config["partition-id"]
    # Times and events
    #   Times: times of the events
    #   Events: 0 - no event, 1 - event occurred
    times, events = load_partition(partition_id)

    if len(times) != len(events):
        raise ValueError("The times and events arrays have to be same shape.")

    # Construct and return reply Message
    model_record = ArrayRecord({"T": Array(times), "E": Array(events)})
    metrics = {
        "num-examples": len(times),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"survival-data": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
