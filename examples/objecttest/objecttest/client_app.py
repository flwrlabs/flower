"""objecttest: A Flower / NumPy app."""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    return Message(content=msg.content, reply_to=msg)
