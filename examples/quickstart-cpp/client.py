"""Placeholder ClientApp for the C++ quickstart.

The actual clients in this example are external C++ processes. Flower App
metadata still requires a ClientApp entry point in recent Flower versions, so
this module exists to make that boundary explicit.
"""

import flwr as fl
from flwr.common import Context


def client_fn(context: Context):
    raise RuntimeError(
        "This quickstart uses external C++ clients. Start build/flwr_client "
        "processes instead of executing the Python ClientApp."
    )


app = fl.client.ClientApp(client_fn=client_fn)
