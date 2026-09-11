"""Placeholder ClientApp for C++ quickstart.

The actual clients are C++ SuperNodes that connect via gRPC.
This file satisfies the Flower app configuration requirement.
"""
import flwr as fl


def client_fn(context) -> fl.client.Client:
    """Placeholder client factory for the C++ quickstart.

    The actual clients for this example are implemented in C++ and connect to
    the server via gRPC. This Python client placeholder exists only to satisfy
    the Flower app configuration requirements and must not be started.
    """
    raise RuntimeError(
        "This Python ClientApp is a placeholder for the C++ quickstart. "
        "Use the C++ SuperNode clients instead of starting this client."
    )


app = fl.client.ClientApp(client_fn=client_fn)
