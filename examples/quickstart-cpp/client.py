"""Placeholder ClientApp for C++ quickstart.

The actual clients are C++ SuperNodes that connect via gRPC.
This file satisfies the Flower app configuration requirement.
"""
import flwr as fl

app = fl.client.ClientApp(client_fn=lambda context: fl.client.NumPyClient().to_client())
