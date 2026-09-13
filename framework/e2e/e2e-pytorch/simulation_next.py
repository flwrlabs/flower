from e2e_pytorch.client_app import app as client_app

import flwr as fl

# Define ServerAppp
server_app = fl.serverapp.ServerApp(
    config=fl.server.ServerConfig(num_rounds=2),
)


# Run with FlowerNext
fl.simulation.run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=2
)
