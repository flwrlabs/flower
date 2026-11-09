"""Flower Server."""
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

from typing import List, Tuple
import shutil
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from .strategy import WeightedStrategy
import os


if os.path.exists("logs"):
    shutil.rmtree("logs")

# Create fresh 'logs/' folder
os.makedirs("logs")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
     for num_examples, m in metrics: print("Client",m["cid"]," recieved accuracy", m["accuracy"])
     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
     examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
     return {"accuracy": sum(accuracies) / sum(examples)}



def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {

        "current_round": server_round ,

    }
    return config

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    
    # Define strategy
    strategy = WeightedStrategy(
      fraction_fit=float(fraction_fit),  # 
      min_fit_clients=10,  # number of clients to sample for fit()
      fraction_evaluate=1,  # similar to fraction_fit, we don't need to use this argument.
      min_evaluate_clients=10,  # number of clients to sample for evaluate()
      min_available_clients=10,  # total clients in the simulation     
      evaluate_metrics_aggregation_fn=weighted_average, 
      on_fit_config_fn =fit_config,
      on_evaluate_config_fn=fit_config,
    
  )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)


