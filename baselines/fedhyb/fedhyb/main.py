
"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import flwr as fl
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

from .client_app import generate_client_fn
from .dataset_preparation import prepare_dataset
from .server_app import weighted_average, fit_config
from .strategy import WeightedStrategy
from hydra.core.hydra_config import HydraConfig
import sys
# pylint: disable=too-many-locals
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    np.random.seed(2020)

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir
    # 2. Prepare your dataset
    trainloaders, valloaders, num_classes_cl, num_features= prepare_dataset(cfg.path_to_dataset,
       cfg.num_clients, cfg.batch_size, cfg.val_ratio , cfg.num_drop, cfg.total_classes    )
    

   

    
    
    # 3. Define your clients
    client_fn = generate_client_fn(trainloaders, valloaders, num_features,cfg.total_classes, num_classes_cl,cfg.local_epochs, cfg.lr, cfg.print_round)

    # 4. Define your strategy
    strategy = WeightedStrategy(
      fraction_fit=1,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
      min_fit_clients=10,  # number of clients to sample for fit()
      fraction_evaluate=1,  # similar to fraction_fit, we don't need to use this argument.
      min_evaluate_clients=10,  # number of clients to sample for evaluate()
      min_available_clients=10,  # total clients in the simulation
        # a function to execute to obtain the configuration to send to the clients during fit()
      #evaluate_fn=get_evaluate_fn(10, testloader),
      evaluate_metrics_aggregation_fn=weighted_average, 
      on_fit_config_fn =fit_config,
      on_evaluate_config_fn=fit_config,
     # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
  )
  
    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )
    
   ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

