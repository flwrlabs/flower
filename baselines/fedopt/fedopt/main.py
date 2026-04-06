"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import pickle
from pathlib import Path

import flwr as fl
import hydra
from flwr.common import ndarrays_to_parameters
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from fedopt.client import gen_client_fn
from fedopt.dataset import get_dataloaders
from fedopt.strategy import gen_evaluate_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    train_loaders, test_loader = get_dataloaders(cfg.dataset)

    # 3. Define your clients
    client_fn = gen_client_fn(train_loaders, cfg.client)

    # 4. Define your strategy
    model = call(cfg.model)
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )
    strategy = instantiate(
        cfg.strategy,
        initial_parameters=initial_parameters,
        evaluate_fn=gen_evaluate_fn(test_loader, cfg.server_device, cfg.client),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = Path(HydraConfig.get().runtime.output_dir)

    # Save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    data = {"history": history}
    history_path = f"{str(save_path)}/history.pkl"
    with open(history_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO: make plots


if __name__ == "__main__":
    main()
