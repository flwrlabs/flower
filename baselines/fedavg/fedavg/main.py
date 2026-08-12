"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedavg.client import gen_client_fn
from fedavg.dataset import load_datasets
from fedavg.strategy import gen_evaluate_fn, weighted_average
from fedavg.utils import plot_metric_from_history


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
    trainloaders, valloaders, testloader = load_datasets(cfg.dataset)

    # 3. Define your clients
    client_fn = gen_client_fn(trainloaders, valloaders, cfg.client)

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        min_available_clients=cfg.num_clients,
        evaluate_fn=gen_evaluate_fn(testloader, cfg.server_device, cfg.client),
        evaluate_metrics_aggregation_fn=weighted_average,
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

    # 6. Save your results
    file_suffix: str = (
        f"{'_iid' if cfg.dataset.iid else ''}"
        f"{'_balanced' if cfg.dataset.balance else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.dataset.batch_size}"
        f"_E={cfg.client.train_fn.epochs}"
        f"_R={cfg.num_rounds}"
    )

    plot_metric_from_history(
        history,
        save_path,
        cfg.expected_maximum,
        file_suffix,
    )


if __name__ == "__main__":
    main()
