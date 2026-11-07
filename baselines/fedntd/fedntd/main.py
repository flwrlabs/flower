"""Runs CNN federated learning for MNIST dataset."""

from typing import Dict

import flwr as fl
import hydra
from datasets.utils.logging import disable_progress_bar
from flwr.common import Scalar
from flwr_datasets import FederatedDataset
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from fedntd.client import get_client_fn
from fedntd.server import get_evaluate_fn
from fedntd.utils import save_results_as_pickle, plot_metric_from_history


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    NUM_CLIENTS = 2

    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_full("test")

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config: Dict[str, Scalar] = {
            "epochs": 1,
            "lr": 0.01,
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    disable_progress_bar()

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )

    print("................")
    print(history)

    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    # strategy_name = strategy.__class__.__name__
    # file_suffix: str = (
    #     f"_{strategy_name}"
    #     f"{'_iid' if cfg.dataset_config.iid else ''}"
    #     f"{'_balanced' if cfg.dataset_config.balance else ''}"
    #     f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
    #     f"_C={cfg.num_clients}"
    #     f"_B={cfg.batch_size}"
    #     f"_E={cfg.num_epochs}"
    #     f"_R={cfg.num_rounds}"
    #     f"_mu={cfg.mu}"
    #     f"_strag={cfg.stragglers_fraction}"
    # )

    plot_metric_from_history(
        history,
        save_path,
        # (file_suffix),
    )


if __name__ == "__main__":
    main()
