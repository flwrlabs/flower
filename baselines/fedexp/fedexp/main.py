"""Main script for running FedExp.
the code is inspired by the authors' code.
https://github.com/Divyansh03/FedExP/
"""

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedexp import client, server, utils
from fedexp.dataset import load_datasets
from fedexp.utils import get_parameters, seed_everything


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    train_loaders, test_loader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_equal=True,
    )

    data_ratios = np.zeros(cfg.num_clients)
    for i in range(cfg.num_clients):
        data_ratios[i] = len(train_loaders[i])
    data_ratios /= np.sum(data_ratios)

    client_fn = client.gen_client_fn(
        train_loaders=train_loaders,
        model=cfg.model,
        num_epochs=cfg.num_epochs,
        args={"data_ratio": data_ratios},
    )

    evaluate_fn = server.gen_evaluate_fn(
        test_loader=test_loader, model=cfg.model, device=cfg.server_device
    )

    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            fit_config = OmegaConf.to_container(cfg.hyperparams, resolve=True)
            fit_config["curr_round"] = server_round
            cfg.hyperparams.eta_l *= cfg.hyperparams.decay
            return fit_config

        return fit_config_fn

    net_glob = instantiate(cfg.model)

    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(net_glob)),
        net_glob=net_glob,
        epsilon=cfg.hyperparams.epsilon,
        decay=cfg.hyperparams.decay,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
    )

    save_path = HydraConfig.get().runtime.output_dir
    file_suffix = "_".join(
        [
            repr(strategy),
            cfg.dataset_config.name,
            f"{cfg.seed}",
            f"{cfg.dataset_config.alpha}",
            f"{cfg.num_clients}",
            f"{cfg.num_rounds}",
            f"{cfg.clients_per_round}",
            f"{cfg.hyperparams.eta_l}",
        ]
    )

    utils.plot_metric_from_history(
        hist=history, save_plot_path=save_path, suffix=file_suffix, cfg=cfg
    )


if __name__ == "__main__":
    main()
