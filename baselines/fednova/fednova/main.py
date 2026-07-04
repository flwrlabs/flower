"""Entry script for Federated training on CIFAR-10 using FedNova."""

import os
import random
import time
from collections import OrderedDict
from functools import partial

import flwr as fl
import hydra
import numpy as np
import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from fednova.dataset import load_datasets
from fednova.models import test
from fednova.strategy import weighted_average
from fednova.utils import fit_config


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    start = time.time()

    # Set seeds for reproduceability
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    # torch.backends.cudnn.deterministic = True

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset and directories

    if not os.path.exists(cfg.datapath):
        os.makedirs(cfg.datapath)
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    trainloaders, testloader, data_sizes = load_datasets(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.mode == "test":
        checkpoint = np.load(
            f"{cfg.checkpoint_path}bestModel_{cfg.exp_name}_"
            f"varEpochs_{cfg.var_local_epochs}.npz",
            allow_pickle=True,
        )
        model = instantiate(cfg.model)
        params_dict = zip(model.state_dict().keys(), checkpoint["arr_0"])
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        model.load_state_dict(state_dict)
        loss, metrics = test(model.to(device), testloader, device)
        print(f"----Loss: {loss}, Accuracy: {metrics['accuracy']} on Test set ------")
        return None

    # 3. Define your clients
    client_fn = call(
        cfg.strategy.client_fn,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        testloader=testloader,
        data_sizes=data_sizes,
        model=cfg.model,
        exp_config=cfg,
    )

    # 4. Define your strategy

    ndarrays = [
        layer_param.cpu().numpy()
        for _, layer_param in instantiate(cfg.model).state_dict().items()
    ]
    init_parameters = ndarrays_to_parameters(ndarrays)

    eval_fn = partial(test, instantiate(cfg.model), testloader, device)
    fit_config_fn = partial(fit_config, cfg)

    extra_args = {} if cfg.strategy.name == "fedavg" else {"exp_config": cfg}
    strategy = instantiate(
        cfg.strategy.strategy,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=False,
        on_fit_config_fn=fit_config_fn,
        initial_parameters=init_parameters,
        evaluate_fn=eval_fn,
        fraction_evaluate=0.0,
        **extra_args,
    )

    # 5. Start Simulation

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
    )
    # in case if you are running experiment on a slurm cluster, it could be helpful
    # to add : ray_init_args={"ignore_reinit_error": True, "num_cpus": 8} as an extra
    # argument in above start_simulation(). This prevents the simulation from using
    # all cores and limits them to 8.

    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir

    # rounds, train_loss = zip(*history.losses_distributed)
    # _, train_accuracy = zip(*history.metrics_distributed["accuracy"])
    rounds, test_loss = zip(*history.losses_centralized)
    _, test_accuracy = zip(*history.metrics_centralized["accuracy"])

    if len(rounds) != cfg.num_rounds:
        # drop zeroth evaluation round before start of training
        test_loss = test_loss[1:]
        test_accuracy = test_accuracy[1:]
        rounds = rounds[1:]

    file_name = os.path.join(
        save_path,
        f"{cfg.exp_name}_{cfg.strategy.name}_varEpoch_"
        f"{cfg.var_local_epochs}_seed_{cfg.seed}.csv",
    )

    # df = pd.DataFrame({"round": rounds, "train_loss": train_loss, "train_accuracy":
    # train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy})

    df = pd.DataFrame(
        {"round": rounds, "test_loss": test_loss, "test_accuracy": test_accuracy}
    )

    df.to_csv(file_name, index=False)

    print(f"---------Experiment Completed in : {(time.time()-start)/60} minutes")

    return None


if __name__ == "__main__":
    main()
