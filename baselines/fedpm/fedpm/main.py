"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import logging

import flwr
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedpm.dataset import get_data_loaders


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
    if not cfg.verbose:
        logging.warn("Disabling logging will also disable creating the log file !")
        logging.warn("Disabling logging now.")
        logging.disable()

    # 2. Prepare your dataset
    trainloaders, valloaders, testloader = get_data_loaders(
        dataset=cfg.dataset.name,
        nclients=cfg.num_clients,
        batch_size=cfg.dataset.minibatch_size,
        classes_pc=cfg.dataset.classes_pc,
        split=cfg.dataset.split,
        data_path=cfg.dataset.data_path,
    )

    # 3. Define your clients
    def client_fn(cid) -> flwr.client.Client:
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return instantiate(
            cfg.client,
            client_id=cid,
            train_data_loader=trainloader,
            test_data_loader=valloader,
        )

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy, model_cfg=cfg.model, global_data_loader=testloader
    )

    # 5. Start Simulation
    # client_resources = {'num_cpus': 1, 'num_gpus': 1}
    flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=flwr.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
