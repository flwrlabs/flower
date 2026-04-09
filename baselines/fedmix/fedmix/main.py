"""..."""
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fedmix import client, server
from fedmix.dataset import load_datasets
from fedmix.utilities import save_results_as_pickle


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    """..."""
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.seed
    server_device = cfg.server_device
    client_resources = cfg.client_resources
    num_clients = cfg.num_clients
    num_rounds = cfg.num_rounds

    dataset_config = cfg.dataset
    model_config = cfg.model
    strategy_config = cfg.strategy
    client_config = cfg.client

    trainloaders, testloader, mashed_data = load_datasets(
        dataset_config, num_clients, seed
    )

    client_fn = client.gen_client_fn(client_config, trainloaders, model_config)
    evaluate_fn = server.gen_evaluate_fn(testloader, server_device, model_config)

    strategy = instantiate(
        strategy_config, mashed_data=mashed_data, evaluate_fn=evaluate_fn
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        client_resources=client_resources,
        strategy=strategy,
    )

    print(OmegaConf.to_yaml(cfg))

    # print("----------------------")
    # print(history)
    # print("----------------------")

    save_path = HydraConfig.get().runtime.output_dir
    save_results_as_pickle(history, file_path=save_path, extra_results={})


if __name__ == "__main__":
    main()
