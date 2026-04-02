"""pytorch-example: A Flower / PyTorch app."""

import torch
from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from torch.utils.data import DataLoader

from pytorch_example.strategy import CustomFedAvg
from pytorch_example.task import Net, apply_eval_transforms, create_run_dir, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]
    device = context.run_config["server-device"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = CustomFedAvg(
        fraction_train=fraction_train, fraction_evaluate=fraction_eval
    )

    # Define directory for results and save config
    save_path, run_dir = create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.1}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(device=device),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_global_evaluate_fn(device: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # This is the exact same dataset as the one downloaded by the clients via
        # FlowerDatasets. However, we don't use FlowerDatasets for the server since
        # partitioning is not needed.
        # We make use of the "test" split only
        global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]

        testloader = DataLoader(
            global_test_set.with_transform(apply_eval_transforms),
            batch_size=32,
        )

        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
