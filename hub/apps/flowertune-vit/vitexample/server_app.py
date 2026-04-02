"""vitexample: A Flower / PyTorch app with Vision Transformers."""

import torch
from datasets import Dataset, load_dataset
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from torch.utils.data import DataLoader

from vitexample.task import apply_eval_transforms, get_model, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Define tested for central evaluation
    dataset_name = context.run_config["dataset-name"]
    dataset = load_dataset(dataset_name)
    test_set = dataset["test"]
    num_rounds = context.run_config["num-server-rounds"]

    # Set initial global model
    num_classes = context.run_config["num-classes"]
    model = get_model(num_classes)
    finetune_layers = model.heads
    arrays = ArrayRecord(finetune_layers.state_dict())

    # Configure the strategy
    strategy = FedAvg(
        fraction_train=0.5,  # Sample 50% of available clients
        fraction_evaluate=0.0,  # No federated evaluation
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            test_set, num_classes
        ),  # Global evaluation function
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def get_evaluate_fn(
    centralized_testset: Dataset,
    num_classes: int,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate model and apply current global parameters
        model = get_model(num_classes)
        finetune_layers = model.heads
        finetune_layers.load_state_dict(arrays.to_torch_state_dict(), strict=True)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)

        # Return the evaluation metrics
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return evaluate
