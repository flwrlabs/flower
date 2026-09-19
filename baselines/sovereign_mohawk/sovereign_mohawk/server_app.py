"""sovereign_mohawk: A Flower Baseline starter."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from sovereign_mohawk.model import Net
from sovereign_mohawk.strategy import verification_report_to_dict, verify_arrayrecord

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Read from config
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])
    verification_hooks: bool = bool(context.run_config["enable-verification-hooks"])
    print(f"verification hooks enabled: {verification_hooks}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    verification_report = verify_arrayrecord(
        result.arrays,
        enabled=verification_hooks,
    )
    verification_payload = verification_report_to_dict(verification_report)
    print(json.dumps({"verification_report": verification_payload}, indent=2))

    Path("verification_report.json").write_text(
        json.dumps(verification_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    if verification_report.status == "failed":
        raise ValueError("verification hook failed due to non-finite tensor values")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
