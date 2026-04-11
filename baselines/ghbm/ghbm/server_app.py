"""ServerApp for the GHBM Flower baseline."""

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from ghbm.config import AlgorithmName, DatasetName, ModelName, NormLayer, parse_enum
from ghbm.dataset import get_num_classes
from ghbm.model import create_model
from ghbm.strategy import EvalLoggingFedAvg, GHBMStrategy

# Create ServerApp
app = ServerApp()


def _as_bool(value: object) -> bool:
    """Convert run-config booleans that may arrive as strings."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _metric_record_to_dict(record: object) -> dict[str, float]:
    """Convert Flower metric containers into plain numeric dicts."""
    items: Iterable[tuple[object, object]]
    if isinstance(record, MetricRecord):
        items = record.items()
    elif isinstance(record, Mapping):
        items = record.items()
    else:
        items = ()
    metrics = {}
    for key, value in items:
        if isinstance(value, int | float):
            metrics[str(key)] = float(value)
    return metrics


def _maybe_init_wandb(context: Context, num_rounds: int, ghbm_tau: int):
    """Initialize a W&B run when requested through run config."""
    if not _as_bool(context.run_config["wandb-enabled"]):
        return None

    import wandb  # pylint: disable=import-outside-toplevel

    dataset_name = parse_enum(context.run_config["dataset-name"], DatasetName)
    algorithm_name = parse_enum(context.run_config["algorithm-name"], AlgorithmName)
    model_name = parse_enum(context.run_config["model-name"], ModelName)
    fraction_train = float(context.run_config["fraction-train"])
    dirichlet_alpha = float(context.run_config["dirichlet-alpha"])
    default_name = (
        f"{algorithm_name}-{dataset_name}-alpha{dirichlet_alpha}-"
        f"C{fraction_train}-R{num_rounds}"
    )
    run_name = str(context.run_config["wandb-run-name"]).strip() or default_name

    return wandb.init(
        project=str(context.run_config["wandb-project"]),
        group=str(context.run_config["wandb-group"]),
        name=run_name,
        settings=wandb.Settings(start_method="fork", init_timeout=300),
        config={
            "algorithm_name": algorithm_name,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "resnet_version": int(context.run_config["resnet-version"]),
            "norm_layer": str(parse_enum(context.run_config["norm-layer"], NormLayer)),
            "num_server_rounds": num_rounds,
            "fraction_train": fraction_train,
            "local_epochs": int(context.run_config["local-epochs"]),
            "learning_rate": float(context.run_config["learning-rate"]),
            "weight_decay": float(context.run_config["weight-decay"]),
            "ghbm_beta": float(context.run_config["ghbm-beta"]),
            "ghbm_tau": ghbm_tau,
            "dirichlet_alpha": dirichlet_alpha,
            "batch_size": int(context.run_config["batch-size"]),
        },
    )


def _maybe_get_wandb_round_logger(context: Context, num_rounds: int, ghbm_tau: int):
    """Create a per-round W&B logger callback when enabled."""
    try:
        run = _maybe_init_wandb(context, num_rounds, ghbm_tau)
    except Exception as err:  # pylint: disable=broad-except
        print(f"W&B init failed: {err}")
        return None, None

    if run is None:
        return None, None

    def log_round(phase: str, server_round: int, metrics: object) -> None:
        metric_dict = _metric_record_to_dict(metrics)
        if not metric_dict:
            return
        payload: dict[str, int | float] = {"round": server_round}
        payload.update({f"{phase}/{key}": value for key, value in metric_dict.items()})
        run.log(payload, step=server_round)

    return run, log_round


def _print_run_summary(context: Context, num_rounds: int, ghbm_tau: int) -> None:
    """Print a compact summary of the configured experiment."""
    algorithm_name = parse_enum(context.run_config["algorithm-name"], AlgorithmName)
    dataset_name = parse_enum(context.run_config["dataset-name"], DatasetName)
    model_name = parse_enum(context.run_config["model-name"], ModelName)
    resnet_version = int(context.run_config["resnet-version"])
    norm_layer = parse_enum(context.run_config["norm-layer"], NormLayer)
    fraction_train = float(context.run_config["fraction-train"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    weight_decay = float(context.run_config["weight-decay"])
    ghbm_beta = float(context.run_config["ghbm-beta"])
    dirichlet_alpha = float(context.run_config["dirichlet-alpha"])
    batch_size = int(context.run_config["batch-size"])
    evaluate_every = int(context.run_config["evaluate-every"])
    wandb_enabled = _as_bool(context.run_config["wandb-enabled"])

    print("\n=== Run Summary ===")
    print(f"algorithm-name: {algorithm_name}")
    print(f"dataset-name: {dataset_name}")
    print(f"model-name: {model_name}")
    if model_name is ModelName.RESNET:
        print(f"resnet-version: {resnet_version}")
        print(f"norm-layer: {norm_layer}")
    print(f"num-server-rounds: {num_rounds}")
    print(f"fraction-train: {fraction_train}")
    print(f"local-epochs: {local_epochs}")
    print(f"batch-size: {batch_size}")
    print(f"learning-rate: {learning_rate}")
    print(f"weight-decay: {weight_decay}")
    print(f"ghbm-beta: {ghbm_beta}")
    print(f"ghbm-tau: {ghbm_tau}")
    print(f"dirichlet-alpha: {dirichlet_alpha}")
    print(f"evaluate-every: {evaluate_every}")
    print(f"wandb-enabled: {wandb_enabled}")
    if wandb_enabled:
        print(f"wandb-project: {context.run_config['wandb-project']}")
        print(f"wandb-group: {context.run_config['wandb-group']}")
        print(f"wandb-run-name: {context.run_config['wandb-run-name']}")
    print("===================\n")


def _summarize_final_evaluation(
    result: Any, window_size: int = 100
) -> dict[str, float]:
    """Average evaluation metrics over the last `window_size` evaluated rounds."""
    eval_history = result.evaluate_metrics_clientapp
    if not eval_history:
        return {}

    eval_rounds = sorted(eval_history.keys())
    selected_rounds = eval_rounds[-window_size:]
    eval_losses: list[float] = []
    eval_accs: list[float] = []
    for server_round in selected_rounds:
        metrics = _metric_record_to_dict(eval_history[server_round])
        if "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])
        if "eval_acc" in metrics:
            eval_accs.append(metrics["eval_acc"])

    summary: dict[str, float] = {
        "num_eval_rounds_averaged": float(len(selected_rounds))
    }
    if eval_losses:
        summary["avg_eval_loss_last_window"] = sum(eval_losses) / len(eval_losses)
    if eval_accs:
        summary["avg_eval_acc_last_window"] = sum(eval_accs) / len(eval_accs)
    return summary


def _print_final_evaluation_summary(summary: Mapping[str, float]) -> None:
    """Print the final averaged evaluation summary."""
    if not summary:
        print("\nNo evaluation rounds were recorded.")
        return

    print("\n=== Final Evaluation Summary ===")
    print(
        "averaging-window:"
        f" {int(summary['num_eval_rounds_averaged'])} evaluated rounds"
    )
    if "avg_eval_loss_last_window" in summary:
        print(f"avg-eval-loss: {summary['avg_eval_loss_last_window']:.6f}")
    if "avg_eval_acc_last_window" in summary:
        print(f"avg-eval-acc: {summary['avg_eval_acc_last_window']:.6f}")
    print("================================\n")


def _log_final_evaluation_summary(run: Any, summary: Mapping[str, float]) -> None:
    """Store the final averaged evaluation summary in W&B when enabled."""
    if run is None or not summary:
        return

    for key, value in summary.items():
        run.summary[key] = value


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Read from config
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])
    algorithm_name = parse_enum(context.run_config["algorithm-name"], AlgorithmName)
    dataset_name = parse_enum(context.run_config["dataset-name"], DatasetName)
    model_name = parse_enum(context.run_config["model-name"], ModelName)
    resnet_version = int(context.run_config["resnet-version"])
    norm_layer = parse_enum(context.run_config["norm-layer"], NormLayer)
    configured_tau = int(context.run_config["ghbm-tau"])
    evaluate_every = int(context.run_config["evaluate-every"])
    final_eval_window = 100
    ghbm_tau = configured_tau or max(1, round(1.0 / fraction_train))
    _print_run_summary(context, num_rounds, ghbm_tau)
    wandb_run, wandb_round_logger = _maybe_get_wandb_round_logger(
        context, num_rounds, ghbm_tau
    )

    # Load global model
    global_model = create_model(
        model_name=model_name,
        num_classes=get_num_classes(dataset_name),
        resnet_version=resnet_version,
        norm_layer=norm_layer,
    )
    arrays = ArrayRecord(torch_state_dict=global_model.state_dict())

    strategy_kwargs = {
        "fraction_train": fraction_train,
        "fraction_evaluate": 1.0,
        "min_available_nodes": 2,
    }
    strategy: GHBMStrategy | EvalLoggingFedAvg
    if algorithm_name is AlgorithmName.GHBM:
        strategy = GHBMStrategy(
            tau=ghbm_tau,
            num_rounds=num_rounds,
            evaluate_every=evaluate_every,
            final_eval_window=final_eval_window,
            round_logger=wandb_round_logger,
            **strategy_kwargs,
        )
    elif algorithm_name in (
        AlgorithmName.FEDAVG,
        AlgorithmName.LOCAL_GHBM,
        AlgorithmName.FED_HBM,
    ):
        strategy = EvalLoggingFedAvg(
            num_rounds=num_rounds,
            evaluate_every=evaluate_every,
            final_eval_window=final_eval_window,
            round_logger=wandb_round_logger,
            **strategy_kwargs,
        )
    else:
        raise ValueError(f"Unsupported algorithm-name: {algorithm_name}")

    # Start strategy, run GHBM for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    final_eval_summary = _summarize_final_evaluation(
        result, window_size=final_eval_window
    )
    _print_final_evaluation_summary(final_eval_summary)
    _log_final_evaluation_summary(wandb_run, final_eval_summary)
    if wandb_run is not None:
        wandb_run.finish()

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
