"""tracefl-baseline: A Flower Baseline."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from tracefl.config import create_tracefl_config
from tracefl.dataset import get_clients_server_data
from tracefl.dp_wrapper import TraceFLWithDP
from tracefl.model import initialize_model
from tracefl.strategy import TraceFLStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create ServerApp
app = ServerApp()

# Global variables for caching server data across strategy initialization
# _SERVER_DATA: Test dataset used for provenance analysis (loaded once)
# _CLIENT2CLASS: Mapping {client_id: {label: count}} for localization accuracy
# These are cached to avoid redundant loading when strategy is initialized
_SERVER_DATA = None
_CLIENT2CLASS = None


def _load_server_data(context):
    """Load server test data and client label mappings using TraceFL preparation."""
    global _SERVER_DATA, _CLIENT2CLASS  # pylint: disable=global-statement
    if _SERVER_DATA is None or _CLIENT2CLASS is None:
        # Create TraceFL config from Flower context
        cfg = create_tracefl_config(context)

        # Load dataset
        ds_dict = get_clients_server_data(cfg)

        # Store server test data
        _SERVER_DATA = ds_dict["server_testdata"]
        _CLIENT2CLASS = ds_dict.get("client2class", {})

        print(f"Server loaded {len(_SERVER_DATA)} test samples")

    return _SERVER_DATA, _CLIENT2CLASS


def _set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Parameters
    ----------
    seed : int
        Random seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For CuDNN backend (if using CUDA)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run main entry point for the ServerApp."""
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])

    # Handle reproducibility configuration
    use_deterministic = context.run_config.get(
        "tracefl.use-deterministic-sampling", "true"
    )
    # Convert string to boolean
    use_deterministic = str(use_deterministic).lower() == "true"

    if use_deterministic:
        # Deterministic mode: Fix random seed for reproducibility
        # This ensures the same clients are selected in each round
        # Critical for comparing results across runs (e.g., original TraceFL vs
        # baseline)
        seed = int(context.run_config.get("tracefl.random-seed", 42))
        _set_random_seed(seed)
        logging.info("Deterministic mode enabled with seed: %s", seed)

        # Note: fraction_train still applies, but min_train_nodes ensures
        # at least N clients participate (prevents empty training rounds)
    else:
        # Random mode: Clients are sampled randomly each round
        # Useful for testing generalization across different client sets
        logging.info("Random mode enabled (non-deterministic)")

    # Get minimum training nodes (for deterministic mode)
    min_train_nodes = int(context.run_config.get("min-train-nodes", 2))

    # Load TraceFL server data
    server_data, client2class = _load_server_data(context)

    # Get TraceFL config
    cfg = create_tracefl_config(context)

    # Log architecture detection results
    print("🔧 Architecture Detection:")
    print(f"   Dataset: {cfg.data_dist.dname} ({cfg.data_dist.architecture})")
    print(f"   Model: {cfg.data_dist.model_name} ({cfg.data_dist.model_architecture})")
    print(
        f"   Classes: {cfg.data_dist.num_classes}, "
        f"Channels: {cfg.data_dist.channels}"
    )
    print(
        f"   Compatibility: {cfg.data_dist.model_architecture} + "
        f"{cfg.data_dist.architecture}"
    )

    # Parse output directory from config
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load global model
    # Initialize model based on configuration
    model_dict = initialize_model(cfg.data_dist.model_name, cfg.data_dist)
    global_model = model_dict["model"]
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize TraceFL strategy
    strategy: TraceFLStrategy | TraceFLWithDP = TraceFLStrategy(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_train_nodes=min_train_nodes,
        min_available_nodes=min_train_nodes if use_deterministic else 2,
        provenance_rounds=cfg.provenance_rounds,
        enable_beta=cfg.enable_beta,
        client_weights_normalization=cfg.client_weights_normalization,
        cfg=cfg,
        output_dir=output_dir,
    )

    # CRITICAL: Set server test data BEFORE applying DP wrapper
    # The DP wrapper delegates attribute access to the underlying strategy,
    # so these methods must be called on the base TraceFLStrategy instance
    # before it gets wrapped. Calling them after wrapping would fail.
    strategy.set_server_test_data(server_data)
    strategy.set_client2class(client2class)

    # Apply custom differential privacy wrapper if enabled
    if cfg.noise_multiplier > 0 and cfg.clipping_norm > 0:
        logging.info(">> Running DP FL")

        # Wrap TraceFLStrategy with DP (matching original TraceFL implementation)
        dp_strategy = TraceFLWithDP(
            strategy,
            noise_multiplier=cfg.noise_multiplier,
            clipping_norm=cfg.clipping_norm,
            num_sampled_clients=min_train_nodes,
        )

        strategy = dp_strategy

        logging.info(
            ">> Custom DP wrapper applied: noise_multiplier=%s, clipping_norm=%s",
            cfg.noise_multiplier,
            cfg.clipping_norm,
        )
        logging.info(
            ">> Provenance analysis ENABLED with differential privacy protection"
        )
    else:
        logging.info(">> Running Non-DP FL")

    # Start strategy, run TraceFL for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    model_path = output_dir / "final_model.pt"
    torch.save(state_dict, model_path)
    print(f"Model saved to: {model_path}")

    # Print TraceFL configuration info
    print("\n🎯 TraceFL Configuration:")
    print(f"📊 Dataset: {cfg.data_dist.dname}")
    print(f"👥 Clients: {cfg.data_dist.num_clients}")
    print(f"📈 Distribution: {cfg.data_dist.dist_type}")
    print(f"🔢 Dirichlet Alpha: {cfg.data_dist.dirichlet_alpha}")
    print(f"💾 Max per client: {cfg.data_dist.max_per_client_data_size}")
    print(f"🖥️  Server test data: {len(server_data)} samples")
    print(f"🔍 Provenance rounds: {cfg.provenance_rounds}")
    print(f"⚡ Enable beta (layer importance): {cfg.enable_beta}")
    print(f"📏 Client weights normalization: {cfg.client_weights_normalization}")
    print(f"🛡️  Noise multiplier: {cfg.noise_multiplier}")
    print(f"✂️  Clipping norm: {cfg.clipping_norm}")
    print(f"⚠️ Faulty clients: {cfg.faulty_clients_ids}")
    print(f"🔁 Label flip map: {cfg.label2flip}")
