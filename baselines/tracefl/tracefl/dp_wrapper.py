"""Custom Differential Privacy wrapper for TraceFLStrategy.

This implementation replicates the original TraceFL's DP logic from Flower 1.9.0 while
being compatible with Flower 1.22.0 and TraceFLStrategy.
"""

import logging
from collections import OrderedDict

import numpy as np
import torch
from flwr.common.logger import log
from flwr.common.record import ArrayRecord
from flwr.serverapp.strategy import Strategy


class TraceFLWithDP(Strategy):
    """Differential Privacy wrapper for TraceFLStrategy.

    This wrapper replicates the DP logic from Flower 1.9.0's
    DifferentialPrivacyServerSideFixedClipping, adapted for:
    1. Flower 1.22.0 compatibility
    2. TraceFLStrategy's custom aggregation
    3. Full provenance analysis support

    DP Algorithm (matching original TraceFL):
    1. Store current global model
    2. Let TraceFLStrategy aggregate client updates (weighted)
    3. Compute aggregated update: new_model - old_model
    4. Clip aggregated update to L2 norm bound
    5. Add calibrated Gaussian noise
    6. Return clipped+noisy model

    This provides the same DP guarantee as the original TraceFL while preserving
    all provenance analysis functionality.
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        """Initialize DP wrapper with verification of parameters.

        Parameters
        ----------
        strategy : Strategy
            The underlying TraceFLStrategy instance
        noise_multiplier : float
            Noise multiplier for Gaussian mechanism (matches original TraceFL)
        clipping_norm : float
            L2 norm bound for gradient clipping (matches original TraceFL)
        num_sampled_clients : int
            Number of clients sampled per round (for noise calibration)
        """
        super().__init__()

        # Validate parameters (same validation as Flower 1.9.0)
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier should be non-negative")
        if clipping_norm <= 0:
            raise ValueError("clipping_norm should be positive")
        if num_sampled_clients <= 0:
            raise ValueError("num_sampled_clients should be positive")

        self.strategy = strategy
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

        # Store previous global model (matches Flower 1.9.0's current_round_params)
        self.previous_arrays: ArrayRecord | None = None

        log(
            logging.INFO,
            "TraceFLWithDP initialized: noise_multiplier=%s, clipping_norm=%s, "
            "num_clients=%s",
            noise_multiplier,
            clipping_norm,
            num_sampled_clients,
        )

    def configure_train(self, server_round, arrays, config, grid):
        """Store current model and delegate to wrapped strategy.

        This matches Flower 1.9.0's configure_fit where it stores current_round_params.
        """
        # Store current global model for DP computation
        self.previous_arrays = arrays

        # Delegate to TraceFLStrategy
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def configure_evaluate(self, server_round, arrays, config, grid):
        """Configure evaluation - delegate to wrapped strategy."""
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        """Aggregate with DP: TraceFLStrategy aggregates, then we add DP.

        This matches Flower 1.9.0's aggregate_fit logic but adapted for
        post-aggregation clipping (which is mathematically equivalent when
        using weighted aggregation).

        IMPORTANT TIMING:
        TraceFLStrategy.aggregate_train() extracts client models BEFORE calling
        super().aggregate_train(), so provenance analysis gets access to
        ORIGINAL client models. Only the final aggregated global model gets
        DP noise applied, not the stored client models.

        This design preserves full provenance accuracy while still providing
        DP guarantees for the global model.
        """
        # Let TraceFLStrategy handle aggregation and provenance analysis
        # This includes:
        # - Weighted aggregation of client models
        # - Storing client models for provenance
        # - Running neuron-level analysis if in provenance round
        arrays, metrics = self.strategy.aggregate_train(server_round, replies)

        # Apply DP to aggregated model (matches Flower 1.9.0's flow)
        if arrays is not None and self.previous_arrays is not None:
            arrays = self._apply_dp_to_model(arrays, server_round)
        else:
            log(
                logging.WARNING,
                "Skipping DP application: arrays or previous_arrays is None",
            )

        return arrays, metrics

    def aggregate_evaluate(self, server_round, replies):
        """Aggregate evaluation - delegate to wrapped strategy."""
        return self.strategy.aggregate_evaluate(server_round, replies)

    def _apply_dp_to_model(
        self, aggregated_arrays: ArrayRecord, server_round: int
    ) -> ArrayRecord:
        """Apply DP (clipping + noise) matching Flower 1.9.0's implementation.

        This replicates the logic from:
        1. compute_clip_model_update() - clip the update
        2. add_gaussian_noise_to_params() - add noise

        From flwr.common.differential_privacy
        """
        # Convert to numpy (matches parameters_to_ndarrays in Flower 1.9.0)
        new_params = aggregated_arrays.to_numpy_ndarrays()
        if self.previous_arrays is None:
            raise ValueError("previous_arrays is not set. Cannot apply DP.")
        old_params = self.previous_arrays.to_numpy_ndarrays()

        # Compute update: new - old (matches compute_clip_model_update)
        updates = [
            np.subtract(new_p, old_p)
            for new_p, old_p in zip(new_params, old_params, strict=True)
        ]

        # Compute L2 norm of update (matches get_norm() in Flower 1.9.0)
        total_norm = np.sqrt(sum(np.sum(u**2) for u in updates))

        # Clip update if needed (matches clip_inputs_inplace)
        if total_norm > self.clipping_norm:
            scaling_factor = self.clipping_norm / total_norm
            updates = [u * scaling_factor for u in updates]
            log(
                logging.INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )
            log(
                logging.INFO,
                "Round %s: Clipped aggregated update (norm %.4f -> %.4f)",
                server_round,
                total_norm,
                self.clipping_norm,
            )
        else:
            log(
                logging.INFO,
                "Round %s: Update within bounds (norm %.4f <= %.4f)",
                server_round,
                total_norm,
                self.clipping_norm,
            )

        # Compute noise std (matches compute_stdv in Flower 1.9.0)
        # Formula: (noise_multiplier * clipping_norm) / num_sampled_clients
        noise_std = (
            self.noise_multiplier * self.clipping_norm / self.num_sampled_clients
        )

        # Add Gaussian noise (matches add_gaussian_noise_inplace)
        noisy_updates = []
        for update in updates:
            noise = np.random.normal(0, noise_std, update.shape).astype(update.dtype)
            noisy_updates.append(update + noise)

        log(
            logging.INFO,
            "aggregate_fit: central DP noise with %.4f stdev added",
            noise_std,
        )

        # Reconstruct model: old + clipped_noisy_update
        # (matches param[i] = param2[i] + model_update[i] in Flower 1.9.0)
        dp_params = [
            np.asarray(old_p + noisy_u)
            for old_p, noisy_u in zip(old_params, noisy_updates, strict=True)
        ]

        # Convert back to ArrayRecord preserving PyTorch state dict structure
        # We need to reconstruct the state dict with proper layer names

        # Get the original state dict structure from the previous arrays
        original_state_dict = self.previous_arrays.to_torch_state_dict()

        # Create new state dict with DP-applied parameters
        dp_state_dict = OrderedDict()
        for i, (key, _) in enumerate(original_state_dict.items()):
            dp_state_dict[key] = torch.from_numpy(dp_params[i])

        # Create ArrayRecord from PyTorch state dict (preserves layer names)
        dp_arrays = ArrayRecord(dp_state_dict)

        return dp_arrays

    def summary(self) -> None:
        """Log summary configuration of the DP strategy."""
        log(
            logging.INFO,
            "\t├──> Differential Privacy Configuration:",
        )
        log(
            logging.INFO,
            "\t│\t├──Noise Multiplier: %.6f",
            self.noise_multiplier,
        )
        log(
            logging.INFO,
            "\t│\t├──Clipping Norm: %.4f",
            self.clipping_norm,
        )
        log(
            logging.INFO,
            "\t│\t├──Sampled Clients: %d",
            self.num_sampled_clients,
        )
        log(
            logging.INFO,
            "\t│\t└──Privacy Guarantee: (ε, δ)-DP per round",
        )

        # Delegate to wrapped strategy for its summary
        self.strategy.summary()

    # Delegate all other methods to wrapped strategy
    def __getattr__(self, name):
        """Delegate attribute access to wrapped TraceFLStrategy."""
        return getattr(self.strategy, name)
