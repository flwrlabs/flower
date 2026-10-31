"""Null MAC model - the ``dsme-enabled = false`` control arm.

The baseline had no un-gated FedAvg curve to compare against, so the cost the
MAC layer imposes on convergence could not be quantified.

``NullMACModel`` satisfies the same interface as ``DSMEMACModel`` but always
answers "eligible, free, full bandwidth". A single factory call in
``client_app.py`` picks the model and the rest of the client is untouched, so
the control arm provably runs the identical code path with the constraints set
to their no-op values rather than taking a separate branch.

Interface parity is exact against ``dsme_model.DSMEMACModel``:

    get_client_profile(client_id, fl_round, model_size_kb, n_local_epochs=1)
    is_eligible(profile, model_size_kb, n_local_epochs=1)
    effective_bandwidth_fraction(client_id, fl_round, cap_mode="CR")
    energy_per_round_mj(model_size_kb, n_local_epochs=1, cap_mode="CR")
    power_consumption_mw(packet_size_bytes=250, cap_mode="CR")
"""

from fldsme.dsme_model import DSMEClientProfile

#: Sentinel written to ``energy_budget_mj`` when no budget model is in effect.
#: A negative value is used rather than ``inf`` because ``MetricRecord`` values
#: are serialised and JSON has no representation for infinity. Any consumer
#: reading ``residual_mj`` should treat a negative value as "unknown/unbounded".
NO_BUDGET = -1.0


class NullMACModel:
    """A MAC model that imposes no constraints at all.

    Every method returns the value that makes the corresponding DSME mechanism
    a no-op, so a run with this model is plain FedAvg over the same partitions,
    the same CNN, and the same initialisation.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Accept and discard whatever DSMEMACModel takes, so the factory can
        # construct either without knowing which.
        self.num_clients = int(kwargs.get("num_clients", 10))
        self.num_clusters = int(kwargs.get("num_clusters", 4))
        self.seed = int(kwargs.get("seed", 0))
        self._cluster_map = {
            i: i % self.num_clusters for i in range(self.num_clients)
        }

    # -- interface parity with DSMEMACModel ---------------------------------
    def power_consumption_mw(
        self, packet_size_bytes: int = 250, cap_mode: str = "CR"
    ) -> float:
        """No radio model in this arm."""
        return 0.0

    def energy_per_round_mj(
        self,
        model_size_kb: float,
        n_local_epochs: int = 1,
        cap_mode: str = "CR",
    ) -> float:
        """No energy accounting in this arm."""
        return 0.0

    def effective_bandwidth_fraction(
        self, client_id: int, fl_round: int, cap_mode: str = "CR"
    ) -> float:
        """Full uplink - the top-k mask short-circuits at 1.0."""
        return 1.0

    def get_client_profile(
        self,
        client_id: int,
        fl_round: int,
        model_size_kb: float,
        n_local_epochs: int = 1,
    ) -> DSMEClientProfile:
        """Same dataclass the DSME model returns, with no-op values.

        ``gts_slots`` is reported as 0 because there is no superframe in this
        arm; that keeps the metric distinguishable from a DSME run that
        genuinely allocated slots.
        """
        return DSMEClientProfile(
            client_id=client_id,
            cluster_id=self._cluster_map.get(
                client_id % max(1, self.num_clients), 0
            ),
            energy_budget_mj=NO_BUDGET,
            gts_slots=0,
            cap_mode="CR",
        )

    def is_eligible(
        self,
        profile: DSMEClientProfile,
        model_size_kb: float,
        n_local_epochs: int = 1,
    ) -> bool:
        """No energy gate - every client trains every round."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NullMACModel(no energy gate, no GTS limit)"


def build_mac_model(run_config, **kwargs):
    """Return the MAC model selected by ``dsme-enabled`` in the run config.

    ``kwargs`` are forwarded to ``DSMEMACModel`` and ignored by
    ``NullMACModel``, so the caller does not need to know which it will get.
    """
    from fldsme.dsme_model import DSMEMACModel

    enabled = run_config.get("dsme-enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"false", "0", "no", ""}

    return DSMEMACModel(**kwargs) if enabled else NullMACModel(**kwargs)
