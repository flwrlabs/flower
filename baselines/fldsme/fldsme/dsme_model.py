"""DSME MAC-layer model for FL client energy and bandwidth constraints.

Bridges two published papers into the FL world:

Paper 1 (PSO-DSME): Power model P = 2^(MO-SO)*{Ptx*Ttx+Prx*Trx+Pidle*Tidle}/TMD
Paper 2 (SeCAP):    Adaptive CAP -> effective service rate changes per round
"""

from dataclasses import dataclass

import numpy as np

P_TX_MW = 255.0
P_RX_MW = 135.0
P_IDLE_MW = 1.3
BASE_SF_SYMBOLS = 960
SYMBOL_DURATION_S = 16e-6


@dataclass
class DSMEClientProfile:
    """MAC-layer profile for one FL client (IoT end device)."""

    client_id: int
    cluster_id: int
    energy_budget_mj: float
    gts_slots: int
    cap_mode: str = "CR"


class DSMEMACModel:
    """Computes energy cost and bandwidth availability per FL round."""

    def __init__(
        self,
        bo: int = 6,
        mo: int = 5,
        so: int = 3,
        num_clients: int = 10,
        num_clusters: int = 4,
        energy_budget: float = 60.0,
        bandwidth_frac: float = 0.8,
        seed: int = 0,
    ):
        self.bo = bo
        self.mo = mo
        self.so = so
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.energy_budget = energy_budget
        self.bandwidth_frac = bandwidth_frac
        self.seed = int(seed)
        self._cluster_map = {i: i % num_clusters for i in range(num_clients)}

    @property
    def superframe_duration_s(self) -> float:
        return BASE_SF_SYMBOLS * (2**self.so) * SYMBOL_DURATION_S

    @property
    def multisf_duration_s(self) -> float:
        return BASE_SF_SYMBOLS * (2**self.mo) * SYMBOL_DURATION_S

    def power_consumption_mw(
        self, packet_size_bytes: int = 250, cap_mode: str = "CR"
    ) -> float:
        """Average power per multi-superframe (mW). Eq. 3 of PSO paper."""
        data_rate_bps = 650e3
        t_tx = (packet_size_bytes * 8) / data_rate_bps
        t_rx = t_tx * 1.1
        t_idle = self.superframe_duration_s * (2**self.bo + 1)

        if cap_mode == "NCR":
            t_idle *= 0.6

        t_md = self.multisf_duration_s
        sf = 2 ** (self.mo - self.so)

        return float(sf * (P_TX_MW * t_tx + P_RX_MW * t_rx + P_IDLE_MW * t_idle) / t_md)

    def energy_per_round_mj(
        self, model_size_kb: float, n_local_epochs: int = 1, cap_mode: str = "CR"
    ) -> float:
        """Estimated energy for one client for one FL round (mJ).

        Radio cost scales with model size relative to a 240KB reference
        (the stock CIFAR-10 CNN). NCR mode roughly halves radio cost
        per the SeCAP model (more frequent CAPs -> faster transmission).
        Compute cost scales with local epochs and model size.
        """
        p_mw = self.power_consumption_mw(cap_mode=cap_mode)
        size_ratio = model_size_kb / 240.0

        radio_mj = p_mw * size_ratio * self.multisf_duration_s
        if cap_mode == "NCR":
            radio_mj *= 0.5

        compute_mj = n_local_epochs * size_ratio * 5.0

        return radio_mj + compute_mj

    def effective_bandwidth_fraction(
        self, client_id: int, fl_round: int, cap_mode: str = "CR"
    ) -> float:
        """Fraction of the model update this client can transmit this round."""
        base = self.bandwidth_frac
        if cap_mode == "NCR":
            base = min(1.0, base * 1.15)

        # Mix the run seed in, otherwise every "seed" in a multi-seed sweep
        # draws an identical bandwidth sequence and the error bars are fiction.
        # seed=0 reproduces the original draws exactly.
        rng_client = np.random.default_rng(
            self.seed * 1_000_003 + client_id * 1000 + fl_round
        )
        variation = rng_client.uniform(-0.1, 0.05)
        return float(np.clip(base + variation, 0.3, 1.0))

    def get_client_profile(
        self,
        client_id: int,
        fl_round: int,
        model_size_kb: float,
        n_local_epochs: int = 1,
    ) -> DSMEClientProfile:
        """Compute one client's energy budget and bandwidth for this round."""
        cluster_id = self._cluster_map[client_id % self.num_clients]

        # Per-client depletion rate varies by cluster (simulates different
        # sensor duty cycles). Recharge every 5 rounds (solar harvesting).
        # Net: some clients deplete below CR threshold after ~10 rounds
        # but stay above NCR threshold, illustrating SeCAP's benefit.
        depletion_rate = 1.5 + (cluster_id % 3) * 0.8  # 1.5, 2.3, or 3.1 mJ/round
        depletion = fl_round * depletion_rate
        recharge = 5.0 * (fl_round // 5)
        remaining = max(5.0, self.energy_budget - depletion + recharge)

        cap_mode = "NCR" if (fl_round % 3 == 0 and cluster_id % 2 == 0) else "CR"

        gts = max(
            1, int(7 * self.effective_bandwidth_fraction(client_id, fl_round, cap_mode))
        )

        return DSMEClientProfile(
            client_id=client_id,
            cluster_id=cluster_id,
            energy_budget_mj=remaining,
            gts_slots=gts,
            cap_mode=cap_mode,
        )

    def is_eligible(
        self,
        profile: DSMEClientProfile,
        model_size_kb: float,
        n_local_epochs: int = 1,
    ) -> bool:
        """True if the client has enough energy to participate this round."""
        cost = self.energy_per_round_mj(model_size_kb, n_local_epochs, profile.cap_mode)
        return profile.energy_budget_mj >= cost
