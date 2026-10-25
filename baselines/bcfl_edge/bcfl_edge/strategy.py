"""bcfl_edge: FedAvg on the devices selected by Algorithm 1 (Kushwaha et al.)."""

import time
from collections.abc import Iterable
from logging import INFO
from typing import cast

import numpy as np
from bcfl import DevicePool, MinerPool, forking_multiplier, optimize, round_cost
from flwr.app import ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class BCFL(FedAvg):
    """FedAvg whose training devices minimize blockchain latency and energy.

    Implementation based on https://doi.org/10.1109/TCSII.2023.3322340

    Before the first round, every node reports its partition id and number of
    training examples. The strategy then builds the blockchain-enabled network of
    the paper (`num_devices` devices randomly associated with `num_miners`
    miners, with hardware and channel parameters drawn from the paper's Table I),
    selects `selected_devices` of them and trains on those in every round. As in
    the paper's experiments, the network is static, so the selection is made once.

    Parameters
    ----------
    num_devices : int
        Number of nodes (edge devices) in the federation.
    num_miners : int
        Number of miners N_M.
    selected_devices : int
        Number of devices S_D trained on per round.
    beta : float (default: 0.5)
        Weight of latency against energy in the objective (Eq. 11).
    selection : str (default: "bcfl")
        "bcfl" for Algorithm 1, or "random" for the paper's random baseline.
    seed : int (default: 42)
        Seed of the simulated network and of the random baseline.
    local_epochs : int (default: 1)
        Local epochs per round, used in the devices' computation cost (Eq. 2).
    discovery_timeout : float (default: 600)
        Seconds to wait for all nodes to connect and report their profile.
    **kwargs
        Passed to FedAvg.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        num_devices: int,
        num_miners: int,
        selected_devices: int,
        beta: float = 0.5,
        selection: str = "bcfl",
        seed: int = 42,
        local_epochs: int = 1,
        discovery_timeout: float = 600.0,
        **kwargs,
    ) -> None:
        if selection not in ("bcfl", "random"):
            raise ValueError(f"selection must be 'bcfl' or 'random', got {selection!r}")
        super().__init__(min_available_nodes=num_devices, **kwargs)
        self.num_devices = num_devices
        self.num_miners = num_miners
        self.selected_devices = selected_devices
        self.beta = beta
        self.selection = selection
        self.seed = seed
        self.local_epochs = local_epochs
        self.discovery_timeout = discovery_timeout
        self.round_cost: dict[str, float | int | list[int]] = {}
        self._train_nodes: list[int] = []

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> BCFL settings:")
        log(INFO, "\t│\t├── Devices: %d, miners: %d", self.num_devices, self.num_miners)
        log(INFO, "\t│\t├── Selected devices: %d", self.selected_devices)
        log(INFO, "\t│\t└── Selection: %s (beta=%s)", self.selection, self.beta)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Send the global model to the selected devices."""
        if not self._train_nodes:
            self._train_nodes = self._select_devices(grid)
        config["server-round"] = server_round
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, self._train_nodes, MessageType.TRAIN)

    def _select_devices(self, grid: Grid) -> list[int]:
        """Query every node, build the network and run the device selection."""
        deadline = time.time() + self.discovery_timeout
        while len(node_ids := sorted(grid.get_node_ids())) < self.num_devices:
            if time.time() > deadline:
                raise RuntimeError(
                    f"Only {len(node_ids)} of {self.num_devices} nodes connected "
                    f"within {self.discovery_timeout} s."
                )
            log(INFO, "Waiting for nodes: %d of %d", len(node_ids), self.num_devices)
            time.sleep(1)
        node_ids = node_ids[: self.num_devices]
        queries = [
            Message(
                content=RecordDict({self.configrecord_key: ConfigRecord()}),
                message_type=MessageType.QUERY,
                dst_node_id=node_id,
            )
            for node_id in node_ids
        ]
        profiles = []
        replies = list(
            grid.send_and_receive(queries, timeout=max(1.0, deadline - time.time()))
        )
        if len(replies) < self.num_devices:
            raise RuntimeError(
                f"Only {len(replies)} of {self.num_devices} nodes reported their "
                f"profile within {self.discovery_timeout} s."
            )
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(f"Node query failed: {reply.error.reason}")
            profile = reply.content.metric_records["profile"]
            profiles.append(
                (
                    cast(int, profile["partition-id"]),
                    cast(int, profile["num-examples"]),
                    reply.metadata.src_node_id,
                )
            )
        profiles.sort()  # device index = partition id
        data_size = [size for _, size, _ in profiles]

        devices = DevicePool.random(
            len(profiles),
            self.num_miners,
            seed=self.seed,
            data_size=data_size,
            epochs=self.local_epochs,
        )
        miners = MinerPool.random(self.num_miners, seed=self.seed)
        if self.selection == "bcfl":
            result = optimize(devices, miners, self.selected_devices, self.beta)
            chosen, miner_set = result.selected_devices, result.selected_miners
        else:
            rng = np.random.default_rng(self.seed)
            chosen = np.sort(
                rng.choice(len(profiles), self.selected_devices, replace=False)
            )
            miner_set = np.unique(devices.miner[chosen])

        latency, energy = round_cost(devices, miners, chosen, miner_set)
        all_miners = np.unique(devices.miner)
        all_latency, all_energy = round_cost(
            devices, miners, np.arange(devices.n), all_miners
        )
        self.round_cost = {
            "selected-devices": [int(i) for i in chosen],
            "selected-miners": [int(m) for m in miner_set],
            "a-fork": forking_multiplier(
                miners.block_time(miner_set), miners.mining_rate
            ),
            "latency": float(latency),
            "energy": float(energy),
            "latency-all-devices": float(all_latency),
            "energy-all-devices": float(all_energy),
            "a-fork-all-devices": forking_multiplier(
                miners.block_time(all_miners), miners.mining_rate
            ),
        }
        log(
            INFO,
            "BCFL (%s): %d devices on %d miners, A_fork=%d, "
            "round latency %.2f s (all devices: %.2f s), energy %.2f J (all: %.2f J)",
            self.selection,
            len(chosen),
            len(miner_set),
            self.round_cost["a-fork"],
            latency,
            all_latency,
            energy,
            all_energy,
        )
        return [profiles[i][2] for i in chosen]
