# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Workflow for the SecAgg+ protocol."""


import random
from dataclasses import dataclass, field
from logging import DEBUG, ERROR, INFO, WARN
from typing import cast

import flwr.common.recorddict_compat as compat
from flwr.app.message_type import MessageType
from flwr.common import (
    ConfigRecord,
    Context,
    FitRes,
    Message,
    NDArrays,
    RecordDict,
    bytes_to_ndarray,
    log,
    ndarrays_to_parameters,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    generate_shared_key,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_extract,
    get_parameters_shape,
    parameters_addition,
    parameters_mod,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import dequantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    Key,
    Stage,
)
from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen
from flwr.server.client_proxy import ClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.grid import Grid
from flwr.supercore.primitives.asymmetric import (
    bytes_to_private_key,
    bytes_to_public_key,
)

from ..constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from ..constant import Key as WorkflowKey


@dataclass
class WorkflowState:  # pylint: disable=R0902
    """The state of the SecAgg+ protocol."""

    nid_to_proxies: dict[int, ClientProxy] = field(default_factory=dict)
    nid_to_fitins: dict[int, RecordDict] = field(default_factory=dict)
    sampled_node_ids: set[int] = field(default_factory=set)
    active_node_ids: set[int] = field(default_factory=set)
    num_shares: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    quantization_range: int = 0
    mod_range: int = 0
    max_weight: float = 0.0
    nid_to_neighbours: dict[int, set[int]] = field(default_factory=dict)
    nid_to_publickeys: dict[int, list[bytes]] = field(default_factory=dict)
    forward_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    aggregate_ndarrays: NDArrays = field(default_factory=list)
    legacy_results: list[tuple[ClientProxy, FitRes]] = field(default_factory=list)
    failures: list[Exception] = field(default_factory=list)


class SecAggPlusWorkflow:
    """The workflow for the SecAgg+ protocol.

    The SecAgg+ protocol ensures the secure summation of integer vectors owned by
    multiple parties, without accessing any individual integer vector. This workflow
    allows the server to compute the weighted average of model parameters across all
    clients, ensuring individual contributions remain private. This is achieved by
    clients sending both, a weighting factor and a weighted version of the locally
    updated parameters, both of which are masked for privacy. Specifically, each
    client uploads "[w, w * params]" with masks, where weighting factor 'w' is the
    number of examples ('num_examples') and 'params' represents the model parameters
    ('parameters') from the client's `FitRes`. The server then aggregates these
    contributions to compute the weighted average of model parameters.

    The protocol involves four main stages:

    - 'setup': Send SecAgg+ configuration to clients and collect their public keys.
    - 'share keys': Broadcast public keys among clients and collect encrypted secret
      key shares.
    - 'collect masked vectors': Forward encrypted secret key shares to target clients
      and collect masked model parameters.
    - 'unmask': Collect secret key shares to decrypt and aggregate the model parameters.

    Only the aggregated model parameters are exposed and passed to
    `Strategy.aggregate_fit`, ensuring individual data privacy.

    Parameters
    ----------
    num_shares : Union[int, float]
        The number of shares into which each client's private key is split under
        the SecAgg+ protocol. If specified as a float, it represents the proportion
        of all selected clients, and the number of shares will be set dynamically in
        the run time. A private key can be reconstructed from these shares, allowing
        for the secure aggregation of model updates. Each client sends one share to
        each of its neighbors while retaining one.
    reconstruction_threshold : Union[int, float]
        The minimum number of shares required to reconstruct a client's private key,
        or, if specified as a float, it represents the proportion of the total number
        of shares needed for reconstruction. This threshold ensures privacy by allowing
        for the recovery of contributions from dropped clients during aggregation,
        without compromising individual client data.
    max_weight : Optional[float] (default: 1000.0)
        The maximum value of the weight that can be assigned to any single client's
        update during the weighted average calculation on the server side, e.g., in the
        FedAvg algorithm.
    clipping_range : float, optional (default: 8.0)
        The range within which model parameters are clipped before quantization.
        This parameter ensures each model parameter is bounded within
        [-clipping_range, clipping_range], facilitating quantization.
    quantization_range : int, optional (default: 4194304, this equals 2**22)
        The size of the range into which floating-point model parameters are quantized,
        mapping each parameter to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 4294967296, this equals 2**32)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]). `modulus_range` must be less than 4294967296.
        Please use 2**n values for `modulus_range` to prevent overflow issues.
    timeout : Optional[float] (default: None)
        The timeout duration in seconds. If specified, the workflow will wait for
        replies for this duration each time. If `None`, there is no time limit and
        the workflow will wait until replies for all messages are received.

    Notes
    -----
    - Generally, higher `num_shares` means more robust to dropouts while increasing the
      computational costs; higher `reconstruction_threshold` means better privacy
      guarantees but less tolerance to dropouts.
    - Too large `max_weight` may compromise the precision of the quantization.
    - `modulus_range` must be 2**n and larger than `quantization_range`.
    - When `num_shares` is a float, it is interpreted as the proportion of all selected
      clients, and hence the number of shares will be determined in the runtime. This
      allows for dynamic adjustment based on the total number of participating clients.
    - Similarly, when `reconstruction_threshold` is a float, it is interpreted as the
      proportion of the number of shares needed for the reconstruction of a private key.
      This feature enables flexibility in setting the security threshold relative to the
      number of distributed shares.
    - `num_shares`, `reconstruction_threshold`, and the quantization parameters
      (`clipping_range`, `quantization_range`, `modulus_range`) play critical roles in
      balancing privacy, robustness, and efficiency within the SecAgg+ protocol.
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_shares: int | float,
        reconstruction_threshold: int | float,
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: float | None = None,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.max_weight = max_weight
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout

        self._check_init_params()

    def __call__(self, grid: Grid, context: Context) -> None:
        """Run the SecAgg+ protocol."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )
        state = WorkflowState()

        steps = (
            self.setup_stage,
            self.share_keys_stage,
            self.collect_masked_vectors_stage,
            self.unmask_stage,
        )
        log(INFO, "Secure aggregation commencing.")
        for step in steps:
            if not step(grid, context, state):
                log(INFO, "Secure aggregation halted.")
                return
        log(INFO, "Secure aggregation completed.")

    def _check_init_params(self) -> None:  # pylint: disable=R0912
        # Check `num_shares`
        if not isinstance(self.num_shares, (int | float)):
            raise TypeError("`num_shares` must be of type int or float.")
        if isinstance(self.num_shares, int):
            if self.num_shares == 1:
                self.num_shares = 1.0
            elif self.num_shares <= 2:
                raise ValueError("`num_shares` as an integer must be greater than 2.")
            elif self.num_shares > self.modulus_range / self.quantization_range:
                log(
                    WARN,
                    "A `num_shares` larger than `modulus_range / quantization_range` "
                    "will potentially cause overflow when computing the aggregated "
                    "model parameters.",
                )
        elif self.num_shares <= 0:
            raise ValueError("`num_shares` as a float must be greater than 0.")

        # Check `reconstruction_threshold`
        if not isinstance(self.reconstruction_threshold, (int | float)):
            raise TypeError("`reconstruction_threshold` must be of type int or float.")
        if isinstance(self.reconstruction_threshold, int):
            if self.reconstruction_threshold == 1:
                self.reconstruction_threshold = 1.0
            elif isinstance(self.num_shares, int):
                if self.reconstruction_threshold >= self.num_shares:
                    raise ValueError(
                        "`reconstruction_threshold` must be less than `num_shares`."
                    )
        else:
            if not 0 < self.reconstruction_threshold <= 1:
                raise ValueError(
                    "If `reconstruction_threshold` is a float, "
                    "it must be greater than 0 and less than or equal to 1."
                )

        # Check `max_weight`
        if self.max_weight <= 0:
            raise ValueError("`max_weight` must be greater than 0.")

        # Check `quantization_range`
        if self.quantization_range <= 0:
            raise ValueError("`quantization_range` must be greater than 0.")

        # Check `quantization_range`
        if not isinstance(self.quantization_range, int) or self.quantization_range <= 0:
            raise ValueError(
                "`quantization_range` must be an integer and greater than 0."
            )

        # Check `modulus_range`
        if (
            not isinstance(self.modulus_range, int)
            or self.modulus_range <= self.quantization_range
        ):
            raise ValueError(
                "`modulus_range` must be an integer and "
                "greater than `quantization_range`."
            )
        if bin(self.modulus_range).count("1") != 1:
            raise ValueError("`modulus_range` must be a power of 2.")

    def _check_threshold(self, state: WorkflowState) -> bool:
        for node_id in state.sampled_node_ids:
            active_neighbors = state.nid_to_neighbours[node_id] & state.active_node_ids
            if len(active_neighbors) < state.threshold:
                log(ERROR, "Insufficient available nodes.")
                return False
        return True

    def setup_stage(  # pylint: disable=R0912, R0914, R0915
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'setup' stage."""
        # Obtain fit instructions
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])
        parameters = compat.arrayrecord_to_parameters(
            context.state.array_records[MAIN_PARAMS_RECORD],
            keep_input=True,
        )
        proxy_fitins_lst = context.strategy.configure_fit(
            current_round, parameters, context.client_manager
        )
        if not proxy_fitins_lst:
            log(INFO, "configure_fit: no clients selected, cancel")
            return False
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(proxy_fitins_lst),
            context.client_manager.num_available(),
        )

        state.nid_to_fitins = {
            proxy.node_id: compat.fitins_to_recorddict(fitins, True)
            for proxy, fitins in proxy_fitins_lst
        }
        state.nid_to_proxies = {proxy.node_id: proxy for proxy, _ in proxy_fitins_lst}

        # Protocol config
        sampled_node_ids = list(state.nid_to_fitins.keys())
        num_samples = len(sampled_node_ids)
        if num_samples < 2:
            log(ERROR, "The number of samples should be greater than 1.")
            return False
        if isinstance(self.num_shares, float):
            state.num_shares = round(self.num_shares * num_samples)
            # If even
            if state.num_shares < num_samples and state.num_shares & 1 == 0:
                state.num_shares += 1
            # If too small
            if state.num_shares <= 2:
                state.num_shares = num_samples
        else:
            state.num_shares = self.num_shares
        if isinstance(self.reconstruction_threshold, float):
            state.threshold = round(self.reconstruction_threshold * state.num_shares)
            # Avoid too small threshold
            state.threshold = max(state.threshold, 2)
        else:
            state.threshold = self.reconstruction_threshold
        state.active_node_ids = set(sampled_node_ids)
        state.clipping_range = self.clipping_range
        state.quantization_range = self.quantization_range
        state.mod_range = self.modulus_range
        state.max_weight = self.max_weight
        sa_params_dict = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_samples,
            Key.SHARE_NUMBER: state.num_shares,
            Key.THRESHOLD: state.threshold,
            Key.CLIPPING_RANGE: state.clipping_range,
            Key.TARGET_RANGE: state.quantization_range,
            Key.MOD_RANGE: state.mod_range,
            Key.MAX_WEIGHT: state.max_weight,
        }

        # The number of shares should better be odd in the SecAgg+ protocol.
        if num_samples != state.num_shares and state.num_shares & 1 == 0:
            log(WARN, "Number of shares in the SecAgg+ protocol should be odd.")
            state.num_shares += 1

        # Shuffle node IDs
        random.shuffle(sampled_node_ids)
        # Build neighbour relations (node ID -> secure IDs of neighbours)
        half_share = state.num_shares >> 1
        state.nid_to_neighbours = {
            nid: {
                sampled_node_ids[(idx + offset) % num_samples]
                for offset in range(-half_share, half_share + 1)
            }
            for idx, nid in enumerate(sampled_node_ids)
        }

        state.sampled_node_ids = state.active_node_ids

        # Send setup configuration to clients
        cfg_record = ConfigRecord(sa_params_dict)  # type: ignore
        content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})

        def make(nid: int) -> Message:
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(
            DEBUG,
            "[Stage 0] Sending configurations to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 0] Received public keys from %s clients.",
            len(state.active_node_ids),
        )

        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            key_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            node_id = msg.metadata.src_node_id
            pk1, pk2 = key_dict[Key.PUBLIC_KEY_1], key_dict[Key.PUBLIC_KEY_2]
            state.nid_to_publickeys[node_id] = [cast(bytes, pk1), cast(bytes, pk2)]

        return self._check_threshold(state)

    def share_keys_stage(  # pylint: disable=R0914
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'share keys' stage."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid] & state.active_node_ids
            cfg_record = ConfigRecord(
                {str(nid): state.nid_to_publickeys[nid] for nid in neighbours}
            )
            cfg_record[Key.STAGE] = Stage.SHARE_KEYS
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        # Broadcast public keys to clients and receive secret key shares
        log(
            DEBUG,
            "[Stage 1] Forwarding public keys to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 1] Received encrypted key shares from %s clients.",
            len(state.active_node_ids),
        )

        # Build forward packet list dictionary
        srcs: list[int] = []
        dsts: list[int] = []
        ciphertexts: list[bytes] = []
        fwd_ciphertexts: dict[int, list[bytes]] = {
            nid: [] for nid in state.active_node_ids
        }  # dest node ID -> list of ciphertexts
        fwd_srcs: dict[int, list[int]] = {
            nid: [] for nid in state.active_node_ids
        }  # dest node ID -> list of src node IDs
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            node_id = msg.metadata.src_node_id
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])
            srcs += [node_id] * len(dst_lst)
            dsts += dst_lst
            ciphertexts += ctxt_lst

        for src, dst, ciphertext in zip(srcs, dsts, ciphertexts, strict=True):
            if dst in fwd_ciphertexts:
                fwd_ciphertexts[dst].append(ciphertext)
                fwd_srcs[dst].append(src)

        state.forward_srcs = fwd_srcs
        state.forward_ciphertexts = fwd_ciphertexts

        return self._check_threshold(state)

    def collect_masked_vectors_stage(
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'collect masked vectors' stage."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]

        # Send secret key shares to clients (plus FitIns) and collect masked vectors
        def make(nid: int) -> Message:
            cfg_dict = {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.CIPHERTEXT_LIST: state.forward_ciphertexts[nid],
                Key.SOURCE_LIST: state.forward_srcs[nid],
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = state.nid_to_fitins[nid]
            content.config_records[RECORD_KEY_CONFIGS] = cfg_record
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(cfg[WorkflowKey.CURRENT_ROUND]),
            )

        log(
            DEBUG,
            "[Stage 2] Forwarding encrypted key shares to %s clients.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 2] Received masked vectors from %s clients.",
            len(state.active_node_ids),
        )

        # Clear cache
        del state.forward_ciphertexts, state.forward_srcs, state.nid_to_fitins

        # Sum collected masked vectors and compute active/dead node IDs
        masked_vector = None
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            bytes_list = cast(list[bytes], res_dict[Key.MASKED_PARAMETERS])
            client_masked_vec = [bytes_to_ndarray(b) for b in bytes_list]
            if masked_vector is None:
                masked_vector = client_masked_vec
            else:
                masked_vector = parameters_addition(masked_vector, client_masked_vec)
        if masked_vector is not None:
            masked_vector = parameters_mod(masked_vector, state.mod_range)
            state.aggregate_ndarrays = masked_vector

        # Backward compatibility with Strategy
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            fitres = compat.recorddict_to_fitres(msg.content, True)
            proxy = state.nid_to_proxies[msg.metadata.src_node_id]
            state.legacy_results.append((proxy, fitres))

        return self._check_threshold(state)

    def unmask_stage(  # pylint: disable=R0912, R0914, R0915
        self, grid: Grid, context: LegacyContext, state: WorkflowState
    ) -> bool:
        """Execute the 'unmask' stage."""
        cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
        current_round = cast(int, cfg[WorkflowKey.CURRENT_ROUND])

        # Construct active node IDs and dead node IDs
        active_nids = state.active_node_ids
        dead_nids = state.sampled_node_ids - active_nids

        # Send secure IDs of active and dead clients and collect key shares from clients
        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid]
            cfg_dict = {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: list(neighbours & active_nids),
                Key.DEAD_NODE_ID_LIST: list(neighbours & dead_nids),
            }
            cfg_record = ConfigRecord(cfg_dict)  # type: ignore
            content = RecordDict({RECORD_KEY_CONFIGS: cfg_record})
            return Message(
                content=content,
                dst_node_id=nid,
                message_type=MessageType.TRAIN,
                group_id=str(current_round),
            )

        log(
            DEBUG,
            "[Stage 3] Requesting key shares from %s clients to remove masks.",
            len(state.active_node_ids),
        )
        msgs = grid.send_and_receive(
            [make(node_id) for node_id in state.active_node_ids], timeout=self.timeout
        )
        state.active_node_ids = {
            msg.metadata.src_node_id for msg in msgs if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 3] Received key shares from %s clients.",
            len(state.active_node_ids),
        )

        # Build collected shares dict
        collected_shares_dict: dict[int, list[bytes]] = {}
        for nid in state.sampled_node_ids:
            collected_shares_dict[nid] = []
        for msg in msgs:
            if msg.has_error():
                state.failures.append(Exception(msg.error))
                continue
            res_dict = msg.content.config_records[RECORD_KEY_CONFIGS]
            nids = cast(list[int], res_dict[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res_dict[Key.SHARE_LIST])
            for owner_nid, share in zip(nids, shares, strict=True):
                collected_shares_dict[owner_nid].append(share)

        # Remove masks for every active client after collect_masked_vectors stage
        masked_vector = state.aggregate_ndarrays
        del state.aggregate_ndarrays
        for nid, share_list in collected_shares_dict.items():
            if len(share_list) < state.threshold:
                log(
                    ERROR, "Not enough shares to recover secret in unmask vectors stage"
                )
                return False
            secret = combine_shares(share_list)
            if nid in active_nids:
                # The seed for PRG is the private mask seed of an active client.
                private_mask = pseudo_rand_gen(
                    secret, state.mod_range, get_parameters_shape(masked_vector)
                )
                masked_vector = parameters_subtraction(masked_vector, private_mask)
            else:
                # The seed for PRG is the secret key 1 of a dropped client.
                neighbours = state.nid_to_neighbours[nid]
                neighbours.remove(nid)

                for neighbor_nid in neighbours:
                    shared_key = generate_shared_key(
                        bytes_to_private_key(secret),
                        bytes_to_public_key(state.nid_to_publickeys[neighbor_nid][0]),
                    )
                    pairwise_mask = pseudo_rand_gen(
                        shared_key, state.mod_range, get_parameters_shape(masked_vector)
                    )
                    if nid > neighbor_nid:
                        masked_vector = parameters_addition(
                            masked_vector, pairwise_mask
                        )
                    else:
                        masked_vector = parameters_subtraction(
                            masked_vector, pairwise_mask
                        )
        recon_parameters = parameters_mod(masked_vector, state.mod_range)
        q_total_ratio, recon_parameters = factor_extract(recon_parameters)
        inv_dq_total_ratio = state.quantization_range / q_total_ratio
        # recon_parameters = parameters_divide(recon_parameters, total_weights_factor)
        aggregated_vector = dequantize(
            recon_parameters,
            state.clipping_range,
            state.quantization_range,
        )
        offset = -(len(active_nids) - 1) * state.clipping_range
        for vec in aggregated_vector:
            vec += offset
            vec *= inv_dq_total_ratio

        # Backward compatibility with Strategy
        results = state.legacy_results
        parameters = ndarrays_to_parameters(aggregated_vector)
        for _, fitres in results:
            fitres.parameters = parameters

        # No exception/failure handling currently
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(state.failures),
        )
        aggregated_result = context.strategy.aggregate_fit(
            current_round, results, state.failures  # type: ignore
        )
        parameters_aggregated, metrics_aggregated = aggregated_result

        # Update the parameters and write history
        if parameters_aggregated:
            arr_record = compat.parameters_to_arrayrecord(parameters_aggregated, True)
            context.state.array_records[MAIN_PARAMS_RECORD] = arr_record
            context.history.add_metrics_distributed_fit(
                server_round=current_round, metrics=metrics_aggregated
            )
        return True
