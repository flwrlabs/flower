# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""."""


import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from logging import DEBUG, ERROR, INFO, WARN
from typing import Callable, Optional, Union, cast

from flwr.common import (
    ConfigsRecord,
    Context,
    Message,
    NDArrays,
    ParametersRecord,
    RecordSet,
    array_from_numpy,
    log,
)
from flwr.common.secure_aggregation.crypto.shamir import combine_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    generate_shared_key,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
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
from flwr.server.driver import Driver


@dataclass
class SecAggPlusAggregatorState:  # pylint: disable=R0902
    """The state of the SecAgg+ aggregator."""

    messages: list[Message]
    sampled_node_ids: set[int]
    message_type: str
    current_stage: str = Stage.SETUP
    active_node_ids: set[int] = field(default_factory=set)
    num_shares: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    quantization_range: int = 0
    mod_range: int = 0
    nid_to_neighbours: dict[int, set[int]] = field(default_factory=dict)
    nid_to_publickeys: dict[int, list[bytes]] = field(default_factory=dict)
    forward_srcs: dict[int, list[int]] = field(default_factory=dict)
    forward_ciphertexts: dict[int, list[bytes]] = field(default_factory=dict)
    aggregated_vector: NDArrays = field(default_factory=list)
    prs_info: dict[str, tuple[list[str], list[list[int]]]] = field(default_factory=dict)


MessagesHandler = Callable[[Iterable[Message], SecAggPlusAggregatorState], None]
StageCompletionHandler = Callable[[bool, SecAggPlusAggregatorState], None]


class SecAggPlusAggregator:  # pylint: disable=too-many-instance-attributes
    """The aggregator for the SecAgg+ protocol.

    Please use with the `secaggplus_base_mod` modifier on the ClientApp side.

    The SecAgg+ protocol ensures the secure summation of vectors owned by
    multiple parties, without accessing any individual vector. This aggregator
    allows the Serverapp to compute the sum of all ParametersRecords across all
    ClientApps, ensuring individual contributions remain private.

    The protocol involves four main stages:

    - 'setup': Send SecAgg+ configuration to clients and collect their public keys.
    - 'share keys': Broadcast public keys among clients and collect encrypted secret
      key shares.
    - 'collect masked vectors': Forward encrypted secret key shares to target clients
      and collect masked vectors.
    - 'unmask': Collect secret key shares to decrypt and aggregate the vectors.

    Only the aggregated vector is exposed, ensuring individual data privacy.

    Parameters
    ----------
    driver : Driver
        The Driver object used for communication with the SuperLink, from the ServerApp.
    context : Context
        The Context object from the ServerApp.
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
    clipping_range : float, optional (default: 8.0)
        The range within which vector entries are clipped before quantization.
        This parameter ensures each vector entry is bounded within
        [-clipping_range, clipping_range], facilitating quantization.
    quantization_range : int, optional (default: 4194304, this equals 2**22)
        The size of the range into which floating-point vector entries are quantized,
        mapping each entry to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 4294967296, this equals 2**32)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]). `modulus_range` must be less than 4294967296.
        Please use 2**n values for `modulus_range` to prevent overflow issues.
    timeout : Optional[float] (default: None)
        The timeout duration in seconds. If specified, the workflow will wait for
        replies for this duration each time. If `None`, there is no time limit and
        the workflow will wait until replies for all messages are received.
    on_send : Optional[MessagesHandler] (default: None)
        A callback function to be invoked before messages are sent out via the driver.
        The function receives two arguments:
        - An iterable of `Message` objects that are about to be sent.
        - The current state of the `SecAggPlusAggregator`.
    on_receive : Optional[MessagesHandler] (default: None)
        A callback function to be invoked after reply messages are received via the
        driver. The function receives two arguments:
        - An iterable of `Message` objects that have been received.
        - The current state of the `SecAggPlusAggregator`.
    on_stage_complete : Optional[StageCompletionHandler] (default: None)
        A callback function to be invoked when a stage is completed, regardless of
        whether the stage succeeded or failed. The function receives two arguments:
        - A `bool` indicating if the stage was successful (`True`) or not (`False`).
        - The current state of the `SecAggPlusAggregator`.

    Notes
    -----
    - It is generally *not recommended* to modify the state object or the messages
      within callback functions (i.e., `on_send`, `on_receive` and `on_stage_complete`)
      unless you are very certain of the changes. Unintended modifications can
      compromise or invalidate the secure aggregation process.
    - Generally, higher `num_shares` means more robust to dropouts while increasing the
      computational costs; higher `reconstruction_threshold` means better privacy
      guarantees but less tolerance to dropouts.
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
        driver: Driver,
        context: Context,
        num_shares: Union[int, float],
        reconstruction_threshold: Union[int, float],
        *,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: Optional[float] = None,
        on_send: Optional[MessagesHandler] = None,
        on_receive: Optional[MessagesHandler] = None,
        on_stage_complete: Optional[StageCompletionHandler] = None,
    ) -> None:
        self.num_shares = num_shares
        self.reconstruction_threshold = reconstruction_threshold
        self.clipping_range = clipping_range
        self.quantization_range = quantization_range
        self.modulus_range = modulus_range
        self.timeout = timeout
        self.driver = driver
        self.context = context
        self.on_send = on_send
        self.on_receive = on_receive
        self.on_stage_complete = on_stage_complete

        self._check_init_params()

    def aggregate(self, messages: list[Message]) -> Optional[Message]:
        """Send the messages and aggregate the reply messages."""
        if len(messages) == 0:
            raise ValueError("No messages to aggregate.")
        msg_type = ""
        for msg in messages:
            if msg_type == "":
                msg_type = msg.metadata.message_type
            elif msg.metadata.message_type != msg_type:
                raise ValueError("All messages must have the same message type.")

        state = SecAggPlusAggregatorState(
            messages=messages,
            sampled_node_ids={msg.metadata.dst_node_id for msg in messages},
            message_type=msg_type,
        )

        steps = (
            self.setup_stage,
            self.share_keys_stage,
            self.collect_masked_vectors_stage,
            self.unmask_stage,
        )
        log(INFO, "Secure aggregation commencing.")
        for step in steps:
            res = step(self.driver, state)
            if self.on_stage_complete:
                self.on_stage_complete(res, state)
            if not res:
                log(INFO, "Secure aggregation halted.")
                return None
        agg_msg = self.driver.create_message(
            content=RecordSet(),
            message_type=msg_type,
            dst_node_id=0,
            group_id="",
        )
        _set_all_weights(agg_msg, state.aggregated_vector, state.prs_info)
        log(INFO, "Secure aggregation completed.")
        return agg_msg

    def _check_init_params(self) -> None:  # pylint: disable=R0912
        # Check `num_shares`
        if not isinstance(self.num_shares, (int, float)):
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
        if not isinstance(self.reconstruction_threshold, (int, float)):
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

    def _check_threshold(self, state: SecAggPlusAggregatorState) -> bool:
        for node_id in state.sampled_node_ids:
            active_neighbors = state.nid_to_neighbours[node_id] & state.active_node_ids
            if len(active_neighbors) < state.threshold:
                log(ERROR, "Insufficient available nodes.")
                return False
        return True

    def setup_stage(  # pylint: disable=R0912, R0914, R0915
        self, driver: Driver, state: SecAggPlusAggregatorState
    ) -> bool:
        """Execute the 'setup' stage."""
        state.current_stage = Stage.SETUP

        # Protocol config
        sampled_node_ids = list(state.sampled_node_ids)
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
        sa_params_dict = {
            Key.STAGE: Stage.SETUP,
            Key.SAMPLE_NUMBER: num_samples,
            Key.SHARE_NUMBER: state.num_shares,
            Key.THRESHOLD: state.threshold,
            Key.CLIPPING_RANGE: state.clipping_range,
            Key.TARGET_RANGE: state.quantization_range,
            Key.MOD_RANGE: state.mod_range,
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

        # Send setup configuration to clients
        cfgs_record = ConfigsRecord(sa_params_dict)  # type: ignore
        content = RecordSet(configs_records={RECORD_KEY_CONFIGS: cfgs_record})

        def make(nid: int) -> Message:
            return driver.create_message(
                content=content,
                message_type=state.message_type,
                dst_node_id=nid,
                group_id="",
            )

        msgs = [make(node_id) for node_id in state.active_node_ids]

        # Trigger the event handler before sending messages
        if self.on_send:
            self.on_send(msgs, state)

        log(
            DEBUG,
            "[Stage 0] Sending configurations to %s clients.",
            len(state.active_node_ids),
        )
        replies = driver.send_and_receive(msgs, timeout=self.timeout)
        del msgs

        # Trigger the event handler after receiving replies
        if self.on_receive:
            self.on_receive(replies, state)

        state.active_node_ids = {
            msg.metadata.src_node_id for msg in replies if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 0] Received public keys from %s clients.",
            len(state.active_node_ids),
        )

        for msg in replies:
            if msg.has_error():
                continue
            key_dict = msg.content.configs_records[RECORD_KEY_CONFIGS]
            node_id = msg.metadata.src_node_id
            pk1, pk2 = key_dict[Key.PUBLIC_KEY_1], key_dict[Key.PUBLIC_KEY_2]
            state.nid_to_publickeys[node_id] = [cast(bytes, pk1), cast(bytes, pk2)]

        return self._check_threshold(state)

    def share_keys_stage(  # pylint: disable=R0914
        self, driver: Driver, state: SecAggPlusAggregatorState
    ) -> bool:
        """Execute the 'share keys' stage."""
        state.current_stage = Stage.SHARE_KEYS

        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid] & state.active_node_ids
            cfgs_record = ConfigsRecord(
                {str(nid): state.nid_to_publickeys[nid] for nid in neighbours}
            )
            cfgs_record[Key.STAGE] = Stage.SHARE_KEYS
            content = RecordSet(configs_records={RECORD_KEY_CONFIGS: cfgs_record})
            return driver.create_message(
                content=content,
                message_type=state.message_type,
                dst_node_id=nid,
                group_id="",
            )

        msgs = [make(node_id) for node_id in state.active_node_ids]

        # Trigger the event handler before sending messages
        if self.on_send:
            self.on_send(msgs, state)

        # Broadcast public keys to clients and receive secret key shares
        log(
            DEBUG,
            "[Stage 1] Forwarding public keys to %s clients.",
            len(state.active_node_ids),
        )
        replies = driver.send_and_receive(msgs, timeout=self.timeout)
        del msgs

        # Trigger the event handler after receiving replies
        if self.on_receive:
            self.on_receive(replies, state)

        state.active_node_ids = {
            msg.metadata.src_node_id for msg in replies if not msg.has_error()
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
        for msg in replies:
            if msg.has_error():
                continue
            node_id = msg.metadata.src_node_id
            res_dict = msg.content.configs_records[RECORD_KEY_CONFIGS]
            dst_lst = cast(list[int], res_dict[Key.DESTINATION_LIST])
            ctxt_lst = cast(list[bytes], res_dict[Key.CIPHERTEXT_LIST])
            srcs += [node_id] * len(dst_lst)
            dsts += dst_lst
            ciphertexts += ctxt_lst

        for src, dst, ciphertext in zip(srcs, dsts, ciphertexts):
            if dst in fwd_ciphertexts:
                fwd_ciphertexts[dst].append(ciphertext)
                fwd_srcs[dst].append(src)

        state.forward_srcs = fwd_srcs
        state.forward_ciphertexts = fwd_ciphertexts

        return self._check_threshold(state)

    def collect_masked_vectors_stage(
        self, driver: Driver, state: SecAggPlusAggregatorState
    ) -> bool:
        """Execute the 'collect masked vectors' stage."""
        state.current_stage = Stage.COLLECT_MASKED_VECTORS

        # Send secret key shares to clients (plus FitIns) and collect masked vectors
        def make(msg: Message) -> Message:
            nid = msg.metadata.dst_node_id
            cfgs_dict = {
                Key.STAGE: Stage.COLLECT_MASKED_VECTORS,
                Key.CIPHERTEXT_LIST: state.forward_ciphertexts[nid],
                Key.SOURCE_LIST: state.forward_srcs[nid],
            }
            cfgs_record = ConfigsRecord(cfgs_dict)  # type: ignore
            msg.content.configs_records[RECORD_KEY_CONFIGS] = cfgs_record
            return msg

        msgs = [
            make(msg)
            for msg in state.messages
            if msg.metadata.dst_node_id in state.active_node_ids
        ]

        # Trigger the event handler before sending messages
        if self.on_send:
            self.on_send(msgs, state)

        log(
            DEBUG,
            "[Stage 2] Forwarding encrypted key shares to %s clients.",
            len(state.active_node_ids),
        )
        replies = driver.send_and_receive(msgs, timeout=self.timeout)
        del msgs

        # Trigger the event handler after receiving replies
        if self.on_receive:
            self.on_receive(replies, state)

        state.active_node_ids = {
            msg.metadata.src_node_id for msg in replies if not msg.has_error()
        }
        log(
            DEBUG,
            "[Stage 2] Received masked vectors from %s clients.",
            len(state.active_node_ids),
        )

        # Clear cache
        del state.forward_ciphertexts, state.forward_srcs

        # Sum collected masked vectors and compute active/dead node IDs
        prs_info: Optional[dict[str, tuple[list[str], list[list[int]]]]] = None
        masked_vector = None
        for msg in replies:
            if msg.has_error():
                continue

            # Check if ParametersRecords in all messages are consistent
            # Initialize pr_key2info
            if prs_info is None:
                if len(msg.content.parameters_records) == 0:
                    log(
                        WARN,
                        "No parameters found in the message. Secure Aggregation "
                        "will proceed but will have no effect.",
                    )
                prs_info = _retrieve_prs_info(msg)
                state.prs_info = prs_info
            # Check consistency
            elif not _check_message_consistency(msg, prs_info):
                raise ValueError(
                    "Secure Aggregation Error: The Parameter Records in the "
                    "messages are inconsistent."
                )

            client_masked_vec = _get_all_weights(msg, list(prs_info.keys()))
            if masked_vector is None:
                masked_vector = client_masked_vec
            else:
                masked_vector = parameters_addition(masked_vector, client_masked_vec)
        if masked_vector is not None:
            masked_vector = parameters_mod(masked_vector, state.mod_range)
            state.aggregated_vector = masked_vector

        return self._check_threshold(state)

    def unmask_stage(  # pylint: disable=R0912, R0914, R0915
        self, driver: Driver, state: SecAggPlusAggregatorState
    ) -> bool:
        """Execute the 'unmask' stage."""
        state.current_stage = Stage.UNMASK

        # Construct active node IDs and dead node IDs
        active_nids = state.active_node_ids
        dead_nids = state.sampled_node_ids - active_nids

        # Send secure IDs of active and dead clients and collect key shares from clients
        def make(nid: int) -> Message:
            neighbours = state.nid_to_neighbours[nid]
            cfgs_dict = {
                Key.STAGE: Stage.UNMASK,
                Key.ACTIVE_NODE_ID_LIST: list(neighbours & active_nids),
                Key.DEAD_NODE_ID_LIST: list(neighbours & dead_nids),
            }
            cfgs_record = ConfigsRecord(cfgs_dict)  # type: ignore
            content = RecordSet(configs_records={RECORD_KEY_CONFIGS: cfgs_record})
            return driver.create_message(
                content=content,
                message_type=state.message_type,
                dst_node_id=nid,
                group_id="",
            )

        msgs = [make(node_id) for node_id in state.active_node_ids]

        # Trigger the event handler before sending messages
        if self.on_send:
            self.on_send(msgs, state)

        log(
            DEBUG,
            "[Stage 3] Requesting key shares from %s clients to remove masks.",
            len(state.active_node_ids),
        )
        replies = driver.send_and_receive(msgs, timeout=self.timeout)
        del msgs

        # Trigger the event handler after receiving replies
        if self.on_receive:
            self.on_receive(replies, state)

        state.active_node_ids = {
            msg.metadata.src_node_id for msg in replies if not msg.has_error()
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
        for msg in replies:
            if msg.has_error():
                continue
            res_dict = msg.content.configs_records[RECORD_KEY_CONFIGS]
            nids = cast(list[int], res_dict[Key.NODE_ID_LIST])
            shares = cast(list[bytes], res_dict[Key.SHARE_LIST])
            for owner_nid, share in zip(nids, shares):
                collected_shares_dict[owner_nid].append(share)

        # Remove masks for every active client after collect_masked_vectors stage
        masked_vector = state.aggregated_vector
        state.aggregated_vector = []
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

        aggregated_vector = dequantize(
            recon_parameters,
            state.clipping_range,
            state.quantization_range,
        )
        offset = -(len(active_nids) - 1) * state.clipping_range
        for vec in aggregated_vector:
            vec += offset
            # vec *= inv_dq_total_ratio
        state.aggregated_vector = aggregated_vector
        return True


def _retrieve_prs_info(msg: Message) -> dict[str, tuple[list[str], list[list[int]]]]:
    prs_info = {}
    for key, pr in msg.content.parameters_records.items():
        prs_info[key] = (list(pr.keys()), [arr.shape for arr in pr.values()])
    return prs_info


def _check_message_consistency(
    msg: Message, prs_info: dict[str, tuple[list[str], list[list[int]]]]
) -> bool:
    this_prs_info = _retrieve_prs_info(msg)
    return this_prs_info == prs_info


def _get_all_weights(msg: Message, pr_keys: list[str]) -> NDArrays:
    """Retrieve all weights from the message by the order of pr_keys."""
    all_weights: NDArrays = []
    # Retrieve weights
    for key in pr_keys:
        pr = msg.content.parameters_records[key]
        all_weights += [arr.numpy() for arr in pr.values()]
    return all_weights


def _set_all_weights(
    msg: Message,
    all_weights: NDArrays,
    prs_info: dict[str, tuple[list[str], list[list[int]]]],
    keep_input: bool = False,
) -> None:
    """Set all weights in the message."""
    idx = 0
    msg.content.parameters_records.clear()
    for pr_key in prs_info:
        pr = ParametersRecord()
        for arr_key in prs_info[pr_key][0]:
            pr[arr_key] = array_from_numpy(all_weights[idx])
            if keep_input:
                idx += 1
            else:
                del all_weights[idx]
        msg.content.parameters_records[pr_key] = pr
