# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""NodeApp API for decentralized applications.

`NodeApp` provides a decorator-based API similar to `ClientApp`/`ServerApp`:

>>> app = NodeApp(subject="trainer", run_config={"local-epochs": 1})
>>>
>>> @app.train()
... def train(message: str, run_config: dict) -> None:
...     _ = (message, run_config)
>>>
>>> @app.evaluate()
... def evaluate(message: str, run_config: dict) -> None:
...     _ = (message, run_config)

It also supports loading multiple app definitions from `pyproject.toml`.
"""

from __future__ import annotations

import inspect
import importlib
import json
import logging
import base64
import copy
import dataclasses
from pathlib import Path
import random
from typing import Any, Callable, Dict, Optional, Union

from flwr.app.user_config import UserConfig
from flwr.client.mod.utils import make_ffn
from flwr.common import RecordDict
from flwr.common.logger import log
from flwr.common.constant import NUM_PARTITIONS_KEY, PARTITION_ID_KEY
import numpy as np
from flwr.common.record.arrayrecord import ArrayRecord
from flwr.common.record.array import Array
from flwr.common.record.configrecord import ConfigRecord
from flwr.common.record.metricrecord import MetricRecord
from flwr.common.typing import NDArrays
from flwr.serverapp.strategy.fedavg import FedAvg
from flwr.serverapp.strategy.strategy import Strategy

from flwr.app.message_type import MessageType
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.context import Context
from flwr.common.message import Message
from flwr.supercore.inflatable.inflatable_object import InflatableObject
from flwr.supercore.inflatable.inflatable_utils import inflate_object_from_contents

from flwr.decentralized.common.message import AggregateRequest
from flwr.decentralized.common.run_config import DLRunConfig
from flwr.decentralized.common.typing import Action, Mode
from nodemanager.application._application import App  # type: ignore[import-untyped]

NodeFn = ClientAppCallable
LOGGER = logging.getLogger(__name__)


class NodeApp(App):
    """Decorator-based decentralized app built on top of nodemanager `App`.

    Parameters
    ----------
    subject : str
        Logical topic/application name. This value is used as `App.name`.
    timeout : int, default=30
        Elapsed time between periodic runs (forwarded to `App.elapsed_time`).
    train_config : dict[str, Any] | None, default=None
        Per-app runtime/training configuration.

    Notes
    -----
    - Use :meth:`train` and :meth:`evaluate` decorators to register handlers.
    - Incoming messages are routed by event type (`train` or `evaluate`).
    """

    # one too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    def __init__(
        self,
        subject: str,
        initial_arrays: ArrayRecord | None = None,
        data_config: ConfigRecord | None = None,
        strategy: Strategy | None = None,
        run_config: DLRunConfig | Dict[str, Any] | None = None,
        train_config: Optional[ConfigRecord] = None,
        eval_config: Optional[ConfigRecord] = None,
        mods: list[Mod] | None = None,
        elapsed_time: int = 2,
        id: Optional[str] = None,
        node_name: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """Initialize a NodeApp instance.

        Parameters
        ----------
        subject : str
            Logical topic/application name. This value is used as `App.name`.
        initial_arrays : ArrayRecord | None, default=None
            Initial arrays for the NodeApp. Represents model parameters or other stateful arrays to be used in training/evaluation.
        data_config : ConfigRecord | None, default=None
            Data configuration for the NodeApp.
        strategy : Strategy | None, default=None
            Strategy for the NodeApp.
        run_config : DLRunConfig | Dict[str, Any] | None, default=None
            Run configuration for the NodeApp.
        train_config : Optional[ConfigRecord], default=None
            Training configuration for the NodeApp.
        eval_config : Optional[ConfigRecord], default=None
            Evaluation configuration for the NodeApp.
        mods : list[Mod] | None, default=None
            List of mods for the NodeApp.
        elapsed_time : int, default=2
            Elapsed time for the NodeApp.
        id : Optional[str], default=None
            ID for the NodeApp.
        node_name : Optional[str], default=None
            Node name for the NodeApp.
        timeout : int, default=30
            Timeout for the NodeApp.
        """
        if isinstance(run_config, dict):
            dl_fields = {field.name for field in dataclasses.fields(DLRunConfig)}
            runtime_run_config = {
                key: value for key, value in run_config.items() if key in dl_fields
            }
            extra_train_config = {
                key: value for key, value in run_config.items() if key not in dl_fields
            }
            run_config = runtime_run_config
            if extra_train_config:
                base_train_config: Dict[str, Any] = (
                    dict(train_config) if train_config is not None else {}
                )
                base_train_config.update(extra_train_config)
                train_config = ConfigRecord(base_train_config)

        self.data_config = data_config or ConfigRecord()
        if data_config is not None:
            self._validate_data_config()
        self.arrays = initial_arrays or ArrayRecord({})
        self.train_metrics: MetricRecord | None = None
        self.eval_metrics: MetricRecord | None = None

        run_config = self._parse_run_config(run_config)

        random.seed(run_config.seed)
        super().__init__(
            name=subject,
            elapsed_time=elapsed_time,
            cycles=run_config.get_cycles(),
            id=id,
        )

        self._mods: list[Mod] = mods if mods is not None else []
        self.subject = subject

        self.strategy: Strategy = strategy or FedAvg()
        self.parameters: Optional[NDArrays] = None

        self.run_config = run_config
        self.train_config: ConfigRecord = (
            ConfigRecord() if train_config is None else train_config
        )
        self.eval_config: ConfigRecord = (
            ConfigRecord() if eval_config is None else eval_config
        )
        self._train_fn: Optional[NodeFn] = None
        self._evaluate_fn: Optional[NodeFn] = None

        self.timeout = timeout
        self.node_name = node_name

        self.waiting_for_reply_from: set = (
            set()
        )  # track pending replies to avoid duplicate sends
        self._handled_requests: set[tuple[str, str, int]] = set()
        self.count_steps: int = 0
        self.current_round: int = 0

    def _validate_data_config(self) -> None:
        """Validate that data_config contains required keys."""
        if not self.data_config:
            raise ValueError("data_config must not be empty")
        if PARTITION_ID_KEY not in self.data_config:
            raise ValueError(f"data_config must include '{PARTITION_ID_KEY}' key")
        if NUM_PARTITIONS_KEY not in self.data_config:
            raise ValueError(f"data_config must include '{NUM_PARTITIONS_KEY}' key")

    def set_data_config(self, override: ConfigRecord | Dict[str, Any]) -> None:
        """Override data_config with provided config (merge on top of existing).

        Parameters
        ----------
        override : ConfigRecord | Dict[str, Any]
            Configuration dict containing at minimum PARTITION_ID_KEY and NUM_PARTITIONS_KEY.
            Will be merged into self.data_config.
        """
        if isinstance(override, dict):
            override = ConfigRecord(override)
        self.data_config.update(override)
        self._validate_data_config()
        random.seed(self.run_config.seed + int(self.data_config[PARTITION_ID_KEY]))

    def for_node(self, partition_id: int, num_partitions: int) -> "NodeApp":
        """Return a deep copy of this NodeApp configured for a specific node.

        Parameters
        ----------
        partition_id : int
            Zero-based index of the virtual node this instance will represent.
        num_partitions : int
            Total number of virtual nodes in the simulation.

        Returns
        -------
        NodeApp
            A fully independent copy with ``data_config`` populated for the
            requested partition.
        """
        clone: NodeApp = copy.deepcopy(self)
        clone.set_data_config(
            {
                PARTITION_ID_KEY: partition_id,
                NUM_PARTITIONS_KEY: num_partitions,
            }
        )
        return clone

    def train(self, *, mods: list[Mod] | None = None) -> Callable[[NodeFn], NodeFn]:
        """Register the function handling `train` events.

        Parameters
        ----------
        mods : list[Mod] | None, optional
            List of mods to apply to the train function, by default None.

        Returns
        -------
        Callable[[NodeFn], NodeFn]
            A decorator registering the provided function.
        """

        def decorator(func: NodeFn) -> NodeFn:
            self._train_fn = make_ffn(func, self._mods + (mods or []))
            return func

        return decorator

    def evaluate(self, *, mods: list[Mod] | None = None) -> Callable[[NodeFn], NodeFn]:
        """Register the function handling `evaluate` events.

        Parameters
        ----------
        mods : list[Mod] | None, optional
            List of mods to apply to the evaluate function, by default None.

        Returns
        -------
        Callable[[NodeFn], NodeFn]
            A decorator registering the provided function.
        """

        def decorator(func: NodeFn) -> NodeFn:
            self._evaluate_fn = make_ffn(func, self._mods + (mods or []))
            return func

        return decorator

    def create_message(
        self,
        message_type: str,
        config: ConfigRecord,
    ) -> Message:
        """Create a Flower `Message` for a NodeApp callback.

        Parameters
        ----------
        message_type : str
            Type of the message (e.g., "train" or "evaluate").
        config : ConfigRecord
            Configuration to include in the message content.

        Returns
        -------
        Message
            A Flower `Message` instance containing the provided configuration and current arrays.
        """
        record = RecordDict(
            {
                self.strategy.arrayrecord_key: self.arrays,
                self.strategy.configrecord_key: config,
            }
        )
        return Message(
            content=record,
            dst_node_id=0,
            message_type=message_type,
            ttl=self.timeout,
            group_id=self.subject,
        )

    def create_context(
        self,
        *,
        run_id: int = 0,
        state: Optional[RecordDict] = None,
        run_config: Optional[ConfigRecord] = None,
    ) -> Context:
        """Create a Flower `Context` for a NodeApp callback.

        Parameters
        ----------
        run_id : int, optional
            The run ID, by default 0.
        state : Optional[RecordDict], optional
            The state to include in the context, by default None.
        run_config : Optional[ConfigRecord], optional
            The run configuration to include in the context, by default None.

        Returns
        -------
        Context
            A Flower `Context` instance containing the provided configuration and state.
        """
        resolved_node_config: UserConfig = self.data_config
        if self.id is not None:
            resolved_node_config["node-id"] = self.id

        return Context(
            run_id=run_id,
            node_id=0,
            node_config=resolved_node_config,
            state=state or RecordDict(),
            run_config=dict(run_config or self.train_config),
        )

    @staticmethod
    def _parse_run_config(
        run_config: DLRunConfig | Dict[str, Any] | None,
    ) -> DLRunConfig:
        """Parse a DLRunConfig from a dict or return it if already a DLRunConfig.

        Parameters
        ----------
        run_config : DLRunConfig | Dict[str, Any] | None
            The run configuration to parse. Can be a DLRunConfig instance, a dict containing DLRunConfig fields, or None.

        Returns
        -------
        DLRunConfig
            A parsed DLRunConfig instance.
        """
        if run_config is None:
            return DLRunConfig(rounds=1)

        if isinstance(run_config, DLRunConfig):
            return run_config

        parsed_run_config = dict(run_config)
        parsed_run_config.setdefault("rounds", 1)

        protocol = parsed_run_config.get("protocol")
        if isinstance(protocol, str):
            parsed_run_config["protocol"] = Mode(protocol.upper())

        allowed_fields = {field.name for field in dataclasses.fields(DLRunConfig)}
        filtered_run_config = {
            key: value
            for key, value in parsed_run_config.items()
            if key in allowed_fields
        }

        return DLRunConfig(**filtered_run_config)

    @staticmethod
    def create_aggregate_request(
        action: Action,
        source_node_id: str,
        round_number: int,
        msg: Message | None = None,
    ) -> AggregateRequest:
        """Create an `AggregateRequest` for peer-to-peer communication.

        Parameters
        ----------
        action : Action
            The action to perform.
        source_node_id : str
            The ID of the source node.
        round_number : int
            The round number.
        msg : Message | None, optional
            The message to include, by default None.

        Returns
        -------
        AggregateRequest
            The created aggregate request.
        """
        return AggregateRequest(
            action=action,
            source_node_id=source_node_id,
            round_number=round_number,
            msg=msg,
        )

    @staticmethod
    def _supports_message_context_signature(callback: NodeFn) -> bool:
        """Return whether `callback` looks like a ClientApp-style function."""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return False

        positional_params = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        return len(positional_params) >= 2 and positional_params[1].name in {
            "context",
            "ctxt",
            "ctx",
        }

    def _call_fn(
        self,
        callback: NodeFn,
        message: Message,
        context: Context,
    ) -> Message:
        """Call callback using the ClientApp-style signature."""
        return callback(message, context)

    @staticmethod
    def _legacy_callback_kwargs(
        callback: NodeFn,
        *,
        message: str,
        node_id: Optional[str],
        run_config: ConfigRecord,
        subject: str,
        app: "NodeApp",
    ) -> dict[str, Any]:
        """Return only keyword arguments accepted by a legacy callback."""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            return {
                "message": message,
                "node_id": node_id,
                "run_config": dict(run_config),
                "subject": subject,
                "app": app,
            }

        accepted_kwargs: dict[str, Any] = {}
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        candidate_kwargs = {
            "message": message,
            "node_id": node_id,
            "run_config": dict(run_config),
            "subject": subject,
            "app": app,
        }

        if accepts_var_kwargs:
            return candidate_kwargs

        for name, value in candidate_kwargs.items():
            param = signature.parameters.get(name)
            if param is not None and param.kind != inspect.Parameter.POSITIONAL_ONLY:
                accepted_kwargs[name] = value

        return accepted_kwargs

    def _invoke_callback(
        self,
        callback: NodeFn,
        *,
        message_type: str,
        config: ConfigRecord,
        context: Optional[Context] = None,
        payload: str | None = None,
        node_id: Optional[str] = None,
    ) -> Message:
        """Invoke a callback with new-style or legacy arguments.

        If the callback supports the new signature (message, context), it will be called with a
        constructed Message and Context. Otherwise, it will be called with the legacy signature
        (message, node_id, run_config, subject, app).

        Parameters
        ----------
        callback : NodeFn
            The callback function to invoke.
        message_type : str
            The type of message to create if using the new signature.
        config : ConfigRecord
            The configuration to include in the message if using the new signature.
        context : Optional[Context], optional
            The context to pass if using the new signature. If None, a context will be created from the config.
        payload : Optional[str], optional
            The raw message payload to pass if using the legacy signature, by default None.
        node_id : Optional[str], optional
            The node ID to pass if using the legacy signature, by default None.

        Returns
        -------
        Message
            The message returned by the callback, or a newly created message if the callback uses the legacy signature.
        """
        if self._supports_message_context_signature(callback):
            msg = self.create_message(
                message_type=message_type,
                config=config,
            )
            ctxt = context or self.create_context(
                run_config=config,
            )
            return self._call_fn(callback, msg, ctxt)

        legacy_message = payload if payload is not None else ""
        callback_kwargs = self._legacy_callback_kwargs(
            callback,
            message=legacy_message,
            node_id=node_id,
            run_config=config,
            subject=self.subject,
            app=self,
        )
        callback_result = callback(  # type: ignore[misc]
            **callback_kwargs,
        )
        if isinstance(callback_result, Message):
            return callback_result

        return self.create_message(
            message_type=message_type,
            config=config,
        )

    @staticmethod
    def aggregate_request_from_str(message: str) -> AggregateRequest:
        """Deserialize a JSON string into an :class:`AggregateRequest`.

        Parameters
        ----------
        message : str
            JSON-encoded string produced by :meth:`aggregate_request_to_str`.

        Returns
        -------
        AggregateRequest
            The reconstructed aggregate request.
        """
        data = json.loads(message)
        msg: Message | None = None
        if data.get("msg") is not None:
            object_contents = {k: base64.b64decode(v) for k, v in data["msg"].items()}
            msg = inflate_object_from_contents(  # type: ignore[assignment]
                data["msg_root_id"], object_contents
            )
        return AggregateRequest(
            action=Action(data["action"]),
            source_node_id=str(data["source_node_id"]),
            round_number=int(data["round_number"]),
            msg=msg,
        )

    @staticmethod
    def aggregate_request_to_str(msg: AggregateRequest) -> str:
        """Serialize an :class:`AggregateRequest` to a JSON string.

        Parameters
        ----------
        msg : AggregateRequest
            The aggregate request to serialize.

        Returns
        -------
        str
            JSON-encoded representation of *msg*.
        """
        payload: dict = {
            "action": msg.action.value,
            "source_node_id": msg.source_node_id,
            "round_number": msg.round_number,
            "msg": None,
            "msg_root_id": None,
        }
        if msg.msg is not None:
            all_contents = NodeApp._collect_deflated(msg.msg)
            payload["msg"] = {
                k: base64.b64encode(v).decode() for k, v in all_contents.items()
            }
            payload["msg_root_id"] = msg.msg.object_id
        return json.dumps(payload)

    @staticmethod
    def _collect_deflated(obj: InflatableObject) -> dict[str, bytes]:
        """Recursively collect deflated bytes for *obj* and all its descendants."""
        result: dict[str, bytes] = {obj.object_id: obj.deflate()}
        if obj.children:
            for child in obj.children.values():
                result.update(NodeApp._collect_deflated(child))
        return result

    def _is_supported_action(self, action: Action) -> bool:
        """Return whether the incoming action is compatible with protocol."""
        if self.run_config.protocol == Mode.PUSH:
            return action == Action.PUSH

        return action in {Action.PUSHPULL, Action.PUSH, Action.CANCEL}

    def _is_valid_round(self, request: AggregateRequest) -> bool:
        """Return whether the request carries a valid (positive) round number.

        Rounds are not required to be synchronized across nodes; any round
        number ≥ 1 is accepted.
        """
        if request.round_number < 1:
            log(
                logging.WARNING,
                f"[NodeApp:{self.name}] Ignoring request with invalid round {request.round_number} from {request.source_node_id}.",
            )
            return False

        return True

    def _is_duplicate_request(self, request: AggregateRequest) -> bool:
        """Return whether request was already handled in this round."""
        min_round_to_keep = max(self.current_round - 1, 1)
        if self._handled_requests:
            self._handled_requests = {
                key for key in self._handled_requests if key[2] >= min_round_to_keep
            }

        msg_key = request.msg.object_id if request.msg is not None else ""
        request_key = (
            request.source_node_id,
            f"{request.action.value}:{msg_key}",
            request.round_number,
        )
        if request_key in self._handled_requests:
            log(
                logging.INFO,
                f"[NodeApp:{self.name}] Ignoring duplicate {request.action.value} request from {request.source_node_id} for round {request.round_number}.",
            )
            return True

        self._handled_requests.add(request_key)
        return False

    def handle_message(self, message: str, node_id: Optional[str] = None) -> None:
        """Route incoming message to the registered `train` or `evaluate` callback.

        This method first attempts to parse the message as an `AggregateRequest` for peer-to-peer
        communication. If successful, it validates the request and routes it to the appropriate
        handler based on the action and protocol. If the message is not an `AggregateRequest`,
        it falls back to parsing it as a regular event message (e.g., "train" or "evaluate") and
        invokes the corresponding callback.

        Parameters
        ----------
        message : str
            The raw message payload received by the NodeApp.
        node_id : Optional[str], optional
            The ID of the node that sent the message, if available. This is used for legacy callback signatures that require node_id.
        """
        parsed_aggregate_message: AggregateRequest | None = None
        try:
            parsed_aggregate_message = self.aggregate_request_from_str(message)
        except Exception as exc:
            LOGGER.debug(
                "Failed to parse message as AggregateRequest: %s. Error: %s",
                message,
                exc,
            )

        if parsed_aggregate_message is not None:
            msg = parsed_aggregate_message
            if not msg.source_node_id:
                log(
                    logging.WARNING,
                    f"[NodeApp:{self.name}] Ignoring request with empty source node id.",
                )
                return

            if not self._is_supported_action(msg.action):
                log(
                    logging.WARNING,
                    f"[NodeApp:{self.name}] Ignoring unsupported action {msg.action.value} for protocol {self.run_config.protocol.value}.",
                )
                return

            if not self._is_valid_round(msg) or self._is_duplicate_request(msg):
                return

            if self.run_config.protocol == Mode.PUSH:
                self._update_own_parameters(msg)
                return

            handlers = {
                Action.PUSHPULL: self._handle_pushpull_request,
                Action.PUSH: self._handle_push_request,
                Action.CANCEL: self._handle_cancel_request,
            }

            handler = handlers.get(msg.action)
            if handler is not None:
                handler(msg)
                return
            return

    def periodic_run(self, view: list[str], node_id: Optional[str] = None) -> None:
        """Default periodic behavior: call registered `train` callback if present.

        This method is called by the nodemanager framework at intervals defined by `elapsed_time`.
        It determines the current round and step within the round based on `count_steps` and the
        configured number of steps per round. If it's the first step of a new round, it invokes
        the `_training_round` method to execute the training logic. For subsequent steps within
        the same round, it calls `_try_communication` to potentially send messages to peers based
        on the current view and communication probability defined in the run configuration.

        Parameters
        ----------
        view : list[str]
            List of peer node IDs currently in view for communication.
        node_id : Optional[str], optional
            The ID of this node, if available. This can be used for logging or callback invocation
        """

        if self.node_name is None:
            self.node_name = self.id
        round_num = self.count_steps // self.run_config.get_steps_per_round() + 1
        step_in_round = self.count_steps % self.run_config.get_steps_per_round()

        if step_in_round == 0:
            self.current_round = round_num
            self._training_round(self.current_round)
        else:
            self._try_communication(
                view=view, round_num=self.current_round, step_in_round=step_in_round
            )

        self.count_steps += 1

    def _training_round(self, round_num: int) -> None:
        """Helper to execute logic at the end of each training round."""
        if round_num > 1:
            round_i = round_num - 1
            if self._evaluate_fn is not None:
                log(
                    logging.INFO,
                    f"[NodeApp:{self.name}] Starting evaluation for round {round_i}",
                )
                ctxt = self.create_context(
                    run_config=self.eval_config,
                )
                msg = self._invoke_callback(
                    self._evaluate_fn,
                    message_type=MessageType.EVALUATE,
                    config=self.eval_config,
                    context=ctxt,
                )

                self.eval_metrics = msg.content.get("metrics", MetricRecord())
                log(
                    logging.INFO,
                    f"[NodeApp:{self.name}] Completed evaluation for round {round_i} with metrics: {self.eval_metrics}",
                )
        if round_num > self.run_config.rounds:
            log(
                logging.INFO,
                f"[NodeApp:{self.name}] Reached max rounds ({self.run_config.rounds}). Stopping training.",
            )
            return

        try:
            if self._train_fn is not None:
                ctxt = self.create_context(
                    run_config=self.train_config,
                )
                msg = self._invoke_callback(
                    self._train_fn,
                    message_type=MessageType.TRAIN,
                    config=self.train_config,
                    context=ctxt,
                )
                self._store_arrays_from_message(msg)
                self.train_metrics = msg.content.get("metrics", MetricRecord())
                log(
                    logging.INFO,
                    f"[NodeApp:{self.name}] Completed training round {round_num} with metrics: {self.train_metrics}",
                )
        except Exception as exc:
            log(
                logging.ERROR,
                f"[NodeApp:{self.name}] Error during training round {round_num}: {exc}",
            )

    def _try_communication(
        self, view: list[str], round_num: int, step_in_round: int
    ) -> None:
        """Helper to execute communication logic during training rounds."""
        # This method can be extended to include logic for sending/receiving
        # messages to/from peers based on the current view, round number, and
        # step within the round. For example, it could implement a PUSHPULL
        # protocol by sending parameters to a subset of peers at certain steps.
        if not view:
            log(
                logging.WARNING,
                f"[NodeApp:{self.name}] No peers in view for communication at round {round_num}, step {step_in_round}",
            )
            return
        try:
            if random.random() < self.run_config.communication_probability:

                request = self.create_aggregate_request(
                    action=(
                        Action.PUSHPULL
                        if self.run_config.protocol == Mode.PUSHPULL
                        else Action.PUSH
                    ),
                    source_node_id=self.id,
                    round_number=round_num,
                    msg=self.create_message(
                        message_type=MessageType.TRAIN,
                        config=self.train_config,
                    ),
                )

                n_nodes_to_share = self.run_config.n_nodes_to_share or 1
                peers_to_contact = random.sample(view, min(n_nodes_to_share, len(view)))

                log(
                    logging.INFO,
                    f"Node {self.id} retrieved list of nodes {peers_to_contact} "
                    f"at round {round_num} - Communication step "
                    f"{step_in_round}/{self.run_config.n_aggregation_steps}.",
                )

                for peer_id in peers_to_contact:
                    self._send_request(request, destination=peer_id)

        except Exception as exc:
            log(
                logging.ERROR,
                f"[NodeApp:{self.name}] Communication attempt failed at round {round_num}, step {step_in_round}: {exc}",
            )

    @staticmethod
    def _average_array_records(local: ArrayRecord, peer: ArrayRecord) -> ArrayRecord:
        """Return an equal-weight average of two ``ArrayRecord``s.

        This is a lightweight P2P-specific helper that avoids the central-FL
        validation logic inside ``FedAvg.aggregate_train`` (which requires
        exactly one ``MetricRecord`` with a ``num-examples`` key – constraints
        that do not apply in a gossip/push-pull setting).
        """
        averaged: dict[str, Array] = {}
        for key in set(local.keys()) | set(peer.keys()):
            local_arr = local.get(key)
            peer_arr = peer.get(key)

            if local_arr is not None and peer_arr is not None:
                avg_np = (local_arr.numpy() + peer_arr.numpy()) * 0.5
                averaged[key] = Array(np.asarray(avg_np))
            elif local_arr is not None:
                averaged[key] = local_arr
            elif peer_arr is not None:
                averaged[key] = peer_arr
        return ArrayRecord(averaged)

    def _update_own_parameters(self, request: AggregateRequest) -> None:
        """Merge local weights with the peer's weights using equal-weight averaging."""
        log(
            logging.INFO,
            f"[NodeApp:{self.name}] Updating parameters from message by {request.source_node_id}",
        )
        if request.msg is None:
            return

        peer_record = request.msg.content
        peer_arrays = peer_record.array_records.get(self.strategy.arrayrecord_key)
        if peer_arrays is None:
            log(
                logging.WARNING,
                f"[NodeApp:{self.name}] Peer message from {request.source_node_id} "
                "has no ArrayRecord — skipping parameter update.",
            )
            return

        self.arrays = self._average_array_records(self.arrays, peer_arrays)

    def _store_arrays_from_message(self, message: Message) -> None:
        """Update local arrays from callback result when arrays are returned."""
        returned_arrays = message.content.array_records.get(self.strategy.arrayrecord_key)
        if returned_arrays is not None:
            self.arrays = returned_arrays

    def _handle_pushpull_request(self, request: AggregateRequest) -> None:
        """Handle incoming PUSHPULL request."""
        if not self.waiting_for_reply_from:
            agg_request = self.create_aggregate_request(
                action=Action.PUSH,
                source_node_id=self.id,
                round_number=self.current_round if self.current_round > 0 else 1,
                msg=self.create_message(
                    message_type=MessageType.TRAIN,
                    config=self.train_config,
                ),
            )
            self._send_request(agg_request, destination=request.source_node_id)
            self._update_own_parameters(request)
            return

        if request.source_node_id in self.waiting_for_reply_from:
            self._update_own_parameters(request)
            self.waiting_for_reply_from.remove(request.source_node_id)
            return

        agg_request = self.create_aggregate_request(
            action=Action.CANCEL,
            source_node_id=self.id,
            round_number=self.current_round,
        )
        self._send_request(agg_request, destination=request.source_node_id)

    def _handle_push_request(self, request: AggregateRequest) -> None:
        """Handle incoming PUSH request."""
        if request.source_node_id in self.waiting_for_reply_from:
            self._update_own_parameters(request)
            self.waiting_for_reply_from.remove(request.source_node_id)

    def _handle_cancel_request(self, request: AggregateRequest) -> None:
        """Handle incoming CANCEL request."""
        if request.source_node_id in self.waiting_for_reply_from:
            self.waiting_for_reply_from.remove(request.source_node_id)

    def _send_request(self, request: AggregateRequest, destination: str) -> None:
        """Helper to send requests to peers."""
        self.send_message(
            message=self.aggregate_request_to_str(request),
            destination=destination,
            timeout=self.timeout,
        )

        if (
            self.run_config.protocol == Mode.PUSHPULL
            and request.action == Action.PUSHPULL
        ):
            self.waiting_for_reply_from.add(destination)


def load_nodeapp_configs_from_pyproject(
    pyproject_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Load per-subject NodeApp configurations from `pyproject.toml`.

    Supported layout:

    .. code-block:: toml

        [tool.flwr.nodeapp]
        default-timeout = 30

        [tool.flwr.nodeapp.apps.trainer]
        subject = "trainer"
        timeout = 20
        train = "my_pkg.node_apps:train_trainer"
        evaluate = "my_pkg.node_apps:evaluate_trainer"

        [tool.flwr.nodeapp.apps.trainer.config]
        local-epochs = 2

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping `{subject: {"timeout": int, "run_config": dict}}`.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[import-untyped,no-redef]

    path = Path(pyproject_path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    tool = data.get("tool", {})
    flwr = tool.get("flwr", {}) if isinstance(tool, dict) else {}
    nodeapp = flwr.get("nodeapp", {}) if isinstance(flwr, dict) else {}
    if not isinstance(nodeapp, dict):
        return {}

    default_timeout = int(nodeapp.get("default-timeout", 30))
    apps = nodeapp.get("apps", {})
    if not isinstance(apps, dict):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for app_key, app_cfg in apps.items():
        if not isinstance(app_cfg, dict):
            continue
        subject = str(app_cfg.get("subject", app_key))
        timeout = int(app_cfg.get("timeout", default_timeout))
        run_config = app_cfg.get("config", {})
        if not isinstance(run_config, dict):
            run_config = {}

        result[subject] = {
            "timeout": timeout,
            "run_config": run_config,
            "train": app_cfg.get("train"),
            "evaluate": app_cfg.get("evaluate"),
        }

    return result


def _load_handler(spec: str) -> NodeFn:
    """Load handler callable from a `<module>:<symbol>` spec."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid handler mapping '{spec}'. Expected '<module>:<function>'."
        )
    module_name, symbol_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    handler = getattr(module, symbol_name, None)
    if handler is None or not callable(handler):
        raise ValueError(
            f"Handler '{symbol_name}' not found or not callable in '{module_name}'."
        )
    return handler


def _load_object(spec: str) -> Any:
    """Load object from a `<module>:<symbol>` spec."""
    if ":" not in spec:
        raise ValueError(
            f"Invalid object mapping '{spec}'. Expected '<module>:<symbol>'."
        )

    module_name, symbol_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    obj = getattr(module, symbol_name, None)
    if obj is None:
        raise ValueError(f"Object '{symbol_name}' not found in '{module_name}'.")
    return obj


def _load_nodeapps_from_components(
    pyproject_path: Union[str, Path],
) -> Dict[str, NodeApp]:
    """Load NodeApp objects from `[tool.flwr.app.components]` nodeapp entries.

    Expected layout:

    .. code-block:: toml

        [tool.flwr.app.components]
        nodeapp1 = "my_pkg.node_app:app1"
        nodeapp2 = "my_pkg.node_app:app2"
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[import-untyped,no-redef]

    path = Path(pyproject_path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    tool = data.get("tool", {})
    flwr = tool.get("flwr", {}) if isinstance(tool, dict) else {}
    app = flwr.get("app", {}) if isinstance(flwr, dict) else {}
    components = app.get("components", {}) if isinstance(app, dict) else {}

    if not isinstance(components, dict):
        return {}

    result: Dict[str, NodeApp] = {}
    for component_name, spec in components.items():
        if not str(component_name).startswith("nodeapp"):
            continue
        if not isinstance(spec, str):
            continue

        loaded = _load_object(spec)
        if not isinstance(loaded, NodeApp):
            raise ValueError(
                f"Component '{component_name}' must reference a NodeApp instance. "
                f"Got: {type(loaded).__name__}"
            )
        result[str(component_name)] = loaded

    return result


def create_nodeapps_from_pyproject(
    pyproject_path: Union[str, Path],
) -> Dict[str, NodeApp]:
    """Create NodeApp instances from `pyproject.toml` configuration.

    Returns
    -------
    Dict[str, NodeApp]
        Mapping `{subject: NodeApp}`.
    """
    # Preferred mode: load pre-built NodeApp objects from
    # [tool.flwr.app.components] entries named nodeapp*.
    component_apps = _load_nodeapps_from_components(pyproject_path)
    if component_apps:
        return component_apps

    # Backward-compatible mode: construct NodeApp objects from
    # [tool.flwr.nodeapp.apps.*] mapping.
    configs = load_nodeapp_configs_from_pyproject(pyproject_path)
    apps: Dict[str, NodeApp] = {}
    for subject, cfg in configs.items():
        run_config_from_file = cfg.get("run_config", {})
        if not isinstance(run_config_from_file, dict):
            run_config_from_file = {}

        dl_fields = {field.name for field in dataclasses.fields(DLRunConfig)}
        dl_run_config = {
            key: value
            for key, value in run_config_from_file.items()
            if key in dl_fields
        }
        dl_run_config.setdefault("rounds", 1)
        train_config = {
            key: value
            for key, value in run_config_from_file.items()
            if key not in dl_fields
        }

        app = NodeApp(
            subject=subject,
            initial_arrays=ArrayRecord({}),
            timeout=int(cfg.get("timeout", 30)),
            run_config=dl_run_config,
            train_config=ConfigRecord(train_config),
        )

        train_spec = cfg.get("train")
        if isinstance(train_spec, str) and train_spec:
            app.train()(_load_handler(train_spec))

        evaluate_spec = cfg.get("evaluate")
        if isinstance(evaluate_spec, str) and evaluate_spec:
            app.evaluate()(_load_handler(evaluate_spec))

        apps[subject] = app

    return apps
