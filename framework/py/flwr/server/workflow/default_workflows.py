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
"""Legacy default workflows."""


import io
import timeit
from logging import INFO, WARN
from typing import cast

import flwr.common.recorddict_compat as compat
from flwr.app.message_type import MessageType
from flwr.common import (
    ArrayRecord,
    Code,
    ConfigRecord,
    Context,
    EvaluateRes,
    FitRes,
    GetParametersIns,
    Message,
    log,
)
from flwr.common.constant import MessageTypeLegacy

from ..client_proxy import ClientProxy
from ..compat.app_utils import start_update_client_manager_thread
from ..compat.legacy_context import LegacyContext
from ..grid import Grid
from ..typing import Workflow
from .constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD, Key


class DefaultWorkflow:
    """Default workflow in Flower."""

    def __init__(
        self,
        fit_workflow: Workflow | None = None,
        evaluate_workflow: Workflow | None = None,
    ) -> None:
        if fit_workflow is None:
            fit_workflow = default_fit_workflow
        if evaluate_workflow is None:
            evaluate_workflow = default_evaluate_workflow
        self.fit_workflow: Workflow = fit_workflow
        self.evaluate_workflow: Workflow = evaluate_workflow

    def __call__(self, grid: Grid, context: Context) -> None:
        """Execute the workflow."""
        if not isinstance(context, LegacyContext):
            raise TypeError(
                f"Expect a LegacyContext, but get {type(context).__name__}."
            )

        # Start the thread updating nodes
        thread, f_stop, c_done = start_update_client_manager_thread(
            grid, context.client_manager
        )

        # Wait until the node registration done
        c_done.wait()

        # Initialize parameters
        log(INFO, "[INIT]")
        default_init_params_workflow(grid, context)

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()
        cfg = ConfigRecord()
        cfg[Key.START_TIME] = start_time
        context.state.config_records[MAIN_CONFIGS_RECORD] = cfg

        for current_round in range(1, context.config.num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            cfg[Key.CURRENT_ROUND] = current_round

            # Fit round
            self.fit_workflow(grid, context)

            # Centralized evaluation
            default_centralized_evaluation_workflow(grid, context)

            # Evaluate round
            self.evaluate_workflow(grid, context)

        # Bookkeeping and log results
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        hist = context.history
        log(INFO, "")
        log(INFO, "[SUMMARY]")
        log(
            INFO,
            "Run finished %s round(s) in %.2fs",
            context.config.num_rounds,
            elapsed,
        )
        for idx, line in enumerate(io.StringIO(str(hist))):
            if idx == 0:
                log(INFO, "%s", line.strip("\n"))
            else:
                log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        # Terminate the thread
        f_stop.set()
        thread.join()


def default_init_params_workflow(grid: Grid, context: Context) -> None:
    """Execute the default workflow for parameters initialization."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    parameters = context.strategy.initialize_parameters(
        client_manager=context.client_manager
    )
    if parameters is not None:
        log(INFO, "Using initial global parameters provided by strategy")
        arr_record = compat.parameters_to_arrayrecord(parameters, keep_input=True)
    else:
        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = context.client_manager.sample(1)[0]
        # Send GetParametersIns and get the response
        content = compat.getparametersins_to_recorddict(GetParametersIns({}))
        messages = grid.send_and_receive(
            [
                Message(
                    content=content,
                    dst_node_id=random_client.node_id,
                    message_type=MessageTypeLegacy.GET_PARAMETERS,
                    group_id="0",
                )
            ]
        )
        msg = list(messages)[0]

        if (
            msg.has_content()
            and compat._extract_status_from_recorddict(  # pylint: disable=W0212
                "getparametersres", msg.content
            ).code
            == Code.OK
        ):
            log(INFO, "Received initial parameters from one random client")
            arr_record = next(iter(msg.content.array_records.values()))
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
            arr_record = ArrayRecord()

    context.state.array_records[MAIN_PARAMS_RECORD] = arr_record

    # Evaluate initial parameters
    log(INFO, "Starting evaluation of initial global parameters")
    parameters = compat.arrayrecord_to_parameters(arr_record, keep_input=True)
    res = context.strategy.evaluate(0, parameters=parameters)
    if res is not None:
        log(
            INFO,
            "initial parameters (loss, other metrics): %s, %s",
            res[0],
            res[1],
        )
        context.history.add_loss_centralized(server_round=0, loss=res[0])
        context.history.add_metrics_centralized(server_round=0, metrics=res[1])
    else:
        log(INFO, "Evaluation returned no results (`None`)")


def default_centralized_evaluation_workflow(_: Grid, context: Context) -> None:
    """Execute the default workflow for centralized evaluation."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    # Retrieve current_round and start_time from the context
    cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
    current_round = cast(int, cfg[Key.CURRENT_ROUND])
    start_time = cast(float, cfg[Key.START_TIME])

    # Centralized evaluation
    parameters = compat.arrayrecord_to_parameters(
        record=context.state.array_records[MAIN_PARAMS_RECORD],
        keep_input=True,
    )
    res_cen = context.strategy.evaluate(current_round, parameters=parameters)
    if res_cen is not None:
        loss_cen, metrics_cen = res_cen
        log(
            INFO,
            "fit progress: (%s, %s, %s, %s)",
            current_round,
            loss_cen,
            metrics_cen,
            timeit.default_timer() - start_time,
        )
        context.history.add_loss_centralized(server_round=current_round, loss=loss_cen)
        context.history.add_metrics_centralized(
            server_round=current_round, metrics=metrics_cen
        )


def default_fit_workflow(grid: Grid, context: Context) -> None:  # pylint: disable=R0914
    """Execute the default workflow for a single fit round."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    # Get current_round and parameters
    cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
    current_round = cast(int, cfg[Key.CURRENT_ROUND])
    arr_record = context.state.array_records[MAIN_PARAMS_RECORD]
    parameters = compat.arrayrecord_to_parameters(arr_record, keep_input=True)

    # Get clients and their respective instructions from strategy
    client_instructions = context.strategy.configure_fit(
        server_round=current_round,
        parameters=parameters,
        client_manager=context.client_manager,
    )

    if not client_instructions:
        log(INFO, "configure_fit: no clients selected, cancel")
        return
    log(
        INFO,
        "configure_fit: strategy sampled %s clients (out of %s)",
        len(client_instructions),
        context.client_manager.num_available(),
    )

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Build out messages
    out_messages = [
        Message(
            content=compat.fitins_to_recorddict(fitins, True),
            dst_node_id=proxy.node_id,
            message_type=MessageType.TRAIN,
            group_id=str(current_round),
        )
        for proxy, fitins in client_instructions
    ]

    # Send instructions to clients and
    # collect `fit` results from all clients participating in this round
    messages = list(grid.send_and_receive(out_messages))
    del out_messages
    num_failures = len([msg for msg in messages if msg.has_error()])

    # No exception/failure handling currently
    log(
        INFO,
        "aggregate_fit: received %s results and %s failures",
        len(messages) - num_failures,
        num_failures,
    )

    # Aggregate training results
    results: list[tuple[ClientProxy, FitRes]] = []
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []
    for msg in messages:
        if msg.has_content():
            proxy = node_id_to_proxy[msg.metadata.src_node_id]
            fitres = compat.recorddict_to_fitres(msg.content, False)
            if fitres.status.code == Code.OK:
                results.append((proxy, fitres))
            else:
                failures.append((proxy, fitres))
        else:
            failures.append(Exception(msg.error))

    aggregated_result = context.strategy.aggregate_fit(current_round, results, failures)
    parameters_aggregated, metrics_aggregated = aggregated_result

    # Update the parameters and write history
    if parameters_aggregated:
        arr_record = compat.parameters_to_arrayrecord(parameters_aggregated, True)
        context.state.array_records[MAIN_PARAMS_RECORD] = arr_record
        context.history.add_metrics_distributed_fit(
            server_round=current_round, metrics=metrics_aggregated
        )


# pylint: disable-next=R0914
def default_evaluate_workflow(grid: Grid, context: Context) -> None:
    """Execute the default workflow for a single evaluate round."""
    if not isinstance(context, LegacyContext):
        raise TypeError(f"Expect a LegacyContext, but get {type(context).__name__}.")

    # Get current_round and parameters
    cfg = context.state.config_records[MAIN_CONFIGS_RECORD]
    current_round = cast(int, cfg[Key.CURRENT_ROUND])
    arr_record = context.state.array_records[MAIN_PARAMS_RECORD]
    parameters = compat.arrayrecord_to_parameters(arr_record, keep_input=True)

    # Get clients and their respective instructions from strategy
    client_instructions = context.strategy.configure_evaluate(
        server_round=current_round,
        parameters=parameters,
        client_manager=context.client_manager,
    )
    if not client_instructions:
        log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
        return
    log(
        INFO,
        "configure_evaluate: strategy sampled %s clients (out of %s)",
        len(client_instructions),
        context.client_manager.num_available(),
    )

    # Build dictionary mapping node_id to ClientProxy
    node_id_to_proxy = {proxy.node_id: proxy for proxy, _ in client_instructions}

    # Build out messages
    out_messages = [
        Message(
            content=compat.evaluateins_to_recorddict(evalins, True),
            dst_node_id=proxy.node_id,
            message_type=MessageType.EVALUATE,
            group_id=str(current_round),
        )
        for proxy, evalins in client_instructions
    ]

    # Send instructions to clients and
    # collect `evaluate` results from all clients participating in this round
    messages = list(grid.send_and_receive(out_messages))
    del out_messages
    num_failures = len([msg for msg in messages if msg.has_error()])

    # No exception/failure handling currently
    log(
        INFO,
        "aggregate_evaluate: received %s results and %s failures",
        len(messages) - num_failures,
        num_failures,
    )

    # Aggregate the evaluation results
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []
    for msg in messages:
        if msg.has_content():
            proxy = node_id_to_proxy[msg.metadata.src_node_id]
            evalres = compat.recorddict_to_evaluateres(msg.content)
            if evalres.status.code == Code.OK:
                results.append((proxy, evalres))
            else:
                failures.append((proxy, evalres))
        else:
            failures.append(Exception(msg.error))

    aggregated_result = context.strategy.aggregate_evaluate(
        current_round, results, failures
    )

    loss_aggregated, metrics_aggregated = aggregated_result

    # Write history
    if loss_aggregated is not None:
        context.history.add_loss_distributed(
            server_round=current_round, loss=loss_aggregated
        )
        context.history.add_metrics_distributed(
            server_round=current_round, metrics=metrics_aggregated
        )
