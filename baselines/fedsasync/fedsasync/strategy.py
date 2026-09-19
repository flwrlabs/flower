"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""
import io
import random
import time
from collections.abc import Callable, Iterable
from logging import INFO
from typing import Dict

from flwr.app import MessageType
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from .utils import save_logs


class FedSaSync(FedAvg):
    """Federated Semi-Asynchronous strategy.

    Implementation based on Hidalgo-Izquierdo et al. (2026), arXiv:2606.24230.

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    strategy_name : str (default: "FedAvg")
        Name of the strategy.
    semiasync_deg : int (default: 10)
        Degree of semi-asynchrony.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        *,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn=None,
        evaluate_metrics_aggr_fn=None,

        # Additional parameters for FedSaSync if needed
        strategy_name: str = "FedAvg",
        semiasync_deg: int = 10,
        number_slow: int = 0,
        dataset_name: str = "uoft-cs/cifar10",

    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )

        # Additional initialization for FedSaSync if needed
        self.strategy_name = strategy_name
        self.semiasync_deg = semiasync_deg
        self.number_slow = number_slow
        self.dataset_name = dataset_name


    def sample_nodes_semiasync(
        self, grid: Grid, msg_dict: Dict[str, str], sample_size: int
    ) -> tuple[list[int], list[int]]:
        """Sample semi-asynchronously the specified number of nodes using the Grid.

        Parameters
        ----------
        grid : Grid
            The grid object.
        msg_dict : Dict[str, str]
            A dictionary mapping node IDs (as strings) to in-flight message IDs.
        sample_size : int
            The number of nodes to sample.

        Returns
        -------
        tuple[list[int], list[int]]
            A tuple containing the sampled node IDs and the list
            of all connected node IDs.
        """
        all_nodes = list(grid.get_node_ids())
        # Nodes that are currently running (have an in-flight message)
        running_nodes = [int(node_id) for node_id in msg_dict]
        free_nodes = sorted(set(all_nodes) - set(running_nodes))

        rng = random.Random(42)
        sampled_nodes = rng.sample(free_nodes, min(len(free_nodes), sample_size))
        return sampled_nodes, all_nodes


    # Overwrite of FedAvg configure_train to implement semi-asynchronous sampling
    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
        msg_dict: Dict[str, str] | None = None
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []
        # Sample nodes semiasynchronously, based on current execution state of nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)

        if msg_dict is None:
            msg_dict = {}
        node_ids, num_total = self.sample_nodes_semiasync(grid, msg_dict, sample_size)

        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)


    def send_and_receive_semiasync(
        self,
        grid: Grid,
        messages: Iterable[Message],
        timeout: float | None = None,
        msg_dict: Dict[str, str] | None = None,
        sync_deg: int = 1,
        last_round: bool = False,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull semiasynchronously 'M' reply
        messages.

        This method sends a list of messages to their destination node IDs and then
        waits for 'M' replies. It continues to pull replies until either M replies are
        received or the specified timeout duration is exceeded.
        """
        # Push messages
        messages = list(messages)
        msg_ids = list(grid.push_messages(messages))

        # Register busy nodes in msg_dict
        if msg_dict is None:
            msg_dict = {}

        for msg_id, msg in zip(msg_ids, messages):
            node_id = str(msg.metadata.dst_node_id)
            msg_dict[node_id] = msg_id  # node_id: msg_id
        del messages

        # Debug mode: print the msg_dict after pushing messages
        # print("msg_dict:", msg_dict)

        # Pull messages
        # Get all message IDs that are currently running
        all_msg_ids = set(msg_dict.values())
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: list[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = grid.pull_messages(all_msg_ids)  # Pull all messages in grid
            ret.extend(res_msgs)
            all_msg_ids.difference_update(
                {msg.metadata.reply_to_message_id for msg in res_msgs}
            )
            # Round end condition
            if not last_round:
                if len(ret) >= sync_deg or len(all_msg_ids) == 0:
                    break
            else:
                if len(all_msg_ids) == 0:
                    break
            # Sleep
            time.sleep(3)

        # Update msg_dict to remove unnecessary entries
        for node_id in list(msg_dict.keys()):
            if msg_dict[node_id] not in all_msg_ids:
                del msg_dict[node_id]

        # Debug mode: print the msg_dict after pulling messages
        # print("msg_dict after pulling:", msg_dict)
        return ret


    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.strategy_name)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        # List of messages running in grid
        msg_dict: Dict[str, str] = {}

        # Select sync_deg based on strategy name
        train_clients = max(
            int(self.fraction_train * int(len(list(grid.get_node_ids())))),
            self.min_train_nodes,
        )

        if self.strategy_name == "FedSaSync":
            # For FedSaSync, sync_deg is determined by the semiasync_deg parameter (semi-async)
            sync_deg = min(self.semiasync_deg, train_clients)
        else:
            # For any strategy, sync_deg is equal to the number of training clients (fully sync)
            sync_deg = train_clients

        # print("Sync degree:", sync_deg)

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)
            last_round = current_round == num_rounds
            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = self.send_and_receive_semiasync(
                grid=grid,
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                    msg_dict,
                ),
                timeout=timeout,
                msg_dict=msg_dict,
                sync_deg=sync_deg,
                last_round=last_round,
            )

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )
            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )
            if agg_evaluate_metrics is None:
                agg_evaluate_metrics = MetricRecord()
            agg_evaluate_metrics["time"] = time.time() - t_start

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        # Call utility function to save logs
        save_logs(
            result,
            self.strategy_name,
            self.semiasync_deg,
            self.number_slow,
            self.dataset_name
        )
        return result
