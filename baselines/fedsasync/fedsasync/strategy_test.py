import numpy as np
import pytest
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from unittest.mock import Mock, MagicMock
from .strategy import FedAvg, FedSaSync


def create_mock_messages(
    node_ids: list[int], 
    record: RecordDict | None = None, 
    message_type: str = "train",
) -> list[MagicMock]:
    """Generate a list of mock Message objects carrying the specified 
    RecordDict payload (or an empty one by default)."""
    if record is None:
        record = RecordDict({})

    messages = []
    for node_id in node_ids:
        msg = MagicMock()
        msg.content = record
        msg.message_type = message_type
        # Assign numeric destination node ID to avoid dictionary lookup issues with mocks
        msg.metadata.dst_node_id = node_id
        messages.append(msg)
    return messages


def create_mock_reply(
    arrays: ArrayRecord, 
    num_examples: float, 
    partition_id: int = 0,
    reply_to_msg_id: str | None = None,
    reply_src_node_id: str | None = None,
    server_round: str | int = "1",
) -> Message:
    """Create a mock reply Message object populated with standard training metrics 
    and metadata identifiers."""
    message = Mock(spec=Message)
    message.content = RecordDict(
        {
            "arrays": arrays,
            "metrics": MetricRecord({
                "num-examples": num_examples,
                "partition-id": partition_id,
            })
        }
    )
    message.has_error.side_effect = lambda: False
    message.has_content.side_effect = lambda: True
    
    message.metadata.reply_to_message_id = reply_to_msg_id
    message.metadata.src_node_id = reply_src_node_id
    message.metadata.group_id = str(server_round)
    return message


def test_m_equals_n_equivalence_with_fedavg() -> None:
    """Test that setting M = N in the semi-asynchronous strategy yields
    identical aggregation results and synchronous behavior compared to FedAvg."""
    N = 4
    M = N

    fedavg_strategy = FedAvg()
    fedsasync_strategy = FedSaSync()

    replies = [
        create_mock_reply(ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), 1, 0),
        create_mock_reply(ArrayRecord([np.array([1.0, 0.2, 0.5, 1.0])]), 5, 1),
        create_mock_reply(ArrayRecord([np.array([0.2, 0.5, 1.0, 0.2])]), 2, 2),
        create_mock_reply(ArrayRecord([np.array([0.5, 1.0, 0.2, 0.5])]), 9, 3),
    ]

    # Execute FedAvg baseline aggregation
    expected_fedavg_aggregated, _ = fedavg_strategy.aggregate_train(1, replies)

    # Execute FedSaSync communication and aggregation under strict synchrony (M = N)
    grid = MagicMock()
    grid.pull_messages.return_value = replies

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=[], sync_deg=M
    )
    actual_fedsasync_aggregated, _ = fedsasync_strategy.aggregate_train(1, actual_fedsasync_replies)

    # Assert mathematical equivalence between both strategies
    assert expected_fedavg_aggregated
    assert actual_fedsasync_aggregated
    expected_fedavg = expected_fedavg_aggregated.to_numpy_ndarrays()[0]
    actual_fedsasync = actual_fedsasync_aggregated.to_numpy_ndarrays()[0]
    np.testing.assert_equal(expected_fedavg, actual_fedsasync)


def test_m_less_than_n_exact_threshold() -> None:
    """Test that aggregation triggers successfully when exactly M replies are received."""
    N = 4
    M = 2  # Semiasynchronous threshold

    fedsasync_strategy = FedSaSync()

    exact_replies = [
        create_mock_reply(ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), 5, 0),
        create_mock_reply(ArrayRecord([np.array([3.0, 3.0, 3.0, 3.0])]), 5, 1),
    ]

    grid = MagicMock()
    grid.pull_messages.return_value = exact_replies

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=[], sync_deg=M
    )
    replies_list = list(actual_fedsasync_replies)
    
    # Assert that exactly M replies are collected
    assert len(replies_list) == M

    actual_fedsasync_aggregated, _ = fedsasync_strategy.aggregate_train(1, replies_list)
    
    # Assert correctness of weighted average results
    assert actual_fedsasync_aggregated
    expected_values = np.array([2.0, 2.0, 2.0, 2.0])
    np.testing.assert_equal(
        actual_fedsasync_aggregated.to_numpy_ndarrays()[0], expected_values
    )


def test_m_less_than_n_exceeds_threshold() -> None:
    """Test that if more than M replies arrive, the strategy handles overflow safely."""
    N = 4
    M = 2  # Semiasynchronous threshold

    fedsasync_strategy = FedSaSync()

    overflow_replies = [
        create_mock_reply(ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), 5, 0),
        create_mock_reply(ArrayRecord([np.array([3.0, 3.0, 3.0, 3.0])]), 5, 1),
        create_mock_reply(ArrayRecord([np.array([5.0, 5.0, 5.0, 5.0])]), 5, 2),  # Extra arriving reply
    ]

    grid = MagicMock()
    grid.pull_messages.return_value = overflow_replies

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=[], sync_deg=M
    )
    replies_list = list(actual_fedsasync_replies)
    
    # Assert that at least M replies are gathered
    assert len(replies_list) >= M

    actual_fedsasync_aggregated, _ = fedsasync_strategy.aggregate_train(1, replies_list)
    assert actual_fedsasync_aggregated


def test_final_round_synchronization() -> None:
    """Test that in the final training round, the strategy waits for all remaining nodes 
    regardless of the M threshold to ensure full synchronization."""
    N = 4
    M = 2
    fedsasync_strategy = FedSaSync()

    messages = create_mock_messages(node_ids=[f"node_{i}" for i in range(N)])
    fast_replies = [
        create_mock_reply(ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), 1, 0, "msg_0"),
        create_mock_reply(ArrayRecord([np.array([1.0, 0.2, 0.5, 1.0])]), 5, 1, "msg_1"),
    ]
    straggler_replies = [
        create_mock_reply(ArrayRecord([np.array([0.2, 0.5, 1.0, 0.2])]), 2, 2, "msg_2"),
        create_mock_reply(ArrayRecord([np.array([0.5, 1.0, 0.2, 0.5])]), 9, 3, "msg_3"),
    ]

    grid = MagicMock()
    grid.pull_messages.side_effect = [fast_replies, straggler_replies]
    grid.push_messages.return_value = [f"msg_{i}" for i in range(N)]

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=messages, sync_deg=M, last_round=True
    )
    replies_list = list(actual_fedsasync_replies)

    # Assert that all N clients are collected on the final round
    assert len(replies_list) == N

    actual_fedsasync_aggregated, _ = fedsasync_strategy.aggregate_train(1, replies_list)
    assert actual_fedsasync_aggregated


def test_m_not_reached_due_to_timeout() -> None:
    """Test that the strategy handles timeouts safely when the minimum threshold M 
    cannot be reached within the time limit."""
    N = 4
    M = 2  
    fedsasync_strategy = FedSaSync()

    grid = MagicMock()
    grid.pull_messages.return_value = []  # No messages arrive due to timeout
    grid.push_messages.return_value = [f"msg_{i}" for i in range(N)]

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=[], timeout=0.01, sync_deg=M
    )
    replies_list = list(actual_fedsasync_replies)
    
    # Assert that no replies are collected due to timeout expiration
    assert len(replies_list) == 0


def test_client_failure_handling() -> None:
    """Test that client replies containing runtime errors are automatically filtered out 
    without disrupting the overall aggregation process."""
    N = 4
    M = 2  
    fedsasync_strategy = FedSaSync()

    ok_reply_1 = create_mock_reply(ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), 5, 0, "msg_0")
    
    failed_reply = create_mock_reply(ArrayRecord([np.array([99.0, 99.0, 99.0, 99.0])]), 5, 1, "msg_1")
    failed_reply.has_error.side_effect = lambda: True
    failed_reply.metadata.src_node_id = 1 
    
    ok_reply_2 = create_mock_reply(ArrayRecord([np.array([3.0, 3.0, 3.0, 3.0])]), 5, 2, "msg_2")

    grid = MagicMock()
    grid.pull_messages.return_value = [failed_reply, ok_reply_1, ok_reply_2]
    grid.push_messages.return_value = [f"msg_{i}" for i in range(N)]

    actual_fedsasync_replies = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=[], sync_deg=M
    )
    replies_list = list(actual_fedsasync_replies)

    actual_fedsasync_aggregated, _ = fedsasync_strategy.aggregate_train(1, replies_list)
    
    # Assert successful aggregation despite the presence of a faulty reply
    assert actual_fedsasync_aggregated


def test_pending_message_handling() -> None:
    """Test that pending message tracking correctly maintains active stragglers 
    in msg_dict across rounds."""
    N = 4
    M = 2
    fedsasync_strategy = FedSaSync()

    replies = [
        create_mock_reply(ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), 1, 0, "msg_0"),
        create_mock_reply(ArrayRecord([np.array([1.0, 0.2, 0.5, 1.0])]), 5, 1, "msg_1"),
    ]

    messages = create_mock_messages(node_ids=[f"node_{i}" for i in range(N)])

    grid = MagicMock()
    grid.pull_messages.return_value = replies
    grid.push_messages.return_value = [f"msg_{i}" for i in range(N)]

    msg_dict = {}

    _ = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, messages=messages, msg_dict=msg_dict, sync_deg=M
    )

    # Assert that only non-responding nodes remain tracked as stragglers
    assert msg_dict == {
        "node_2": "msg_2",
        "node_3": "msg_3",
    }


def test_client_re_entry_with_stragglers_reaching_m() -> None:
    """Test that pre-existing stragglers combined with new replies successfully 
    satisfy the M synchronization threshold and clean up resolved entries."""
    N = 4
    M = 3
    fedsasync_strategy = FedSaSync(num_rounds=2)

    msg_dict = {
        "node_2": "msg_2",
        "node_3": "msg_3",
    }

    late_replies = [
        create_mock_reply(ArrayRecord([np.array([1.0, 1.0])]), 5, 2, "msg_2"),
        create_mock_reply(ArrayRecord([np.array([2.0, 2.0])]), 5, 3, "msg_3"),
    ]

    new_reply = [
        create_mock_reply(ArrayRecord([np.array([3.0, 3.0])]), 1, 0, "msg_4"),
    ]

    grid = MagicMock()
    grid.pull_messages.return_value = late_replies + new_reply
    grid.push_messages.return_value = ["msg_4", "msg_5", "msg_2", "msg_3"]

    messages = create_mock_messages(node_ids=[f"node_{i}" for i in range(N)])

    ret = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid, 
        messages=messages, 
        current_round=2, 
        msg_dict=msg_dict, 
        sync_deg=M
    )

    # Assert exact threshold compliance and successful integration of stragglers
    assert len(ret) == M

    reply_ids = [msg.metadata.reply_to_message_id for msg in ret]
    assert "msg_2" in reply_ids
    assert "msg_3" in reply_ids

    # Assert that resolved stragglers are properly purged from the pending dictionary
    assert "node_2" not in msg_dict
    assert "node_3" not in msg_dict


def test_staleness_metric_calculation() -> None:
    """Test that client staleness metrics are accurately calculated and stored 
    at the correct historical index upon delayed response arrival."""
    N = 2
    M = 1
    current_round = 4
    init_round = 2
    
    fedsasync_strategy = FedSaSync(num_rounds=5)

    late_reply = create_mock_reply(ArrayRecord([np.array([1.5])]), 5, 1, "msg_delayed_1", "node_1", init_round)

    grid = MagicMock()
    grid.pull_messages.return_value = [late_reply]
    grid.push_messages.return_value = ["msg_new_0", "msg_new_1"]

    _ = fedsasync_strategy.send_and_receive_semiasync(
        grid=grid,
        messages=[],
        current_round=current_round,
        sync_deg=M
    )

    expected_staleness = current_round - init_round  # 4 - 1 = 3
    stored_staleness = fedsasync_strategy.client_staleness[1][current_round - 1]
    
    # Assert correct computation and placement of the staleness metric
    assert stored_staleness == expected_staleness