"""Unit tests for decentralized node orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from flwr.decentralized.node import DNode, start_node


def test_dnode_dynamic_without_sampling_logs_warning_and_sets_no_config_path() -> None:
    """Dynamic topology without sampling should continue with `config_path=None`."""
    dynamic_mode = object()

    with patch("flwr.decentralized.node.TopologyMode") as topology_mode, patch(
        "flwr.decentralized.node.Node.__init__",
        return_value=None,
    ) as node_init, patch("flwr.decentralized.node.log") as logger:
        topology_mode.dynamic.return_value = dynamic_mode

        DNode(
            context="ctx",
            address="0.0.0.0",
            port=1234,
            topology_mode=dynamic_mode,
            sampling_conf=None,
        )

    node_init.assert_called_once()
    assert node_init.call_args.kwargs["config_path"] is None
    logger.assert_called_once()


def test_dnode_uses_sampling_config_file_when_sampling_is_provided() -> None:
    """Sampling config should be created and forwarded as `config_path`."""
    sampling = SimpleNamespace(config_file="sampling.json", create=MagicMock())

    with patch("flwr.decentralized.node.Node.__init__", return_value=None) as node_init:
        DNode(
            context="ctx",
            address="0.0.0.0",
            port=1234,
            topology_mode=object(),
            sampling_conf=sampling,
        )

    sampling.create.assert_called_once()
    assert node_init.call_args.kwargs["config_path"] == "sampling.json"


def test_start_node_registers_runs_and_unregisters_apps() -> None:
    """`start_node` should wire apps and clean them up in order."""
    node = MagicMock(name="node")
    app_a = SimpleNamespace(name="app_a", node=None)
    app_b = SimpleNamespace(name="app_b", node=None)

    start_node(node=node, applications=[app_a, app_b], timeout=77)

    assert app_a.node is node
    assert app_b.node is node
    node.register.assert_any_call(app_name="app_a", app=app_a)
    node.register.assert_any_call(app_name="app_b", app=app_b)
    node.run.assert_called_once_with(timeout=77)
    node.unregister.assert_any_call(app_name="app_a")
    node.unregister.assert_any_call(app_name="app_b")
