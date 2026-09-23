"""Contract tests for strategy-to-ClientApp layer downloads."""

import sys
import types

import torch
from flwr.app import Context, RecordDict

omegaconf_stub = types.ModuleType("omegaconf")


class DictConfig(dict):
    """Minimal DictConfig stub for importing the client module."""

    def __getattr__(self, name: str):
        """Provide attribute access used by the application."""
        try:
            value = self[name]
            return DictConfig(value) if isinstance(value, dict) else value
        except KeyError as exc:
            raise AttributeError(name) from exc


omegaconf_stub.DictConfig = DictConfig
sys.modules.setdefault("omegaconf", omegaconf_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object()
transformers_stub.AutoTokenizer = types.SimpleNamespace(from_pretrained=None)
transformers_stub.BitsAndBytesConfig = object()
sys.modules.setdefault("transformers", transformers_stub)

from flowertune_llm.client_app import (  # noqa: E402
    _validate_downloaded_layers,
    train_download,
)
from flowertune_llm.fedavgstreaming import FedAvgStreaming  # noqa: E402
from flowertune_llm.task import load_layer_from_disk  # noqa: E402


class _ClientGrid:
    """Execute strategy messages directly through one ClientApp context."""

    def __init__(self, context: Context) -> None:
        self.context = context
        self.messages = []

    def send_and_receive(self, messages, timeout):  # noqa: ARG002
        """Deliver messages synchronously and return their replies."""
        self.messages.extend(messages)
        return [train_download(message, self.context) for message in messages]


def test_strategy_download_metadata_round_trip(tmp_path) -> None:
    """Strategy metadata must reconstruct and verify a split client layer."""
    context = Context(
        run_id=900,
        node_id=901,
        node_config={},
        state=RecordDict(),
        run_config={"aggregation.layer-write-dir": str(tmp_path)},
    )
    grid = _ClientGrid(context)
    expected = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    strategy = FedAvgStreaming(initial_state_dict={"layer.weight": expected})
    strategy._layer_names = ["layer.weight"]  # pylint: disable=protected-access
    strategy._download_max_chunk_bytes = 16  # pylint: disable=protected-access
    strategy._download_pipeline_depth = 1  # pylint: disable=protected-access

    strategy._download_layers_to_clients(  # pylint: disable=protected-access
        grid=grid,
        node_ids=[context.node_id],
        state_dict={"layer.weight": expected},
        timeout=30.0,
    )

    assert len(grid.messages) == 3
    _validate_downloaded_layers(context, ["layer.weight"])
    layer_path = tmp_path / "900" / "901" / "layer.weight.pt"
    actual = load_layer_from_disk(str(layer_path), "layer.weight")
    assert torch.equal(actual, expected)
