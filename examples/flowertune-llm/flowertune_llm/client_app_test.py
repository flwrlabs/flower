"""Tests for client-side layer artifact lifecycle."""

import os
import sys
import types

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict

omegaconf_stub = types.ModuleType("omegaconf")


class DictConfig(dict):
    """Minimal DictConfig stub for importing the client module."""

    def __getattr__(self, name: str):
        try:
            value = self[name]
            return DictConfig(value) if isinstance(value, dict) else value
        except KeyError as exc:
            raise AttributeError(name) from exc


omegaconf_stub.DictConfig = DictConfig
sys.modules.setdefault("omegaconf", omegaconf_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object()
sys.modules.setdefault("transformers", transformers_stub)

from flowertune_llm.client_app import (  # noqa: E402
    _DOWNLOAD_LAYER_CACHE,
    STATE_LAYER_IDX,
    STATE_LAYER_NAMES,
    STATE_LAYER_PATHS,
    STATE_NUM_EXAMPLES,
    _persist_layer_files,
    train,
    train_download,
    train_comms,
)
from flowertune_llm import client_app as client_app_module  # noqa: E402
from flowertune_llm.task import load_layer_from_disk  # noqa: E402


def test_train_comms_cleans_layer_files_after_final_send(tmp_path) -> None:
    """Final layer-wise upload should not leave PT copies on disk."""
    context = Context(
        run_id=123,
        node_id=456,
        node_config={},
        state=RecordDict(),
        run_config={"aggregation.layer-write-dir": str(tmp_path)},
    )
    _persist_layer_files(
        context,
        {
            "layer.a": torch.tensor([1.0, 2.0]),
            "layer.b": torch.tensor([3.0, 4.0]),
        },
        ["layer.a", "layer.b"],
    )
    layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
    assert all(os.path.exists(path) for path in layer_paths)

    message = Message(
        content=RecordDict({
            "config": ConfigRecord({
                "upload_layer_idxs": [0, 1],
                "upload_layer_names": ["layer.a", "layer.b"],
                "upload_chunk_starts": [0, 0],
                "upload_chunk_ends": [0, 0],
                "upload_is_last_chunk": [True, True],
            }),
        }),
        dst_node_id=1,
        message_type="train.layer_wise_communication",
    )

    reply = train_comms(message, context)

    assert reply.content["config"]["send_complete"]
    assert set(reply.content["arrays"].keys()) == {
        "layer.a::chunk_0_0",
        "layer.b::chunk_0_0",
    }
    assert all(not os.path.exists(path) for path in layer_paths)
    assert not os.path.exists(os.path.join(tmp_path, "123"))
    assert STATE_LAYER_NAMES not in context.state
    assert STATE_LAYER_PATHS not in context.state
    assert STATE_LAYER_IDX not in context.state
    assert STATE_NUM_EXAMPLES not in context.state


def test_train_download_persists_split_chunks_across_processes(tmp_path) -> None:
    """Split layer chunks must survive separate ClientApp executions."""
    context = Context(
        run_id=321,
        node_id=654,
        node_config={},
        state=RecordDict(),
        run_config={"aggregation.layer-write-dir": str(tmp_path)},
    )

    last_chunk = Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "layer.big::chunk_2_4": torch.tensor([3.0, 4.0])
            }),
            "config": ConfigRecord({
                "download_layer_idxs": [0],
                "download_layer_names": ["layer.big"],
                "download_layer_shapes": ["4"],
                "download_chunk_starts": [2],
                "download_chunk_ends": [4],
                "download_is_last_chunk": [True],
            }),
        }),
        dst_node_id=1,
        message_type="train.layer_wise_download",
    )
    train_download(last_chunk, context)
    _DOWNLOAD_LAYER_CACHE.clear()

    first_chunk = Message(
        content=RecordDict({
            "arrays": ArrayRecord({
                "layer.big::chunk_0_2": torch.tensor([1.0, 2.0])
            }),
            "config": ConfigRecord({
                "download_layer_idxs": [0],
                "download_layer_names": ["layer.big"],
                "download_layer_shapes": ["4"],
                "download_chunk_starts": [0],
                "download_chunk_ends": [2],
                "download_is_last_chunk": [False],
            }),
        }),
        dst_node_id=1,
        message_type="train.layer_wise_download",
    )
    train_download(first_chunk, context)

    layer_path = context.state[STATE_LAYER_PATHS]["paths"][0]
    layer = load_layer_from_disk(layer_path, "layer.big")
    assert torch.equal(layer, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_layerwise_torchtitan_dcp_does_not_load_hf_model(tmp_path, monkeypatch) -> None:
    """Layerwise DCP training should hand files to the job without get_model."""
    context = Context(
        run_id=700,
        node_id=800,
        node_config={},
        state=RecordDict(),
        run_config={
            "aggregation.mode": "layerwise",
            "aggregation.layer-write-dir": str(tmp_path),
            "trainer.backend": "torchtitan",
            "trainer.torchtitan.dcp-enabled": True,
            "train.disable": False,
        },
    )
    _persist_layer_files(context, {"layer.a": torch.ones(2)}, ["layer.a"])
    message = Message(
        content=RecordDict({
            "config": ConfigRecord({
                "model_preloaded": True,
                "layer_names": ["layer.a"],
            })
        }),
        dst_node_id=1,
        message_type="train",
    )
    calls: dict[str, object] = {}

    def fail_get_model(_cfg):
        raise AssertionError("HF model construction should be skipped")

    def fake_run(_cfg, _context, state_dict, **kwargs):
        calls["state_dict"] = state_dict
        calls.update(kwargs)
        return None

    monkeypatch.setattr(client_app_module, "get_model", fail_get_model)
    monkeypatch.setattr(client_app_module, "run_torchtitan_training", fake_run)

    reply = train(message, context)

    assert calls["state_dict"] is None
    assert calls["layer_paths"] == list(context.state[STATE_LAYER_PATHS]["paths"])
    assert isinstance(reply.content["arrays"], ArrayRecord)
    assert len(reply.content["arrays"]) == 0
