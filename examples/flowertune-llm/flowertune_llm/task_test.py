"""Tests for task artifact cleanup helpers."""

import os
import subprocess
import sys
import types
from pathlib import Path

import torch
from flwr.app import Context, RecordDict

omegaconf_stub = types.ModuleType("omegaconf")
omegaconf_stub.DictConfig = object
sys.modules.setdefault("omegaconf", omegaconf_stub)

from flowertune_llm import task as task_module  # noqa: E402


def test_run_torchtitan_training_cleans_successful_dcp_handoff(
    tmp_path, monkeypatch
) -> None:
    """Successful DCP training should leave cache but remove per-round DCP copies."""
    layer_base = tmp_path / "layers"
    workspace = tmp_path / "workspace"
    dump_folder = tmp_path / "dump"
    context = Context(
        run_id=10,
        node_id=20,
        node_config={},
        state=RecordDict(),
        run_config={
            "aggregation.layer-write-dir": str(layer_base),
            "client.workspace": str(workspace),
            "client.train-steps": 5,
            "model.name": "test/model",
            "trainer.dump-folder": str(dump_folder),
            "trainer.torchtitan.dcp-enabled": True,
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )
    torchtitan_dir = layer_base / "10" / "20" / "torchtitan"
    torchtitan_dir.mkdir(parents=True)
    stale_output_state = torchtitan_dir / "output_state.pt"
    torch.save({"weight": torch.full((1,), -1.0)}, stale_output_state)

    paths: dict[str, str] = {}

    def fake_save_state_dict_as_dcp(_state_dict, output_dir, **_kwargs) -> None:
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "__0_0.distcp").write_bytes(b"cached")

    def fake_run(*args, **kwargs):
        env = kwargs["env"]
        paths.update({
            "cache": env["FLWR_TORCHTITAN_DCP_CACHE_DIR"],
            "input": env["FLWR_TORCHTITAN_INPUT_DCP_DIR"],
            "output": env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"],
            "step0": env["FLWR_TORCHTITAN_STEP0_DCP_DIR"],
        })
        assert not stale_output_state.exists()
        os.makedirs(os.path.dirname(paths["step0"]), exist_ok=True)
        os.symlink(env["FLWR_TORCHTITAN_INPUT_DCP_DIR"], paths["step0"])
        os.makedirs(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], exist_ok=True)
        Path(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], "__0_0.distcp").write_bytes(
            b"trained"
        )
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    def fake_load_state_dict_from_dcp(input_dir, **_kwargs):
        assert input_dir == paths["output"]
        assert not os.path.lexists(paths["step0"])
        assert not os.path.lexists(paths["input"])
        assert os.path.isdir(input_dir)
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(
        task_module, "_save_state_dict_as_dcp", fake_save_state_dict_as_dcp
    )
    monkeypatch.setattr(task_module.subprocess, "run", fake_run)
    monkeypatch.setattr(task_module, "_load_state_dict_from_dcp", fake_load_state_dict_from_dcp)

    trained_state = task_module.run_torchtitan_training(
        cfg, context, {"weight": torch.zeros(1)}, server_round=1
    )

    assert torch.equal(trained_state["weight"], torch.ones(1))
    assert os.path.isdir(paths["cache"])
    assert not os.path.lexists(paths["input"])
    assert not os.path.lexists(paths["output"])
    assert not os.path.lexists(paths["step0"])
    assert not torchtitan_dir.exists()
