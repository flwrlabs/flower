"""Tests for task artifact cleanup helpers."""

import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import torch
from flwr.app import Context, RecordDict

omegaconf_stub = types.ModuleType("omegaconf")


class DictConfig(dict):
    """Minimal DictConfig stub for importing task helpers."""

    def __getattr__(self, name: str):
        try:
            value = self[name]
            return DictConfig(value) if isinstance(value, dict) else value
        except KeyError as exc:
            raise AttributeError(name) from exc


omegaconf_stub.DictConfig = DictConfig
sys.modules.setdefault("omegaconf", omegaconf_stub)

from flowertune_llm import task as task_module  # noqa: E402


def _write_layer_file(path: Path, name: str, tensor: torch.Tensor) -> None:
    with open(path, "wb") as file:
        pickle.dump({name: tensor}, file)


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


def test_layerwise_dcp_dry_run_renders_job_side_conversion(tmp_path) -> None:
    """Layerwise DCP dry-runs should render conversion inside the job."""
    layer_directory = tmp_path / "layers" / "10" / "20"
    layer_directory.mkdir(parents=True)
    layer_path = layer_directory / "layer.a.pt"
    _write_layer_file(layer_path, "layer.a", torch.ones(2))
    context = Context(
        run_id=10,
        node_id=20,
        node_config={},
        state=RecordDict(),
        run_config={
            "aggregation.layer-write-dir": str(tmp_path / "layers"),
            "client.workspace": str(tmp_path / "workspace"),
            "model.name": "test/model",
            "trainer.backend": "torchtitan",
            "trainer.dry-run": True,
            "trainer.python-exec": "python",
            "trainer.torchtitan.dcp-enabled": True,
            "trainer.torchtitan.dcp-train-spec": "llama3",
            "trainer.torchtitan.dcp-model-args": "auto",
            "scheduler.backend": "slurm",
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )

    result = task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )

    assert result is None
    script = layer_directory / "torchtitan" / "torchtitan_slurm.sh"
    script_text = script.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "flowertune_llm.dcp_converter" in script_text
    assert "FLWR_TORCHTITAN_INPUT_LAYERS_DIR" in script_text
    assert "FLWR_TORCHTITAN_OUTPUT_LAYERS_READY" in script_text


def test_dcp_converter_reads_and_publishes_layer_files(tmp_path, monkeypatch) -> None:
    """The conversion worker should use layer files as its only input artifact."""
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    layer_path = input_directory / "layer.a.pt"
    _write_layer_file(layer_path, "layer.a", torch.ones(2))

    captured: dict[str, object] = {}

    def fake_save(state_dict, output_dir, **kwargs) -> None:
        captured["state_dict"] = state_dict
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs

    monkeypatch.setattr(task_module, "_save_state_dict_as_dcp", fake_save)
    task_module.convert_layer_directory_to_dcp(
        str(input_directory),
        str(output_directory),
        train_spec_name="llama3",
        model_args_key="auto",
        dcp_threads=2,
    )

    assert captured["output_dir"] == str(output_directory)
    assert torch.equal(captured["state_dict"]["layer.a"], torch.ones(2))


def test_dcp_converter_publishes_output_layers_with_marker(
    tmp_path, monkeypatch
) -> None:
    """DCP-to-layer conversion should publish files and readiness atomically."""
    reference_directory = tmp_path / "reference"
    reference_directory.mkdir()
    _write_layer_file(reference_directory / "layer.a.pt", "layer.a", torch.zeros(2))
    input_directory = tmp_path / "dcp"
    input_directory.mkdir()
    output_directory = tmp_path / "layers"
    marker = output_directory / ".torchtitan_layers_ready"

    monkeypatch.setattr(
        task_module,
        "_load_state_dict_from_dcp",
        lambda *_args, **_kwargs: {"layer.a": torch.ones(2)},
    )
    task_module.convert_dcp_to_layer_directory(
        str(input_directory),
        str(reference_directory),
        str(output_directory),
        train_spec_name="llama3",
        model_args_key="auto",
        dcp_threads=2,
        ready_marker=str(marker),
    )

    assert marker.read_text(encoding="utf-8") == "ready\n"
    assert torch.equal(
        task_module.load_state_dict_from_layer_directory(str(output_directory))[
            "layer.a"
        ],
        torch.ones(2),
    )
