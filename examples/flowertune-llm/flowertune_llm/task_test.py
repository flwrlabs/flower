"""Tests for task artifact cleanup helpers."""

import json
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
from flowertune_llm import dcp_converter  # noqa: E402


def _write_layer_file(path: Path, name: str, tensor: torch.Tensor) -> None:
    with open(path, "wb") as file:
        pickle.dump({name: tensor}, file)


def test_run_torchtitan_training_cleans_successful_dcp_handoff(
    tmp_path, monkeypatch
) -> None:
    """Successful DCP training should leave cache but remove per-round DCP copies."""
    layer_base = tmp_path / "layers"
    workspace = tmp_path / "workspace"
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
            "trainer.dump-folder": "",
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
        paths.update(
            {
                "cache": env["FLWR_TORCHTITAN_DCP_CACHE_DIR"],
                "input": env["FLWR_TORCHTITAN_INPUT_DCP_DIR"],
                "output": env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"],
                "step0": env["FLWR_TORCHTITAN_STEP0_DCP_DIR"],
            }
        )
        assert not stale_output_state.exists()
        os.makedirs(os.path.dirname(paths["step0"]), exist_ok=True)
        os.symlink(env["FLWR_TORCHTITAN_INPUT_DCP_DIR"], paths["step0"])
        os.makedirs(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], exist_ok=True)
        Path(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], "__0_0.distcp").write_bytes(
            b"trained"
        )
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="", stderr=""
        )

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
    monkeypatch.setattr(
        task_module, "_load_state_dict_from_dcp", fake_load_state_dict_from_dcp
    )

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
    assert "FLWR_TORCHTITAN_CONVERSION_PROFILE" in script_text
    assert str(layer_directory / "torchtitan_conversion_profile.jsonl") in script_text

    context.run_config["trainer.torchtitan.dcp-convert-on-client"] = True
    task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )
    client_script_text = script.read_text(encoding="utf-8")
    assert 'FLWR_TORCHTITAN_DCP_CONVERT_ON_CLIENT="true"' in client_script_text
    assert "flowertune_llm.dcp_converter" not in client_script_text


def test_layerwise_dcp_dry_run_renders_three_scheduler_jobs(tmp_path) -> None:
    """Separate conversion mode should render three single-purpose scripts."""
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
            "trainer.torchtitan.dcp-separate-jobs": True,
            "scheduler.backend": "slurm",
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )

    task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )

    script_dir = layer_directory / "torchtitan"
    to_dcp = script_dir / "torchtitan_slurm_to_dcp.sh"
    train = script_dir / "torchtitan_slurm_train.sh"
    from_dcp = script_dir / "torchtitan_slurm_from_dcp.sh"
    for script in (to_dcp, train, from_dcp):
        subprocess.run(["bash", "-n", str(script)], check=True)
    assert "--direction to-dcp" in to_dcp.read_text(encoding="utf-8")
    assert "true" in train.read_text(encoding="utf-8")
    assert "--direction to-layers" in from_dcp.read_text(encoding="utf-8")
    report = (script_dir / "dry_run_summary.txt").read_text(encoding="utf-8")
    assert "separate_dcp_jobs=true" in report


def test_layerwise_dcp_submits_dependent_slurm_jobs(tmp_path, monkeypatch) -> None:
    """Slurm conversion and training phases should form an afterok chain."""
    layer_directory = tmp_path / "layers" / "10" / "20"
    layer_directory.mkdir(parents=True)
    layer_path = layer_directory / "layer.a.pt"
    _write_layer_file(layer_path, "layer.a", torch.ones(2))
    workspace = tmp_path / "workspace"
    context = Context(
        run_id=10,
        node_id=20,
        node_config={},
        state=RecordDict(),
        run_config={
            "aggregation.layer-write-dir": str(tmp_path / "layers"),
            "client.workspace": str(workspace),
            "client.train-steps": 5,
            "model.name": "test/model",
            "trainer.backend": "torchtitan",
            "trainer.torchtitan.dcp-enabled": True,
            "trainer.torchtitan.dcp-separate-jobs": True,
            "scheduler.backend": "slurm",
            "scheduler.mem": "64G",
            "scheduler.conversion.mem": "256G",
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )
    submissions: list[list[str]] = []

    def fake_run(args, **_kwargs):
        command = [str(arg) for arg in args]
        submissions.append(command)
        script = Path(command[-1])
        env = _kwargs["env"]
        if script.name.endswith("_to_dcp.sh"):
            os.makedirs(env["FLWR_TORCHTITAN_DCP_CONVERSION_DIR"], exist_ok=True)
            os.symlink(
                env["FLWR_TORCHTITAN_DCP_CONVERSION_DIR"],
                env["FLWR_TORCHTITAN_INPUT_DCP_DIR"],
            )
            job_id = "101"
        elif script.name.endswith("_train.sh"):
            os.makedirs(env["FLWR_TORCHTITAN_FINAL_DCP_DIR"], exist_ok=True)
            job_id = "102"
        else:
            os.makedirs(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], exist_ok=True)
            Path(env["FLWR_TORCHTITAN_OUTPUT_LAYERS_READY"]).write_text(
                "ready\n", encoding="utf-8"
            )
            job_id = "103"
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=f"{job_id}\n", stderr=""
        )

    monkeypatch.setattr(task_module.subprocess, "run", fake_run)

    result = task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        server_round=1,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )

    assert result is None
    assert len(submissions) == 3
    assert "--mem" in submissions[0] and "256G" in submissions[0]
    assert not any(arg.startswith("--dependency=") for arg in submissions[0])
    assert "--dependency=afterok:101" in submissions[1]
    assert "64G" in submissions[1]
    assert "--dependency=afterok:102" in submissions[2]
    assert "--wait" in submissions[2]


def test_layerwise_dcp_submits_dependent_flux_jobs(tmp_path, monkeypatch) -> None:
    """Flux conversion and training phases should form an afterok chain."""
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
            "client.train-steps": 5,
            "model.name": "test/model",
            "trainer.backend": "torchtitan",
            "trainer.torchtitan.dcp-enabled": True,
            "trainer.torchtitan.dcp-separate-jobs": True,
            "scheduler.backend": "flux",
            "scheduler.flux.conversion-extra-args": "--queue=high-memory",
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )
    commands: list[list[str]] = []

    def fake_run(args, **kwargs):
        command = [str(arg) for arg in args]
        commands.append(command)
        if command[:3] == ["flux", "job", "attach"]:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )
        script = Path(command[-1])
        env = kwargs["env"]
        if script.name.endswith("_to_dcp.sh"):
            os.makedirs(env["FLWR_TORCHTITAN_DCP_CONVERSION_DIR"], exist_ok=True)
            os.symlink(
                env["FLWR_TORCHTITAN_DCP_CONVERSION_DIR"],
                env["FLWR_TORCHTITAN_INPUT_DCP_DIR"],
            )
            job_id = "f101"
        elif script.name.endswith("_train.sh"):
            os.makedirs(env["FLWR_TORCHTITAN_FINAL_DCP_DIR"], exist_ok=True)
            job_id = "f102"
        else:
            os.makedirs(env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"], exist_ok=True)
            Path(env["FLWR_TORCHTITAN_OUTPUT_LAYERS_READY"]).write_text(
                "ready\n", encoding="utf-8"
            )
            job_id = "f103"
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=f"{job_id}\n", stderr=""
        )

    monkeypatch.setattr(task_module.subprocess, "run", fake_run)

    result = task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        server_round=1,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )

    assert result is None
    assert len(commands) == 4
    assert "--queue=high-memory" in commands[0]
    assert "--dependency=afterok:f101" in commands[1]
    assert "--dependency=afterok:f102" in commands[2]
    assert commands[3][-1] == "f103"


def test_layerwise_dcp_client_conversion_runs_outside_job(
    tmp_path, monkeypatch
) -> None:
    """Client-side conversion should leave the job with only DCP handoff."""
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
            "client.train-steps": 5,
            "model.name": "test/model",
            "trainer.backend": "torchtitan",
            "trainer.python-exec": "python",
            "trainer.torchtitan.dcp-enabled": True,
            "trainer.torchtitan.dcp-convert-on-client": True,
            "scheduler.backend": "local",
        },
    )
    cfg = types.SimpleNamespace(
        trainer=types.SimpleNamespace(
            torchtitan=types.SimpleNamespace(command="true", workdir="")
        )
    )
    phases: list[str] = []

    def fake_to_dcp(input_dir, output_dir, **_kwargs) -> None:
        phases.append("to_dcp")
        assert input_dir == str(layer_directory)
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "__0_0.distcp").write_bytes(b"input")

    def fake_to_layers(input_dir, _reference_dir, output_dir, **kwargs) -> None:
        phases.append("to_layers")
        assert os.path.isdir(input_dir)
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "layer.a.pt").write_bytes(layer_path.read_bytes())
        Path(kwargs["ready_marker"]).write_text("ready\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        env = kwargs["env"]
        assert env["FLWR_TORCHTITAN_DCP_CONVERT_ON_CLIENT"] == "true"
        assert os.path.isdir(env["FLWR_TORCHTITAN_INPUT_DCP_DIR"])
        output_dir = env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"]
        os.makedirs(output_dir, exist_ok=True)
        Path(output_dir, "__0_0.distcp").write_bytes(b"output")
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(task_module, "convert_layer_directory_to_dcp", fake_to_dcp)
    monkeypatch.setattr(task_module, "convert_dcp_to_layer_directory", fake_to_layers)
    monkeypatch.setattr(task_module.subprocess, "run", fake_run)

    result = task_module.run_torchtitan_training(
        cfg,
        context,
        None,
        server_round=1,
        layer_paths=[str(layer_path)],
        output_layer_dir=str(layer_directory),
    )

    assert result is None
    assert phases == ["to_dcp", "to_layers"]
    assert not os.path.lexists(layer_directory / "torchtitan" / "input_state.dcp")
    assert not os.path.lexists(layer_directory / "torchtitan" / "output_state.dcp")
    assert not (layer_directory / "torchtitan").exists()


def test_dcp_converter_records_phase_profile(tmp_path, monkeypatch) -> None:
    """The job-side converter records duration and peak RSS telemetry."""
    profile_path = tmp_path / "conversion.jsonl"
    monkeypatch.setenv("FLWR_TORCHTITAN_CONVERSION_PROFILE", str(profile_path))
    monkeypatch.setattr(
        dcp_converter,
        "convert_layer_directory_to_dcp",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dcp_converter",
            "--direction",
            "to-dcp",
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    dcp_converter.main()

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "end"]
    assert events[0]["phase"] == "to_dcp"
    assert events[1]["success"] is True
    assert events[1]["duration_ms"] >= 0
    assert events[1]["max_rss_mb"] > 0
    metrics = task_module.read_conversion_profile(str(profile_path))
    assert metrics["profile.client.dcp.to_dcp.ms"] >= 0
    assert metrics["profile.client.dcp.to_dcp.mem_mb"] > 0


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
