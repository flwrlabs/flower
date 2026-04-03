"""Shared task helpers for flowertune-llm client training/comms."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import re
import shlex
import subprocess
from typing import Any

import torch
from flwr.app import Context
from omegaconf import DictConfig

STATE_LAYER_PATHS = "layer_paths"


@dataclass
class CachedLayer:
    layer_name: str
    layer_path: str
    tensor: torch.Tensor
    dirty: bool = False


def _config_value(context: Context, key: str, default: Any = None) -> Any:
    """Read config value with node-level override precedence."""
    if key in context.node_config:
        return context.node_config[key]
    return context.run_config.get(key, default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


def training_disabled(context: Context) -> bool:
    """Return whether client-side training should be skipped."""
    return _as_bool(_config_value(context, "train.disable", False), default=False)


def sanitize_layer_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def chunk_key(layer_name: str, start: int, end: int) -> str:
    return f"{layer_name}::chunk_{start}_{end}"


def context_layer_key(context: Context, layer_name: str) -> tuple[int, int, str]:
    return (int(context.run_id), int(context.node_id), layer_name)


def context_path_key(context: Context, layer_path: str) -> tuple[int, int, str]:
    return (int(context.run_id), int(context.node_id), layer_path)


def parse_chunk_ranges(config: dict[str, Any]) -> list[tuple[int, int]]:
    if "chunk_starts" in config and "chunk_ends" in config:
        starts = [int(v) for v in list(config["chunk_starts"])]
        ends = [int(v) for v in list(config["chunk_ends"])]
        range_count = min(len(starts), len(ends))
        return [(starts[i], ends[i]) for i in range(range_count)]
    return [(int(config.get("chunk_start", 0)), int(config.get("chunk_end", 0)))]


def is_last_batch(config: dict[str, Any]) -> bool:
    if "is_last_batch" in config:
        return bool(config["is_last_batch"])
    chunk_idx = int(config.get("chunk_idx", 0))
    chunk_batch_count = int(config.get("chunk_batch_count", 0))
    if chunk_batch_count > 0:
        return chunk_idx >= (chunk_batch_count - 1)
    chunk_count = int(config.get("chunk_count", 0))
    chunks_in_message = max(1, int(config.get("chunks_in_message", 1)))
    if chunk_count > 0:
        return ((chunk_idx + 1) * chunks_in_message) >= chunk_count
    return True


def shape_from_text(shape_text: str) -> list[int]:
    if not shape_text:
        return []
    return [int(part) for part in shape_text.split(",") if part]


def load_layer_from_disk(layer_path: str, layer_name: str) -> torch.Tensor | None:
    if not os.path.exists(layer_path):
        return None
    with open(layer_path, "rb") as file:
        layer_dict = pickle.load(file)
    tensor = layer_dict.get(layer_name)
    if tensor is None and layer_dict:
        tensor = next(iter(layer_dict.values()))
    if tensor is None:
        return None
    return tensor.detach().cpu()


def flush_cached_layer(
    cache: dict[tuple[int, int, str], CachedLayer], cache_key: tuple[int, int, str]
) -> None:
    cached = cache.get(cache_key)
    if cached is None or not cached.dirty:
        return
    with open(cached.layer_path, "wb") as file:
        pickle.dump({cached.layer_name: cached.tensor}, file)
    cached.dirty = False


def flush_caches_for_context(
    cache: dict[tuple[int, int, str], CachedLayer],
    context: Context,
    *,
    flush_before_drop: bool,
) -> None:
    run_id = int(context.run_id)
    node_id = int(context.node_id)
    keys_to_clear = [
        key for key in cache if key[0] == run_id and key[1] == node_id
    ]
    for key in keys_to_clear:
        if flush_before_drop:
            flush_cached_layer(cache, key)
        cache.pop(key, None)


def layer_dir(context: Context) -> str:
    configured_base = _config_value(context, "layer-write-dir", "")
    if not configured_base:
        configured_base = _config_value(context, "aggregation.layer-write-dir", "")
    if isinstance(configured_base, str) and configured_base.strip():
        layer_base_dir = os.path.abspath(
            os.path.expandvars(os.path.expanduser(configured_base.strip()))
        )
    else:
        layer_base_dir = os.path.join(os.getcwd(), "layers")

    final_layer_dir = os.path.join(
        layer_base_dir, str(context.run_id), str(context.node_id)
    )
    os.makedirs(final_layer_dir, exist_ok=True)
    return final_layer_dir


def load_state_dict_from_layer_files(context: Context) -> dict[str, torch.Tensor]:
    """Load a full state_dict from layer files tracked in context state."""
    if STATE_LAYER_PATHS not in context.state:
        return {}

    layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
    state_dict: dict[str, torch.Tensor] = {}
    for layer_path in layer_paths:
        if not os.path.exists(layer_path):
            continue
        with open(layer_path, "rb") as file:
            layer_dict = pickle.load(file)
        for layer_name, tensor in layer_dict.items():
            state_dict[str(layer_name)] = tensor.detach().cpu()
    return state_dict


def extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    """Extract state_dict from common checkpoint layouts."""
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


def _normalize_state_dict_for_hf(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize nested checkpoint dicts to plain HF-like state_dict."""
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        model_state = state_dict["model"]
        return {
            str(name): tensor.detach().cpu()
            for name, tensor in model_state.items()
            if torch.is_tensor(tensor)
        }
    return {
        str(name): tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }


def _save_state_dict_as_dcp(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    *,
    train_spec_name: str,
    model_args_key: str,
    dcp_threads: int,
) -> None:
    """Save state_dict in DCP format, preferring TorchTitan adapter when available."""
    from torch.distributed import checkpoint as dcp

    os.makedirs(output_dir, exist_ok=True)
    writer = dcp.filesystem.FileSystemWriter(output_dir, thread_count=dcp_threads)
    try:
        import torchtitan.protocols.train_spec as train_spec_module
    except Exception:
        dcp.save(state_dict, storage_writer=writer)
        return

    train_spec = train_spec_module.get_train_spec(train_spec_name)
    model_args = train_spec.model_args[model_args_key]
    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    titan_state_dict = sd_adapter.from_hf(state_dict)
    dcp.save(titan_state_dict, storage_writer=writer)


def _load_state_dict_from_dcp(
    input_dir: str,
    *,
    train_spec_name: str,
    model_args_key: str,
) -> dict[str, torch.Tensor]:
    """Load state_dict from DCP format, converting back to HF-like mapping."""
    from torch.distributed import checkpoint as dcp

    reader = dcp.filesystem.FileSystemReader(input_dir)
    checkpoint_dict: dict[str, Any] = {}
    dcp.load(checkpoint_dict, storage_reader=reader, no_dist=True)

    try:
        import torchtitan.protocols.train_spec as train_spec_module
    except Exception:
        return _normalize_state_dict_for_hf(extract_state_dict(checkpoint_dict))

    train_spec = train_spec_module.get_train_spec(train_spec_name)
    model_args = train_spec.model_args[model_args_key]
    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    hf_state = sd_adapter.to_hf(checkpoint_dict)
    return _normalize_state_dict_for_hf(extract_state_dict(hf_state))


def run_torchtitan_training(
    cfg: DictConfig,
    context: Context,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Execute TorchTitan training command and load the updated state_dict."""
    trainer_cfg = getattr(cfg, "trainer", {})
    titan_cfg = getattr(trainer_cfg, "torchtitan", {})
    command = str(getattr(titan_cfg, "command", "")).strip()
    if not command:
        raise ValueError(
            "trainer.backend is 'torchtitan' but trainer.torchtitan.command is empty"
        )

    output_dir = os.path.join(layer_dir(context), "torchtitan")
    os.makedirs(output_dir, exist_ok=True)
    input_state_path = os.path.join(output_dir, "input_state.pt")
    output_state_path = os.path.join(output_dir, "output_state.pt")
    input_dcp_dir = os.path.join(output_dir, "input_state.dcp")
    output_dcp_dir = os.path.join(output_dir, "output_state.dcp")
    dcp_enabled = _as_bool(
        _config_value(
            context,
            "trainer.torchtitan.dcp-enabled",
            _config_value(context, "trainer.torchtitan.dcp_enabled", False),
        ),
        default=False,
    )
    dcp_train_spec = str(
        _config_value(
            context,
            "trainer.torchtitan.dcp-train-spec",
            _config_value(context, "trainer.torchtitan.dcp_train_spec", "llama3"),
        )
    ).strip()
    dcp_model_args = str(
        _config_value(
            context,
            "trainer.torchtitan.dcp-model-args",
            _config_value(context, "trainer.torchtitan.dcp_model_args", "8B"),
        )
    ).strip()
    dcp_threads = int(
        _config_value(
            context,
            "trainer.torchtitan.dcp-threads",
            _config_value(context, "trainer.torchtitan.dcp_threads", 8),
        )
    )
    torch.save(state_dict, input_state_path)
    if dcp_enabled:
        _save_state_dict_as_dcp(
            state_dict,
            input_dcp_dir,
            train_spec_name=dcp_train_spec,
            model_args_key=dcp_model_args,
            dcp_threads=dcp_threads,
        )

    env = os.environ.copy()
    env["FLWR_TORCHTITAN_INPUT_STATE"] = input_state_path
    env["FLWR_TORCHTITAN_OUTPUT_STATE"] = output_state_path
    env["FLWR_TORCHTITAN_INPUT_DCP_DIR"] = input_dcp_dir
    env["FLWR_TORCHTITAN_OUTPUT_DCP_DIR"] = output_dcp_dir
    env["FLWR_RUN_ID"] = str(context.run_id)
    env["FLWR_NODE_ID"] = str(context.node_id)
    scheduler_env = {
        "FLWR_TORCHTITAN_INPUT_STATE": input_state_path,
        "FLWR_TORCHTITAN_OUTPUT_STATE": output_state_path,
        "FLWR_TORCHTITAN_INPUT_DCP_DIR": input_dcp_dir,
        "FLWR_TORCHTITAN_OUTPUT_DCP_DIR": output_dcp_dir,
        "FLWR_RUN_ID": str(context.run_id),
        "FLWR_NODE_ID": str(context.node_id),
    }

    workdir = str(getattr(titan_cfg, "workdir", "")).strip() or None
    scheduler_backend = str(
        _config_value(context, "scheduler.backend", "local")
    ).strip().lower()

    def run_local() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            shell=True,
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )

    if scheduler_backend in {"", "none", "local"}:
        result = run_local()
    elif scheduler_backend == "slurm":
        slurm_submit = str(
            _config_value(context, "scheduler.slurm.submit-command", "sbatch")
        ).strip() or "sbatch"
        slurm_extra_args = str(
            _config_value(context, "scheduler.slurm.extra-args", "")
        ).strip()
        slurm_wait = _as_bool(
            _config_value(context, "scheduler.slurm.wait", True), default=True
        )

        script_path = os.path.join(output_dir, "torchtitan_slurm.sh")
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write("#!/usr/bin/env bash\n")
            script_file.write("set -euo pipefail\n")
            if workdir:
                script_file.write(f"cd {shlex.quote(workdir)}\n")
            for key, value in scheduler_env.items():
                script_file.write(
                    f"export {key}={shlex.quote(str(value))}\n"
                )
            script_file.write(f"{command}\n")
        os.chmod(script_path, 0o755)

        submit_parts = [slurm_submit]
        if slurm_wait:
            submit_parts.append("--wait")
        submit_parts.append("--parsable")
        if slurm_extra_args:
            submit_parts.extend(shlex.split(slurm_extra_args))
        submit_parts.append(script_path)

        result = subprocess.run(
            submit_parts,
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
    elif scheduler_backend == "flux":
        flux_run = str(
            _config_value(context, "scheduler.flux.run-command", "flux run")
        ).strip() or "flux run"
        flux_extra_args = str(
            _config_value(context, "scheduler.flux.extra-args", "")
        ).strip()

        wrapped_command = command
        if workdir:
            wrapped_command = f"cd {shlex.quote(workdir)} && {wrapped_command}"
        export_prefix = " ".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in scheduler_env.items()
        )
        wrapped_command = f"{export_prefix} bash -lc {shlex.quote(wrapped_command)}"
        flux_parts = shlex.split(flux_run)
        if flux_extra_args:
            flux_parts.extend(shlex.split(flux_extra_args))
        flux_parts.extend(["bash", "-lc", wrapped_command])

        result = subprocess.run(
            flux_parts,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        raise ValueError(
            f"Unsupported scheduler.backend '{scheduler_backend}'. "
            "Use local, slurm, or flux."
        )

    if result.returncode != 0:
        raise RuntimeError(
            "TorchTitan command failed with exit code "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if os.path.exists(output_state_path):
        payload = torch.load(output_state_path, map_location="cpu")
        trained_state = extract_state_dict(payload)
        return _normalize_state_dict_for_hf(trained_state)

    if dcp_enabled and os.path.isdir(output_dcp_dir):
        return _load_state_dict_from_dcp(
            output_dcp_dir,
            train_spec_name=dcp_train_spec,
            model_args_key=dcp_model_args,
        )

    raise FileNotFoundError(
        "TorchTitan command completed but did not write either "
        f"{output_state_path} or {output_dcp_dir}. "
        "Set FLWR_TORCHTITAN_OUTPUT_STATE or FLWR_TORCHTITAN_OUTPUT_DCP_DIR."
    )
