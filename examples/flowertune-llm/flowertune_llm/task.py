"""Shared task helpers for flowertune-llm client training/comms."""

from __future__ import annotations

import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from flwr.app import Context
    from omegaconf import DictConfig

STATE_LAYER_PATHS = "layer_paths"
DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
FLOWERTUNE_LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def _config_str(context: Context, key: str, default: str = "") -> str:
    value = _config_value(context, key, default)
    if value is None:
        return default
    return str(value)


def _template_path(context: Context, key: str, fallback_name: str) -> str:
    configured = _config_str(context, key, "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(configured)))
    return os.path.join(DEFAULT_TEMPLATE_DIR, fallback_name)


def _render_template_text(template_text: str, values: dict[str, Any]) -> str:
    """Render {{ var }} placeholders with stringified values."""

    def replace(match: re.Match[str]) -> str:
        template_key = match.group(1).strip()
        return str(values.get(template_key, ""))

    pattern = re.compile(r"{{\s*([a-zA-Z0-9_.-]+)\s*}}")
    return pattern.sub(replace, template_text)


def _render_template_file(template_path: str, values: dict[str, Any]) -> str:
    with open(template_path, "r", encoding="utf-8") as file:
        template_text = file.read()
    return _render_template_text(template_text, values)


def _python_module_command(
    python_exec: str,
    module: str,
    arguments: list[str],
) -> str:
    """Build a shell-safe Python module command for a scheduler script."""
    executable = shlex.split(python_exec) or ["python"]
    return " ".join(
        shlex.quote(part)
        for part in [*executable, "-m", module, *arguments]
    )


def read_conversion_profile(profile_path: str) -> dict[str, float]:
    """Read successful DCP conversion phase metrics from a JSONL profile."""
    metrics: dict[str, float] = {}
    if not os.path.exists(profile_path):
        return metrics

    with open(profile_path, encoding="utf-8") as file:
        for line in file:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                event.get("event") != "end"
                or not event.get("success")
                or not isinstance(event.get("phase"), str)
            ):
                continue
            phase = event["phase"]
            duration_ms = event.get("duration_ms")
            max_rss_mb = event.get("max_rss_mb")
            if isinstance(duration_ms, (int, float)):
                metrics[f"profile.client.dcp.{phase}.ms"] = float(duration_ms)
            if isinstance(max_rss_mb, (int, float)):
                metrics[f"profile.client.dcp.{phase}.mem_mb"] = float(max_rss_mb)
    return metrics


def training_disabled(context: Context) -> bool:
    """Return whether client-side training should be skipped."""
    return _as_bool(_config_value(context, "train.disable", False), default=False)


def torchtitan_dcp_enabled(context: Context) -> bool:
    """Return whether TorchTitan DCP handoff is enabled."""
    return _as_bool(
        _config_value(
            context,
            "trainer.torchtitan.dcp-enabled",
            _config_value(context, "trainer.torchtitan.dcp_enabled", False),
        ),
        default=False,
    )


def sanitize_layer_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _sanitize_model_cache_path(model_name: str, fallback: str) -> str:
    """Return a safe relative cache path for a model name/repo id."""
    parts = [
        sanitize_layer_name(part)
        for part in re.split(r"[\\/]+", model_name.strip())
        if part not in {"", ".", ".."}
    ]
    parts = [part for part in parts if part]
    if not parts:
        return sanitize_layer_name(fallback)
    return os.path.join(*parts)


def _dcp_checkpoint_exists(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def _get_attr_any(obj: object, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return int(value)
    return None


def _state_dict_signature(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, int | None]:
    """Infer model-shape hints from an HF-like state_dict."""
    dim = None
    vocab_size = None
    q_out_dim = None
    kv_out_dim = None
    layer_ids: set[int] = set()

    layer_pattern = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
    for name, tensor in state_dict.items():
        match = layer_pattern.search(name)
        if match is not None:
            layer_ids.add(int(match.group(1)))
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        shape = tuple(int(x) for x in tensor.shape)
        if (
            dim is None
            and (
                name.endswith("embed_tokens.weight")
                or name.endswith("tok_embeddings.weight")
                or name == "wte.weight"
            )
        ):
            vocab_size, dim = shape
        if (
            q_out_dim is None
            and (
                name.endswith("self_attn.q_proj.weight")
                or name.endswith("attention.wq.weight")
            )
        ):
            q_out_dim, dim = shape
        if (
            kv_out_dim is None
            and (
                name.endswith("self_attn.k_proj.weight")
                or name.endswith("attention.wk.weight")
            )
        ):
            kv_out_dim = shape[0]

    return {
        "dim": dim,
        "vocab_size": vocab_size,
        "q_out_dim": q_out_dim,
        "kv_out_dim": kv_out_dim,
        "n_layers": (max(layer_ids) + 1) if layer_ids else None,
    }


def _model_args_signature(model_args: object) -> dict[str, int | None]:
    dim = _get_attr_any(model_args, ("dim", "hidden_size", "n_embd"))
    n_layers = _get_attr_any(model_args, ("n_layers", "num_hidden_layers"))
    n_heads = _get_attr_any(model_args, ("n_heads", "num_attention_heads"))
    n_kv_heads = _get_attr_any(
        model_args, ("n_kv_heads", "num_key_value_heads", "n_heads")
    )
    vocab_size = _get_attr_any(model_args, ("vocab_size",))
    kv_out_dim = None
    if dim is not None and n_heads not in (None, 0) and n_kv_heads is not None:
        kv_out_dim = dim * n_kv_heads // n_heads
    return {
        "dim": dim,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "kv_out_dim": kv_out_dim,
    }


def _model_args_match_state_dict(
    model_args: object, state_sig: dict[str, int | None]
) -> bool:
    args_sig = _model_args_signature(model_args)
    for key in ("dim", "n_layers", "kv_out_dim"):
        state_value = state_sig.get(key)
        args_value = args_sig.get(key)
        if (
            state_value is not None
            and args_value is not None
            and state_value != args_value
        ):
            return False
    q_out_dim = state_sig.get("q_out_dim")
    args_dim = args_sig.get("dim")
    if q_out_dim is not None and args_dim is not None and q_out_dim != args_dim:
        return False
    return True


def _format_signature(sig: dict[str, int | None]) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in sig.items() if value is not None
    )


def state_dict_fingerprint(
    state_dict: dict[str, torch.Tensor],
    *,
    max_tensors: int = 8,
) -> float:
    """Return a cheap numeric fingerprint from a few tensor scalar samples."""
    max_tensors = max(1, int(max_tensors))
    tensor_names = sorted(
        name for name, tensor in state_dict.items() if torch.is_tensor(tensor)
    )
    if not tensor_names:
        return 0.0

    if len(tensor_names) <= max_tensors:
        selected_names = tensor_names
    elif max_tensors == 1:
        selected_names = [tensor_names[0]]
    else:
        selected_indices = {
            round(idx * (len(tensor_names) - 1) / (max_tensors - 1))
            for idx in range(max_tensors)
        }
        selected_names = [tensor_names[idx] for idx in sorted(selected_indices)]

    fingerprint = 0.0
    for name_idx, name in enumerate(selected_names, start=1):
        tensor = state_dict[name]
        if tensor.numel() == 0:
            continue
        detached = tensor.detach()
        try:
            flat = detached.view(-1)
        except RuntimeError:
            continue
        sample_indices = {0, flat.numel() // 2, flat.numel() - 1}
        for sample_idx in sorted(sample_indices):
            try:
                value = float(flat[sample_idx].float().item())
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                continue
            fingerprint += value * (name_idx * 1009 + sample_idx % 997)
    return float(fingerprint)


def _resolve_torchtitan_model_args_key(
    train_spec: Any,
    state_dict: dict[str, torch.Tensor],
    requested_key: str,
) -> str:
    """Resolve TorchTitan model args against actual state_dict shapes."""
    model_args_map = train_spec.model_args
    state_sig = _state_dict_signature(state_dict)
    requested_key = requested_key.strip()
    auto_requested = requested_key.lower() in {"", "auto"}

    if not auto_requested:
        if requested_key not in model_args_map:
            available = ", ".join(sorted(str(key) for key in model_args_map))
            raise KeyError(
                f"Unknown TorchTitan model args key '{requested_key}'. "
                f"Available keys: {available}"
            )
        if _model_args_match_state_dict(model_args_map[requested_key], state_sig):
            return requested_key

    matches = [
        str(key)
        for key, model_args in model_args_map.items()
        if _model_args_match_state_dict(model_args, state_sig)
    ]
    if len(matches) == 1:
        return matches[0]

    state_text = _format_signature(state_sig) or "unknown"
    candidates = ", ".join(
        f"{key}({_format_signature(_model_args_signature(model_args))})"
        for key, model_args in model_args_map.items()
    )
    if auto_requested:
        raise ValueError(
            "Could not infer a unique TorchTitan model args key from the "
            f"state_dict shape ({state_text}). Matching keys: {matches or 'none'}. "
            f"Available keys: {candidates}"
        )
    requested_sig = _format_signature(
        _model_args_signature(model_args_map[requested_key])
    )
    raise ValueError(
        f"Configured trainer.torchtitan.dcp-model-args='{requested_key}' does not "
        f"match the incoming state_dict shape ({state_text}). "
        f"Requested key shape: {requested_sig}. "
        f"Auto-detected matches: {matches or 'none'}. "
        "Set trainer.torchtitan.dcp-model-args to the matching TorchTitan key, "
        "or use 'auto' when exactly one key matches."
    )


def _remove_path(path: str) -> None:
    """Remove a file, symlink, or directory if present."""
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def remove_empty_dirs_up_to(path: str, stop_dir: str) -> None:
    """Remove empty parent directories from path up to, but not including, stop_dir."""
    current = os.path.abspath(path)
    stop = os.path.abspath(stop_dir)
    while current != stop and os.path.commonpath([current, stop]) == stop:
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)


def cleanup_layer_paths(layer_paths: list[str]) -> None:
    """Remove layer files tracked for layer-wise communication."""
    for layer_path in dict.fromkeys(str(path) for path in layer_paths if path):
        _remove_path(layer_path)


def _replace_symlink(link_path: str, target_path: str) -> None:
    _remove_path(link_path)
    os.symlink(target_path, link_path, target_is_directory=True)


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
    return load_state_dict_from_layer_paths(layer_paths)


def load_state_dict_from_layer_paths(
    layer_paths: list[str],
) -> dict[str, torch.Tensor]:
    """Load a state_dict from persisted layer files."""
    state_dict: dict[str, torch.Tensor] = {}
    for layer_path in layer_paths:
        if not os.path.exists(layer_path):
            continue
        with open(layer_path, "rb") as file:
            layer_dict = pickle.load(file)
        for layer_name, tensor in layer_dict.items():
            state_dict[str(layer_name)] = tensor.detach().cpu()
    return state_dict


def layer_file_paths(layer_directory: str) -> list[str]:
    """Return direct layer files in deterministic order."""
    if not os.path.isdir(layer_directory):
        raise FileNotFoundError(f"Layer directory not found: {layer_directory}")
    return [
        os.path.join(layer_directory, name)
        for name in sorted(os.listdir(layer_directory))
        if name.endswith(".pt")
        and os.path.isfile(os.path.join(layer_directory, name))
    ]


def load_state_dict_from_layer_directory(
    layer_directory: str,
) -> dict[str, torch.Tensor]:
    """Load a state_dict from direct layer files in a directory."""
    return load_state_dict_from_layer_paths(layer_file_paths(layer_directory))


def state_dict_fingerprint_from_layer_paths(layer_paths: list[str]) -> float:
    """Fingerprint layer files without constructing a model object."""
    fingerprint = 0.0
    for layer_path in layer_paths:
        if not os.path.exists(layer_path):
            continue
        with open(layer_path, "rb") as file:
            layer_dict = pickle.load(file)
        fingerprint += state_dict_fingerprint(layer_dict)
    return fingerprint


def state_dict_fingerprint_from_layer_directory(layer_directory: str) -> float:
    """Fingerprint a layer directory without constructing a model object."""
    return state_dict_fingerprint_from_layer_paths(layer_file_paths(layer_directory))


def _write_state_dict_as_layer_files(
    state_dict: dict[str, torch.Tensor],
    output_directory: str,
    *,
    layer_names: list[str] | None = None,
    ready_marker: str | None = None,
) -> None:
    """Publish layer files atomically after a successful conversion."""
    os.makedirs(output_directory, exist_ok=True)
    names = layer_names or list(state_dict.keys())
    missing = [name for name in names if name not in state_dict]
    if missing:
        raise KeyError(f"Converted checkpoint is missing layers: {missing[:3]}")

    temporary_directory = tempfile.mkdtemp(
        prefix=".torchtitan-layers-", dir=output_directory
    )
    try:
        expected_files: set[str] = set()
        for layer_name in names:
            file_name = f"{sanitize_layer_name(layer_name)}.pt"
            expected_files.add(file_name)
            temporary_path = os.path.join(temporary_directory, file_name)
            with open(temporary_path, "wb") as file:
                pickle.dump(
                    {layer_name: state_dict[layer_name].detach().cpu()},
                    file,
                )

        for existing_path in layer_file_paths(output_directory):
            if os.path.basename(existing_path) not in expected_files:
                _remove_path(existing_path)

        for file_name in expected_files:
            os.replace(
                os.path.join(temporary_directory, file_name),
                os.path.join(output_directory, file_name),
            )

        if ready_marker:
            temporary_marker = f"{ready_marker}.tmp"
            with open(temporary_marker, "w", encoding="utf-8") as file:
                file.write("ready\n")
            os.replace(temporary_marker, ready_marker)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def convert_layer_directory_to_dcp(
    input_directory: str,
    output_directory: str,
    *,
    train_spec_name: str,
    model_args_key: str,
    dcp_threads: int,
) -> None:
    """Convert persisted HF-like layer files into TorchTitan DCP."""
    state_dict = load_state_dict_from_layer_directory(input_directory)
    if not state_dict:
        raise ValueError(f"No layer files found in {input_directory}")

    resolved_model_args = model_args_key
    try:
        import torchtitan.protocols.train_spec as train_spec_module
    except Exception:
        pass
    else:
        train_spec = train_spec_module.get_train_spec(train_spec_name)
        resolved_model_args = _resolve_torchtitan_model_args_key(
            train_spec, state_dict, model_args_key
        )

    _save_state_dict_as_dcp(
        state_dict,
        output_directory,
        train_spec_name=train_spec_name,
        model_args_key=resolved_model_args,
        dcp_threads=dcp_threads,
    )


def convert_dcp_to_layer_directory(
    input_directory: str,
    reference_directory: str,
    output_directory: str,
    *,
    train_spec_name: str,
    model_args_key: str,
    dcp_threads: int,
    ready_marker: str | None = None,
) -> None:
    """Convert a TorchTitan DCP checkpoint into atomically published layers."""
    reference_state = load_state_dict_from_layer_directory(reference_directory)
    if not reference_state:
        raise ValueError(f"No reference layer files found in {reference_directory}")

    resolved_model_args = model_args_key
    try:
        import torchtitan.protocols.train_spec as train_spec_module
    except Exception:
        pass
    else:
        train_spec = train_spec_module.get_train_spec(train_spec_name)
        resolved_model_args = _resolve_torchtitan_model_args_key(
            train_spec, reference_state, model_args_key
        )

    state_dict = _load_state_dict_from_dcp(
        input_directory,
        train_spec_name=train_spec_name,
        model_args_key=resolved_model_args,
        reference_state_dict=reference_state,
    )
    _write_state_dict_as_layer_files(
        state_dict,
        output_directory,
        layer_names=list(reference_state.keys()),
        ready_marker=ready_marker,
    )


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


def _empty_like_tensor_structure(value: Any) -> Any:
    """Clone a checkpoint structure with empty CPU tensors for DCP loading."""
    if torch.is_tensor(value):
        return torch.empty_like(value.detach(), device="cpu")
    if isinstance(value, dict):
        return {
            key: _empty_like_tensor_structure(item)
            for key, item in value.items()
        }
    return value


def _dcp_load_into(state_dict: dict[str, Any], reader: Any) -> None:
    """Load a DCP state_dict across PyTorch versions."""
    from torch.distributed import checkpoint as dcp

    try:
        dcp.load(state_dict, storage_reader=reader, no_dist=True)
    except TypeError as exc:
        if "no_dist" not in str(exc):
            raise
        dcp.load(state_dict, storage_reader=reader)


def _load_first_matching_dcp_state(
    input_dir: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load DCP into the first candidate structure matching checkpoint keys."""
    from torch.distributed import checkpoint as dcp
    from torch.distributed.checkpoint.api import CheckpointException

    last_error: CheckpointException | None = None
    for candidate in candidates:
        try:
            _dcp_load_into(candidate, dcp.filesystem.FileSystemReader(input_dir))
            return candidate
        except CheckpointException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"No DCP load candidates provided for {input_dir}")


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
    try:
        titan_state_dict = sd_adapter.from_hf(state_dict)
    except Exception as exc:
        state_text = _format_signature(_state_dict_signature(state_dict)) or "unknown"
        args_text = _format_signature(_model_args_signature(model_args)) or "unknown"
        raise RuntimeError(
            "TorchTitan HF-to-DCP conversion failed with "
            f"dcp-train-spec='{train_spec_name}', "
            f"dcp-model-args='{model_args_key}'. "
            f"Incoming state_dict shape: {state_text}. "
            f"TorchTitan model args shape: {args_text}."
        ) from exc
    dcp.save(titan_state_dict, storage_writer=writer)


def _load_state_dict_from_dcp(
    input_dir: str,
    *,
    train_spec_name: str,
    model_args_key: str,
    reference_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load state_dict from DCP format, converting back to HF-like mapping."""
    reference_hf_state = _normalize_state_dict_for_hf(reference_state_dict)

    try:
        import torchtitan.protocols.train_spec as train_spec_module
    except Exception:
        checkpoint_dict = _load_first_matching_dcp_state(
            input_dir,
            [
                _empty_like_tensor_structure(reference_hf_state),
                {"model": _empty_like_tensor_structure(reference_hf_state)},
            ],
        )
        loaded_state = _normalize_state_dict_for_hf(extract_state_dict(checkpoint_dict))
        if not loaded_state:
            raise ValueError(f"DCP checkpoint loaded no tensors from {input_dir}")
        return loaded_state

    train_spec = train_spec_module.get_train_spec(train_spec_name)
    model_args = train_spec.model_args[model_args_key]
    sd_adapter = train_spec.state_dict_adapter(model_args, None)
    titan_reference = sd_adapter.from_hf(reference_hf_state)
    checkpoint_dict = _load_first_matching_dcp_state(
        input_dir,
        [
            _empty_like_tensor_structure(titan_reference),
            {"model": _empty_like_tensor_structure(titan_reference)},
        ],
    )
    titan_state = (
        checkpoint_dict["model"]
        if isinstance(checkpoint_dict.get("model"), dict)
        else checkpoint_dict
    )
    try:
        hf_state = sd_adapter.to_hf(titan_state)
    except Exception as exc:
        args_text = _format_signature(_model_args_signature(model_args)) or "unknown"
        raise RuntimeError(
            "TorchTitan DCP-to-HF conversion failed with "
            f"dcp-train-spec='{train_spec_name}', "
            f"dcp-model-args='{model_args_key}'. "
            f"TorchTitan model args shape: {args_text}."
        ) from exc
    loaded_state = _normalize_state_dict_for_hf(extract_state_dict(hf_state))
    if not loaded_state:
        raise ValueError(f"DCP checkpoint loaded no tensors from {input_dir}")
    return loaded_state


def run_torchtitan_training(
    cfg: DictConfig,
    context: Context,
    state_dict: dict[str, torch.Tensor] | None,
    *,
    server_round: int | None = None,
    layer_paths: list[str] | None = None,
    output_layer_dir: str | None = None,
) -> dict[str, torch.Tensor] | None:
    """Execute TorchTitan training and return state only for legacy paths.

    Layerwise DCP training keeps the state on disk. The scheduler job performs
    both DCP conversions, so the ClientApp never needs to materialize the full
    model state.
    """
    trainer_cfg = getattr(cfg, "trainer", {})
    titan_cfg = getattr(trainer_cfg, "torchtitan", {})
    command = str(getattr(titan_cfg, "command", "")).strip()

    round_id = (
        int(server_round)
        if server_round is not None
        else int(_config_value(context, "current-round", 0))
    )
    output_dir = os.path.join(layer_dir(context), "torchtitan")
    os.makedirs(output_dir, exist_ok=True)
    conversion_profile_path = os.path.join(
        os.path.dirname(output_dir), "torchtitan_conversion_profile.jsonl"
    )
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
    layerwise_dcp = dcp_enabled and layer_paths is not None
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
            _config_value(context, "trainer.torchtitan.dcp_model_args", "auto"),
        )
    ).strip()
    dcp_threads = int(
        _config_value(
            context,
            "trainer.torchtitan.dcp-threads",
            _config_value(context, "trainer.torchtitan.dcp_threads", 8),
        )
    )

    workdir = str(getattr(titan_cfg, "workdir", "")).strip() or None
    scheduler_backend = str(
        _config_value(context, "scheduler.backend", "local")
    ).strip().lower()
    dry_run = _as_bool(
        _config_value(
            context,
            "trainer.dry-run",
            _config_value(context, "trainer.dry_run", False),
        ),
        default=False,
    )
    client_name = _config_str(context, "client.name", str(context.node_id))
    dataset_name = _config_str(
        context, "client.dataset-name", _config_str(context, "dataset.name", "")
    )
    dataset_path = _config_str(context, "client.dataset-path", "")
    hf_assets_path = _config_str(context, "client.hf-assets-path", "")
    train_steps = int(
        _config_value(
            context,
            "client.train-steps",
            _config_value(context, "trainer.train-steps", 0),
        )
    )
    model_name = _config_str(context, "model.name", "")
    model_flavor = _config_str(context, "trainer.torchtitan.model-flavor", "")
    python_exec = _config_str(context, "trainer.python-exec", "python")
    torchtitan_entrypoint = _config_str(context, "trainer.torchtitan.entrypoint", "")
    client_workspace = _config_str(
        context,
        "client.workspace",
        workdir or os.getcwd(),
    )
    client_workspace = os.path.abspath(
        os.path.expandvars(os.path.expanduser(client_workspace))
    )
    dump_folder = _config_str(context, "trainer.dump-folder", "").strip()
    if not dump_folder:
        dump_folder = os.path.join(output_dir, "dump")
    config_filename = _config_str(
        context,
        "trainer.torchtitan.config-filename",
        "torchtitan_generated.toml",
    )
    num_nodes = int(
        _config_value(
            context,
            "trainer.num-nodes",
            _config_value(context, "trainer.num_nodes", 1),
        )
    )
    if not workdir:
        workdir = client_workspace
    os.makedirs(dump_folder, exist_ok=True)
    resolved_dcp_model_args = dcp_model_args
    if dcp_enabled and not layerwise_dcp:
        try:
            import torchtitan.protocols.train_spec as train_spec_module
        except Exception:
            pass
        else:
            if state_dict is None:
                raise ValueError(
                    "state_dict is required for non-layerwise DCP training"
                )
            train_spec = train_spec_module.get_train_spec(dcp_train_spec)
            resolved_dcp_model_args = _resolve_torchtitan_model_args_key(
                train_spec, state_dict, dcp_model_args
            )
    model_cache_path = _sanitize_model_cache_path(
        model_name,
        f"{dcp_train_spec}-{resolved_dcp_model_args}",
    )
    dcp_cache_dir = os.path.join(
        client_workspace, "flower_dcp_cache", model_cache_path
    )
    checkpoint_dir = os.path.join(dump_folder, "checkpoint")
    step0_dcp_dir = os.path.join(checkpoint_dir, "step-0")
    final_dcp_dir = os.path.join(checkpoint_dir, f"step-{train_steps}")
    input_layer_dir = (
        os.path.abspath(os.path.dirname(layer_paths[0]))
        if layer_paths
        else ""
    )
    output_layer_dir = (
        os.path.abspath(output_layer_dir or input_layer_dir)
        if layerwise_dcp
        else ""
    )
    output_layers_ready = (
        os.path.join(output_layer_dir, ".torchtitan_layers_ready")
        if layerwise_dcp
        else ""
    )
    cache_available = round_id <= 1 and _dcp_checkpoint_exists(dcp_cache_dir)
    conversion_dir = (
        dcp_cache_dir if round_id <= 1 else input_dcp_dir
    )
    env = os.environ.copy()
    scheduler_env = {
        "FLWR_TORCHTITAN_INPUT_STATE": input_state_path,
        "FLWR_TORCHTITAN_OUTPUT_STATE": output_state_path,
        "FLWR_TORCHTITAN_INPUT_DCP_DIR": input_dcp_dir,
        "FLWR_TORCHTITAN_OUTPUT_DCP_DIR": output_dcp_dir,
        "FLWR_TORCHTITAN_DCP_CACHE_DIR": dcp_cache_dir,
        "FLWR_TORCHTITAN_CHECKPOINT_DIR": checkpoint_dir,
        "FLWR_TORCHTITAN_STEP0_DCP_DIR": step0_dcp_dir,
        "FLWR_TORCHTITAN_FINAL_DCP_DIR": final_dcp_dir,
        "FLWR_TORCHTITAN_INPUT_LAYERS_DIR": input_layer_dir,
        "FLWR_TORCHTITAN_DCP_CONVERSION_DIR": conversion_dir,
        "FLWR_TORCHTITAN_OUTPUT_LAYERS_DIR": output_layer_dir,
        "FLWR_TORCHTITAN_OUTPUT_LAYERS_READY": output_layers_ready,
        "FLWR_TORCHTITAN_CONVERSION_PROFILE": conversion_profile_path,
        "FLWR_FLOWERTUNE_LLM_ROOT": FLOWERTUNE_LLM_ROOT,
        "FLWR_RUN_ID": str(context.run_id),
        "FLWR_NODE_ID": str(context.node_id),
    }
    env.update(scheduler_env)
    scheduler_account = _config_str(context, "scheduler.account", "")
    scheduler_partition = _config_str(context, "scheduler.partition", "")
    scheduler_qos = _config_str(context, "scheduler.qos", "")
    scheduler_gpus = _config_str(context, "scheduler.gpus", "")
    scheduler_cpus_per_task = _config_str(context, "scheduler.cpus-per-task", "")
    scheduler_mem = _config_str(context, "scheduler.mem", "")
    scheduler_time = _config_str(context, "scheduler.time", "")
    scheduler_extra_args = _config_str(context, "scheduler.extra-args", "")
    env_setup = _config_str(context, "trainer.env-setup", "")

    dcp_conversion_command = ""
    dcp_to_layers_command = ""
    if layerwise_dcp:
        dcp_conversion_command = _python_module_command(
            python_exec,
            "flowertune_llm.dcp_converter",
            [
                "--direction",
                "to-dcp",
                "--input-dir",
                input_layer_dir,
                "--output-dir",
                conversion_dir,
                "--train-spec",
                dcp_train_spec,
                "--model-args",
                dcp_model_args,
                "--threads",
                str(dcp_threads),
            ],
        )
        dcp_to_layers_command = _python_module_command(
            python_exec,
            "flowertune_llm.dcp_converter",
            [
                "--direction",
                "to-layers",
                "--input-dir",
                final_dcp_dir,
                "--reference-dir",
                input_layer_dir,
                "--output-dir",
                output_layer_dir,
                "--ready-marker",
                output_layers_ready,
                "--train-spec",
                dcp_train_spec,
                "--model-args",
                dcp_model_args,
                "--threads",
                str(dcp_threads),
            ],
        )

    render_context: dict[str, Any] = {
        "run_id": context.run_id,
        "round_id": round_id,
        "node_id": context.node_id,
        "client_name": client_name,
        "model_name": model_name,
        "model_flavor": model_flavor,
        "dcp_train_spec": dcp_train_spec,
        "dcp_model_args": resolved_dcp_model_args,
        "hf_assets_path": hf_assets_path,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "train_steps": train_steps,
        "steps_per_round": train_steps,
        "input_checkpoint_path": input_state_path,
        "output_checkpoint_path": output_state_path,
        "input_dcp_dir": input_dcp_dir,
        "output_dcp_dir": output_dcp_dir,
        "dcp_cache_dir": dcp_cache_dir,
        "checkpoint_dir": checkpoint_dir,
        "step0_dcp_dir": step0_dcp_dir,
        "final_dcp_dir": final_dcp_dir,
        "input_layers_dir": input_layer_dir,
        "dcp_conversion_dir": conversion_dir,
        "output_layers_dir": output_layer_dir,
        "output_layers_ready": output_layers_ready,
        "conversion_profile_path": conversion_profile_path,
        "flowertune_llm_root": FLOWERTUNE_LLM_ROOT,
        "dcp_conversion_command": dcp_conversion_command,
        "dcp_to_layers_command": dcp_to_layers_command,
        "work_dir": output_dir,
        "client_workspace": client_workspace,
        "dump_folder": dump_folder,
        "config_filename": config_filename,
        "num_nodes": num_nodes,
        "log_path": os.path.join(output_dir, "trainer.log"),
        "scheduler_backend": scheduler_backend,
        "scheduler_account": scheduler_account,
        "scheduler_partition": scheduler_partition,
        "scheduler_qos": scheduler_qos,
        "scheduler_gpus": scheduler_gpus,
        "scheduler_cpus_per_task": scheduler_cpus_per_task,
        "scheduler_mem": scheduler_mem,
        "scheduler_time": scheduler_time,
        "scheduler_extra_args": scheduler_extra_args,
        "env_setup": env_setup,
        "python_exec": python_exec,
        "torchtitan_entrypoint": torchtitan_entrypoint,
        "torchtitan_command": command,
        "torchtitan_config_path": os.path.join(output_dir, config_filename),
    }

    if scheduler_backend not in {"", "none", "local", "slurm", "flux"}:
        raise ValueError(
            f"Unsupported scheduler.backend '{scheduler_backend}'. "
            "Use local, slurm, or flux."
        )

    if _config_str(context, "trainer.torchtitan.config-template", "").strip():
        config_template = _template_path(
            context,
            "trainer.torchtitan.config-template",
            "torchtitan.toml.j2",
        )
        rendered_toml = _render_template_file(config_template, render_context)
        with open(
            render_context["torchtitan_config_path"], "w", encoding="utf-8"
        ) as file:
            file.write(rendered_toml)

    def write_scheduler_script(backend: str) -> str:
        """Render the configured scheduler script and return its path."""
        if backend == "local":
            script_path = os.path.join(output_dir, "torchtitan_local.sh")
            template_path = _template_path(
                context,
                "scheduler.slurm.script-template",
                "slurm_train.sh.j2",
            )
        elif backend == "slurm":
            script_path = os.path.join(output_dir, "torchtitan_slurm.sh")
            template_path = _template_path(
                context,
                "scheduler.slurm.script-template",
                "slurm_train.sh.j2",
            )
        elif backend == "flux":
            script_path = os.path.join(output_dir, "torchtitan_flux.sh")
            template_path = _template_path(
                context,
                "scheduler.flux.script-template",
                "flux_train.sh.j2",
            )
        else:
            return ""

        render_context["workdir"] = workdir or ""
        render_context["script_path"] = script_path
        script_text = _render_template_file(template_path, render_context)
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(script_text)
        os.chmod(script_path, 0o755)
        return script_path

    custom_scheduler_template = False
    if scheduler_backend == "slurm":
        custom_scheduler_template = bool(
            _config_str(context, "scheduler.slurm.script-template", "").strip()
        )
    elif scheduler_backend == "flux":
        custom_scheduler_template = bool(
            _config_str(context, "scheduler.flux.script-template", "").strip()
        )

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

    if dry_run:
        script_path = write_scheduler_script(scheduler_backend)
        dry_run_report = os.path.join(output_dir, "dry_run_summary.txt")
        with open(dry_run_report, "w", encoding="utf-8") as file:
            file.write(
                dedent(
                    f"""\
                    dry_run=true
                    scheduler.backend={scheduler_backend}
                    command={command}
                    workdir={workdir or ''}
                    script_path={script_path}
                    run_id={context.run_id}
                    node_id={context.node_id}
                    client.name={client_name}
                    dataset.name={dataset_name}
                    dataset.path={dataset_path}
                    """
                )
            )
        return (
            _normalize_state_dict_for_hf(state_dict)
            if state_dict is not None
            else None
        )

    if not command and (
        scheduler_backend in {"", "none", "local"} or not custom_scheduler_template
    ):
        raise ValueError(
            "trainer.backend is 'torchtitan' but no TorchTitan command was "
            "provided. Set trainer.torchtitan.command, set trainer.dry-run=true, "
            "or provide scheduler.slurm.script-template / "
            "scheduler.flux.script-template containing the training command."
        )

    # Clear stale handoff artifacts before launching TorchTitan. Otherwise a
    # previous successful or failed run could be mistaken for fresh output.
    _remove_path(input_state_path)
    _remove_path(output_state_path)
    _remove_path(input_dcp_dir)
    _remove_path(output_dcp_dir)
    _remove_path(step0_dcp_dir)
    _remove_path(output_layers_ready)
    _remove_path(conversion_profile_path)

    if layerwise_dcp:
        if cache_available:
            _replace_symlink(input_dcp_dir, dcp_cache_dir)
        else:
            _remove_path(conversion_dir)
    elif dcp_enabled:
        if round_id <= 1 and cache_available:
            _replace_symlink(input_dcp_dir, dcp_cache_dir)
        else:
            conversion_dir = dcp_cache_dir if round_id <= 1 else input_dcp_dir
            _remove_path(conversion_dir)
            if state_dict is None:
                raise ValueError("state_dict is required for DCP training")
            _save_state_dict_as_dcp(
                state_dict,
                conversion_dir,
                train_spec_name=dcp_train_spec,
                model_args_key=resolved_dcp_model_args,
                dcp_threads=dcp_threads,
            )
            if conversion_dir == dcp_cache_dir:
                _replace_symlink(input_dcp_dir, dcp_cache_dir)
    else:
        if state_dict is None:
            raise ValueError("state_dict is required for non-DCP training")
        torch.save(state_dict, input_state_path)

    if scheduler_backend in {"", "none", "local"}:
        if layerwise_dcp:
            local_script = write_scheduler_script("local")
            result = subprocess.run(
                [local_script],
                env=env,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )
        else:
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

        submit_parts = [slurm_submit]
        if slurm_wait:
            submit_parts.append("--wait")
        submit_parts.append("--parsable")
        if scheduler_account:
            submit_parts.extend(["--account", scheduler_account])
        if scheduler_partition:
            submit_parts.extend(["--partition", scheduler_partition])
        if scheduler_qos:
            submit_parts.extend(["--qos", scheduler_qos])
        if scheduler_time:
            submit_parts.extend(["--time", scheduler_time])
        if scheduler_mem:
            submit_parts.extend(["--mem", scheduler_mem])
        if scheduler_gpus:
            submit_parts.extend(["--gpus", scheduler_gpus])
        if scheduler_cpus_per_task:
            submit_parts.extend(["--cpus-per-task", scheduler_cpus_per_task])
        if scheduler_extra_args:
            submit_parts.extend(shlex.split(scheduler_extra_args))
        if slurm_extra_args:
            submit_parts.extend(shlex.split(slurm_extra_args))
        submit_parts.append(write_scheduler_script("slurm"))

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
        flux_parts = shlex.split(flux_run)
        if (
            len(flux_parts) >= 2
            and os.path.basename(flux_parts[0]) == "flux"
            and flux_parts[1] == "batch"
        ):
            raise ValueError(
                "scheduler.flux.run-command must run the generated script in "
                "the foreground, for example 'flux run'. 'flux batch' submits "
                "asynchronously, so Flower cannot wait for TorchTitan to write "
                "FLWR_TORCHTITAN_OUTPUT_DCP_DIR."
            )
        if scheduler_extra_args:
            flux_parts.extend(shlex.split(scheduler_extra_args))
        if flux_extra_args:
            flux_parts.extend(shlex.split(flux_extra_args))
        flux_parts.append(write_scheduler_script("flux"))

        result = subprocess.run(
            flux_parts,
            env=env,
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        _remove_path(step0_dcp_dir)
        _remove_path(input_dcp_dir)
        _remove_path(output_dcp_dir)
        if layerwise_dcp and not cache_available:
            _remove_path(conversion_dir)
        raise RuntimeError(
            "TorchTitan command failed with exit code "
            f"{result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if dcp_enabled:
        _remove_path(step0_dcp_dir)
        _remove_path(input_dcp_dir)

    if layerwise_dcp:
        if not os.path.isdir(output_dcp_dir):
            _remove_path(output_layers_ready)
            _remove_path(output_dcp_dir)
            _remove_path(output_dir)
            raise FileNotFoundError(
                "TorchTitan command completed but did not write the output DCP "
                f"directory: {output_dcp_dir}"
            )
        if not os.path.exists(output_layers_ready):
            _remove_path(output_dcp_dir)
            _remove_path(output_dir)
            raise FileNotFoundError(
                "TorchTitan job completed without publishing converted layer "
                f"files: {output_layers_ready}"
            )
        _remove_path(output_layers_ready)
        _remove_path(output_dcp_dir)
        _remove_path(output_dir)
        return None

    if os.path.exists(output_state_path):
        payload = torch.load(output_state_path, map_location="cpu")
        trained_state = extract_state_dict(payload)
        _remove_path(input_state_path)
        _remove_path(output_state_path)
        trained_state = _normalize_state_dict_for_hf(trained_state)
        _remove_path(output_dir)
        return trained_state

    if os.path.isdir(output_dcp_dir):
        trained_state = _load_state_dict_from_dcp(
            output_dcp_dir,
            train_spec_name=dcp_train_spec,
            model_args_key=resolved_dcp_model_args,
            reference_state_dict=state_dict or {},
        )
        _remove_path(output_dcp_dir)
        _remove_path(output_dir)
        return trained_state

    if os.path.islink(output_dcp_dir):
        raise FileNotFoundError(
            "TorchTitan command wrote an output_state.dcp symlink, but its "
            f"target is not a readable directory: {output_dcp_dir} -> "
            f"{os.readlink(output_dcp_dir)}"
        )

    raise FileNotFoundError(
        "TorchTitan command completed but did not write either "
        f"{output_state_path} or {output_dcp_dir}. "
        "Set FLWR_TORCHTITAN_OUTPUT_STATE or FLWR_TORCHTITAN_OUTPUT_DCP_DIR."
    )
