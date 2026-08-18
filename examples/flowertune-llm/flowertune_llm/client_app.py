"""flowertune-llm: A Flower / FlowerTune app."""

from __future__ import annotations

import os
import pickle
import warnings
from time import perf_counter

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model
from flowertune_llm.task import (
    CachedLayer,
    chunk_key,
    cleanup_layer_paths,
    context_layer_key,
    context_path_key,
    flush_cached_layer,
    flush_caches_for_context,
    is_last_batch,
    layer_dir,
    load_layer_from_disk,
    load_state_dict_from_layer_files,
    parse_chunk_ranges,
    remove_empty_dirs_up_to,
    read_conversion_profile,
    run_torchtitan_training,
    sanitize_layer_name,
    shape_from_text,
    state_dict_fingerprint,
    state_dict_fingerprint_from_layer_paths,
    training_disabled,
    torchtitan_dcp_enabled,
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

STATE_LAYER_NAMES = "layer_names"
STATE_LAYER_PATHS = "layer_paths"
STATE_LAYER_IDX = "layer_idx"
STATE_NUM_EXAMPLES = "num_examples"


# Flower ClientApp
app = ClientApp()


_DOWNLOAD_LAYER_CACHE: dict[tuple[int, int, str], CachedLayer] = {}
_COMMS_LAYER_CACHE: dict[tuple[int, int, str], CachedLayer] = {}


def _layer_file_path(context: Context, layer_name: str) -> str:
    return os.path.join(layer_dir(context), f"{sanitize_layer_name(layer_name)}.pt")


def _restore_layer_state_from_names(
    context: Context,
    layer_names: list[str],
    *,
    require_all: bool,
) -> None:
    """Restore layer-wise context state from deterministic layer file paths."""
    if not layer_names:
        return

    layer_paths = [_layer_file_path(context, layer_name) for layer_name in layer_names]
    missing = [path for path in layer_paths if not os.path.exists(path)]
    if missing and require_all:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
        raise FileNotFoundError(
            "Layer-wise model was marked as preloaded, but expected layer files "
            f"were not found: {preview}{suffix}"
        )
    existing_pairs = [
        (layer_name, layer_path)
        for layer_name, layer_path in zip(layer_names, layer_paths, strict=True)
        if os.path.exists(layer_path)
    ]
    if not existing_pairs:
        return

    context.state[STATE_LAYER_NAMES] = ConfigRecord(
        {"names": [name for name, _ in existing_pairs]}
    )
    context.state[STATE_LAYER_PATHS] = ConfigRecord(
        {"paths": [path for _, path in existing_pairs]}
    )
    context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": 0})
    context.state[STATE_NUM_EXAMPLES] = ConfigRecord({"num_examples": 1})


def _flush_download_caches_for_context(context: Context) -> None:
    """Flush and drop all cached download layers for a run/node."""
    flush_caches_for_context(
        _DOWNLOAD_LAYER_CACHE, context, flush_before_drop=True
    )


def _flush_comms_caches_for_context(context: Context) -> None:
    """Drop all cached comms layers for a run/node."""
    flush_caches_for_context(_COMMS_LAYER_CACHE, context, flush_before_drop=False)


def _cleanup_layer_files_for_context(
    context: Context, layer_paths: list[str] | None = None
) -> None:
    """Remove persisted layer files and clear layer-wise context state."""
    if layer_paths is None:
        layer_paths = (
            list(context.state[STATE_LAYER_PATHS]["paths"])
            if STATE_LAYER_PATHS in context.state
            else []
        )
    _flush_download_caches_for_context(context)
    _flush_comms_caches_for_context(context)
    cleanup_layer_paths(layer_paths)
    node_layer_dir = layer_dir(context)
    layer_base_dir = os.path.dirname(os.path.dirname(node_layer_dir))
    remove_empty_dirs_up_to(node_layer_dir, layer_base_dir)
    context.state.pop(STATE_LAYER_NAMES, None)
    context.state.pop(STATE_LAYER_PATHS, None)
    context.state.pop(STATE_LAYER_IDX, None)
    context.state.pop(STATE_NUM_EXAMPLES, None)


def _persist_layer_files(
    context: Context,
    state_dict: dict[str, torch.Tensor],
    layer_names: list[str],
) -> None:
    """Persist selected state_dict layers and update layer-wise context state."""
    write_dir = layer_dir(context)
    previous_layer_paths = (
        list(context.state[STATE_LAYER_PATHS]["paths"])
        if STATE_LAYER_PATHS in context.state
        else []
    )
    cleanup_layer_paths(previous_layer_paths)
    serialized_layer_paths: list[str] = []
    for layer_name in layer_names:
        if layer_name not in state_dict:
            continue
        file_name = f"{sanitize_layer_name(layer_name)}.pt"
        file_path = os.path.join(write_dir, file_name)
        serialized_layer_paths.append(file_path)
        with open(file_path, "wb") as file:
            pickle.dump({layer_name: state_dict[layer_name]}, file)

    context.state[STATE_LAYER_NAMES] = ConfigRecord({"names": layer_names})
    context.state[STATE_LAYER_PATHS] = ConfigRecord({"paths": serialized_layer_paths})
    context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": 0})
    context.state[STATE_NUM_EXAMPLES] = ConfigRecord({"num_examples": 1})


def _debug_add_noise_to_state_dict(
    state_dict: dict[str, torch.Tensor], scale: float
) -> tuple[str, float, float] | None:
    """Perturb one floating tensor to force a distinct outgoing payload."""
    if scale == 0.0:
        return None

    for layer_name in sorted(state_dict):
        tensor = state_dict[layer_name]
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        if tensor.numel() == 0:
            continue
        try:
            flat = tensor.detach().view(-1)
        except RuntimeError:
            continue
        before = float(flat[0].float().item())
        with torch.no_grad():
            flat[0].add_(float(scale))
        after = float(flat[0].float().item())
        return layer_name, before, after

    return None


@app.train("layer_wise_download")
def train_download(msg: Message, context: Context):
    """Receive layer chunks from the server and persist to disk."""
    t0 = perf_counter()
    if msg.content is None or "arrays" not in msg.content or "config" not in msg.content:
        return Message(content=RecordDict({"metrics": MetricRecord()}), reply_to=msg)

    config = msg.content["config"]
    arrays = msg.content["arrays"]
    entries: list[tuple[int | None, str, list[int], int, int, bool]] = []
    if "download_layer_names" in config:
        layer_idxs = (
            [int(v) for v in list(config["download_layer_idxs"])]
            if "download_layer_idxs" in config
            else list(range(len(config["download_layer_names"])))
        )
        layer_names = [str(v) for v in list(config["download_layer_names"])]
        layer_shapes = [str(v) for v in list(config["download_layer_shapes"])]
        chunk_starts = [int(v) for v in list(config["download_chunk_starts"])]
        chunk_ends = [int(v) for v in list(config["download_chunk_ends"])]
        is_last_values = list(config["download_is_last_chunk"])
        range_count = min(
            len(layer_idxs),
            len(layer_names),
            len(layer_shapes),
            len(chunk_starts),
            len(chunk_ends),
            len(is_last_values),
        )
        for idx in range(range_count):
            entries.append(
                (
                    layer_idxs[idx],
                    layer_names[idx],
                    shape_from_text(layer_shapes[idx]),
                    chunk_starts[idx],
                    chunk_ends[idx],
                    bool(is_last_values[idx]),
                )
            )
    else:
        layer_name = str(config.get("layer_name", ""))
        if layer_name:
            layer_shape = [int(x) for x in list(config.get("layer_shape", []))]
            chunk_ranges = parse_chunk_ranges(config)
            for start, end in chunk_ranges:
                entries.append(
                    (None, layer_name, layer_shape, start, end, is_last_batch(config))
                )

    if not entries:
        return Message(content=RecordDict({"metrics": MetricRecord()}), reply_to=msg)

    layer_base_dir = layer_dir(context)
    touched_layers: list[tuple[int | None, str, str]] = []
    touched_layer_paths_seen: set[str] = set()

    for layer_idx, layer_name, layer_shape, start, end, is_last_chunk in entries:
        chunk_name = chunk_key(layer_name, start, end)
        array = arrays.pop(chunk_name, None)
        if array is None:
            array = arrays.pop(layer_name, None)
        if array is None:
            continue
        incoming = torch.from_numpy(array.numpy())
        del array
        incoming = incoming.detach().cpu()

        file_name = f"{sanitize_layer_name(layer_name)}.pt"
        file_path = os.path.join(layer_base_dir, file_name)
        cache_key = context_layer_key(context, layer_name)
        cached = _DOWNLOAD_LAYER_CACHE.get(cache_key)
        if cached is None:
            loaded = load_layer_from_disk(file_path, layer_name)
            if loaded is None:
                if getattr(incoming, "ndim", 0) == 0 or not layer_shape:
                    loaded = incoming.clone()
                else:
                    loaded = torch.zeros(
                        tuple(int(x) for x in layer_shape),
                        dtype=incoming.dtype,
                    )
            cached = CachedLayer(
                layer_name=layer_name,
                layer_path=file_path,
                tensor=loaded,
            )
            _DOWNLOAD_LAYER_CACHE[cache_key] = cached

        if (
            getattr(cached.tensor, "ndim", 0) == 0
            or getattr(incoming, "ndim", 0) == 0
            or end <= start
        ):
            cached.tensor = incoming.clone()
        else:
            cached.tensor[start:end] = incoming
        cached.dirty = True

        # Persist every chunk because deployment can execute each download
        # message in a fresh ClientApp process. Keeping partial chunks only in
        # the process-local cache can corrupt split layers.
        flush_cached_layer(_DOWNLOAD_LAYER_CACHE, cache_key)
        if is_last_chunk:
            _DOWNLOAD_LAYER_CACHE.pop(cache_key, None)

        if file_path not in touched_layer_paths_seen:
            touched_layer_paths_seen.add(file_path)
            touched_layers.append((layer_idx, layer_name, file_path))

    # Keep context state aligned for subsequent train/train_comms calls.
    layer_paths: list[str] = []
    if STATE_LAYER_PATHS in context.state:
        layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])

    layer_names: list[str] = []
    if STATE_LAYER_NAMES in context.state:
        layer_names = list(context.state[STATE_LAYER_NAMES]["names"])

    for layer_idx, layer_name, file_path in touched_layers:
        if layer_idx is None:
            if file_path not in layer_paths:
                layer_paths.append(file_path)
            if layer_name not in layer_names:
                layer_names.append(layer_name)
            continue

        while len(layer_paths) <= layer_idx:
            layer_paths.append("")
        while len(layer_names) <= layer_idx:
            layer_names.append("")
        layer_paths[layer_idx] = file_path
        layer_names[layer_idx] = layer_name

    context.state[STATE_LAYER_PATHS] = ConfigRecord({"paths": layer_paths})
    context.state[STATE_LAYER_NAMES] = ConfigRecord({"names": layer_names})

    t1 = perf_counter()
    metrics = MetricRecord({"profile.client.train_download.ms": (t1 - t0) * 1000.0})
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)


@app.train()
def train(msg: Message, context: Context):
    """Run training (or optionally skip it) and prepare layer/state responses."""
    t0 = perf_counter()
    _flush_download_caches_for_context(context)
    _flush_comms_caches_for_context(context)

    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    aggregation_mode = getattr(cfg, "aggregation", {}).get("mode", "layerwise")
    trainer_backend = str(getattr(getattr(cfg, "trainer", {}), "backend", "none"))
    disable_train = training_disabled(context)
    model_preloaded = bool(
        msg.content
        and "config" in msg.content
        and msg.content["config"].get("model_preloaded", False)
    )
    config = msg.content["config"] if msg.content and "config" in msg.content else {}
    if (
        aggregation_mode != "all_at_once"
        and model_preloaded
        and STATE_LAYER_PATHS not in context.state
        and "layer_names" in config
    ):
        _restore_layer_state_from_names(
            context,
            [str(layer_name) for layer_name in list(config["layer_names"])],
            require_all=True,
        )

    # If layerwise model was streamed from server already, skip full model load.
    if (
        trainer_backend == "none"
        and aggregation_mode != "all_at_once"
        and model_preloaded
        and STATE_LAYER_PATHS in context.state
    ):
        layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
        if STATE_LAYER_NAMES not in context.state:
            layer_names = [
                os.path.splitext(os.path.basename(path))[0] for path in layer_paths
            ]
            context.state[STATE_LAYER_NAMES] = ConfigRecord({"names": layer_names})
        context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": 0})
        context.state[STATE_NUM_EXAMPLES] = ConfigRecord({"num_examples": 1})
        t1 = perf_counter()
        metrics = MetricRecord(
            {
                "train_loss": 0.0,
                "num-examples": 1,
                "profile.client.train.ms": (t1 - t0) * 1000.0,
            }
        )
        return Message(
            content=RecordDict({"arrays": ArrayRecord(), "metrics": metrics}),
            reply_to=msg,
        )

    incoming_state: dict[str, torch.Tensor] | None = None
    incoming_state_loaded_from_layers = False
    if msg.content and "arrays" in msg.content:
        incoming_state = msg.content["arrays"].to_torch_state_dict()
    elif (
        aggregation_mode != "all_at_once"
        and model_preloaded
        and STATE_LAYER_PATHS in context.state
    ):
        incoming_state = load_state_dict_from_layer_files(context)
        incoming_state_loaded_from_layers = True
    elif aggregation_mode != "all_at_once" and model_preloaded:
        raise FileNotFoundError(
            "Layer-wise model was marked as preloaded, but no layer files were "
            "available to load for training."
        )

    if disable_train:
        # Keep communication path alive without invoking local training workloads.
        if aggregation_mode == "all_at_once":
            input_fingerprint = (
                state_dict_fingerprint(incoming_state)
                if incoming_state is not None
                else 0.0
            )
            noise_scale = float(context.run_config.get("train.debug-noise-scale", 0.0))
            noise_result = (
                _debug_add_noise_to_state_dict(incoming_state, noise_scale)
                if incoming_state is not None
                else None
            )
            output_fingerprint = (
                state_dict_fingerprint(incoming_state)
                if incoming_state is not None
                else 0.0
            )
            arrays_out = (
                ArrayRecord(incoming_state)
                if incoming_state is not None
                else ArrayRecord()
            )
            t1 = perf_counter()
            metrics_dict = {
                "train_loss": 0.0,
                "num-examples": 1,
                "train_skipped": 1,
                "profile.client.train.ms": (t1 - t0) * 1000.0,
                "model.input_fingerprint": input_fingerprint,
                "model.output_fingerprint": output_fingerprint,
                "model.fingerprint_delta": output_fingerprint - input_fingerprint,
                "debug.noise_scale": noise_scale,
                "debug.noise_applied": 1 if noise_result is not None else 0,
            }
            if noise_result is not None:
                _, before, after = noise_result
                metrics_dict["debug.noise_before"] = before
                metrics_dict["debug.noise_after"] = after
                metrics_dict["debug.noise_delta"] = after - before
            metrics = MetricRecord(metrics_dict)
            return Message(
                content=RecordDict({"arrays": arrays_out, "metrics": metrics}),
                reply_to=msg,
            )

        if incoming_state is not None:
            layer_names = list(incoming_state.keys())
            if msg.content and "config" in msg.content:
                config = msg.content["config"]
                if "layer_names" in config:
                    layer_names = list(config["layer_names"])
            _persist_layer_files(context, incoming_state, layer_names)
        elif STATE_LAYER_PATHS in context.state:
            if STATE_LAYER_NAMES not in context.state:
                layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
                layer_names = [
                    os.path.splitext(os.path.basename(path))[0] for path in layer_paths
                ]
                context.state[STATE_LAYER_NAMES] = ConfigRecord({"names": layer_names})
            context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": 0})
            context.state[STATE_NUM_EXAMPLES] = ConfigRecord({"num_examples": 1})

        t1 = perf_counter()
        metrics = MetricRecord(
            {
                "train_loss": 0.0,
                "num-examples": 1,
                "train_skipped": 1,
                "profile.client.train.ms": (t1 - t0) * 1000.0,
            }
        )
        return Message(
            content=RecordDict({"arrays": ArrayRecord(), "metrics": metrics}),
            reply_to=msg,
        )

    layerwise_dcp = (
        trainer_backend == "torchtitan"
        and aggregation_mode != "all_at_once"
        and torchtitan_dcp_enabled(context)
        and model_preloaded
        and STATE_LAYER_PATHS in context.state
    )
    if layerwise_dcp:
        layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
        input_fingerprint = state_dict_fingerprint_from_layer_paths(layer_paths)

        server_round = None
        if "server-round" in config:
            server_round = int(config["server-round"])
        elif "current-round" in config:
            server_round = int(config["current-round"])

        run_torchtitan_training(
            cfg,
            context,
            None,
            server_round=server_round,
            layer_paths=layer_paths,
            output_layer_dir=layer_dir(context),
        )
        output_fingerprint = state_dict_fingerprint_from_layer_paths(layer_paths)
        t1 = perf_counter()
        metrics_dict = {
            "train_loss": 0.0,
            "num-examples": 1,
            "profile.client.train.ms": (t1 - t0) * 1000.0,
            "model.input_fingerprint": input_fingerprint,
            "model.output_fingerprint": output_fingerprint,
            "model.fingerprint_delta": output_fingerprint - input_fingerprint,
        }
        metrics_dict.update(
            read_conversion_profile(
                os.path.join(
                    layer_dir(context), "torchtitan_conversion_profile.jsonl"
                )
            )
        )
        metrics = MetricRecord(metrics_dict)
        return Message(
            content=RecordDict({"arrays": ArrayRecord(), "metrics": metrics}),
            reply_to=msg,
        )

    # Load model
    model = get_model(cfg.model)
    if incoming_state is not None:
        model.load_state_dict(incoming_state, strict=True)
        if incoming_state_loaded_from_layers:
            _cleanup_layer_files_for_context(context)
    input_fingerprint = state_dict_fingerprint(model.state_dict())

    server_round = None
    if msg.content and "config" in msg.content:
        config = msg.content["config"]
        if "server-round" in config:
            server_round = int(config["server-round"])
        elif "current-round" in config:
            server_round = int(config["current-round"])

    if trainer_backend == "torchtitan":
        trained_state = run_torchtitan_training(
            cfg, context, model.state_dict(), server_round=server_round
        )
        model.load_state_dict(trained_state, strict=True)
    elif trainer_backend != "none":
        raise ValueError(f"Unsupported trainer.backend: {trainer_backend}")

    state_dict = model.state_dict()
    output_fingerprint = state_dict_fingerprint(state_dict)
    layer_names = list(state_dict.keys())
    if "layer_names" in config:
        layer_names = list(config["layer_names"])

    # Persist layers to disk for per-layer sending and track in context state.
    _persist_layer_files(context, state_dict, layer_names)

    t1 = perf_counter()
    metrics = {
        "train_loss": 0.0,
        "num-examples": 1,
        "profile.client.train.ms": (t1 - t0) * 1000.0,
        "model.input_fingerprint": input_fingerprint,
        "model.output_fingerprint": output_fingerprint,
        "model.fingerprint_delta": output_fingerprint - input_fingerprint,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": ArrayRecord(), "metrics": metric_record})

    if aggregation_mode == "all_at_once":
        content["arrays"] = ArrayRecord(state_dict)

    return Message(content=content, reply_to=msg)


@app.train("layer_wise_communication")
def train_comms(msg: Message, context: Context):
    """Send the model layer by layer from disk."""
    t0 = perf_counter()
    config = msg.content["config"] if msg.content and "config" in msg.content else {}
    chunk_ranges = parse_chunk_ranges(config)
    usechunk_keys = "chunk_starts" in config and "chunk_ends" in config
    layer_paths = (
        list(context.state[STATE_LAYER_PATHS]["paths"])
        if STATE_LAYER_PATHS in context.state
        else []
    )

    arrays: dict[str, torch.Tensor] = {}
    entries: list[tuple[int, str, int, int, bool]] = []
    if "upload_layer_idxs" in config:
        layer_idxs = [int(v) for v in list(config["upload_layer_idxs"])]
        layer_names = [str(v) for v in list(config["upload_layer_names"])]
        chunk_starts = [int(v) for v in list(config["upload_chunk_starts"])]
        chunk_ends = [int(v) for v in list(config["upload_chunk_ends"])]
        is_last_values = list(config["upload_is_last_chunk"])
        range_count = min(
            len(layer_idxs),
            len(layer_names),
            len(chunk_starts),
            len(chunk_ends),
            len(is_last_values),
        )
        for idx in range(range_count):
            entries.append(
                (
                    layer_idxs[idx],
                    layer_names[idx],
                    chunk_starts[idx],
                    chunk_ends[idx],
                    bool(is_last_values[idx]),
                )
            )
        usechunk_keys = True
    else:
        layer_idx = int(config.get("layer_idx", 0))
        if layer_idx >= len(layer_paths):
            layer_idx = len(layer_paths) - 1
        expected_layer_name = str(config.get("layer_name", ""))
        if not chunk_ranges:
            chunk_ranges = [(0, 0)]
        for start, end in chunk_ranges:
            entries.append(
                (layer_idx, expected_layer_name, start, end, is_last_batch(config))
            )

    if not entries:
        entries = [(0, "", 0, 0, True)]

    for layer_idx, expected_layer_name, start, end, is_last_chunk in entries:
        layer_path = ""
        if layer_paths:
            if layer_idx >= len(layer_paths):
                layer_idx = len(layer_paths) - 1
            layer_path = layer_paths[layer_idx]
        if not expected_layer_name and STATE_LAYER_NAMES in context.state:
            layer_names = list(context.state[STATE_LAYER_NAMES]["names"])
            if layer_idx < len(layer_names):
                expected_layer_name = str(layer_names[layer_idx])
        if not layer_path and expected_layer_name:
            layer_path = _layer_file_path(context, expected_layer_name)
        if not layer_path:
            continue
        if not os.path.exists(layer_path):
            raise FileNotFoundError(
                "Expected layer file for layer-wise upload was not found: "
                f"{layer_path}"
            )

        cache_key = context_path_key(context, layer_path)
        cached = _COMMS_LAYER_CACHE.get(cache_key)
        if (
            cached is None
            or (expected_layer_name and cached.layer_name != expected_layer_name)
        ):
            loaded = load_layer_from_disk(layer_path, expected_layer_name)
            if loaded is None:
                with open(layer_path, "rb") as file:
                    layer_dict = pickle.load(file)
                layer_name = next(iter(layer_dict.keys()))
                loaded = layer_dict[layer_name].detach().cpu()
            else:
                layer_name = expected_layer_name
                if not layer_name:
                    with open(layer_path, "rb") as file:
                        layer_dict = pickle.load(file)
                    layer_name = next(iter(layer_dict.keys()))
                    loaded = layer_dict[layer_name].detach().cpu()
            cached = CachedLayer(
                layer_name=layer_name,
                layer_path=layer_path,
                tensor=loaded,
            )
            _COMMS_LAYER_CACHE[cache_key] = cached

        tensor = cached.tensor
        if (
            end > start
            and hasattr(tensor, "__getitem__")
            and getattr(tensor, "ndim", 0) > 0
        ):
            chunk_tensor = tensor[start:end]
        else:
            chunk_tensor = tensor
        key_name = (
            chunk_key(cached.layer_name, start, end)
            if usechunk_keys
            else cached.layer_name
        )
        arrays[key_name] = chunk_tensor

        if is_last_chunk:
            _COMMS_LAYER_CACHE.pop(cache_key, None)

    final_layer_idx, _, _, _, final_is_last_chunk = entries[-1]
    send_complete = (
        bool(layer_paths)
        and final_layer_idx >= (len(layer_paths) - 1)
        and final_is_last_chunk
    )

    num_examples = (
        int(context.state[STATE_NUM_EXAMPLES]["num_examples"])
        if STATE_NUM_EXAMPLES in context.state
        else 1
    )
    metric_record = MetricRecord({"num-examples": num_examples})

    t1 = perf_counter()
    config_record = ConfigRecord({"send_complete": send_complete})
    content = RecordDict({
        "arrays": ArrayRecord(arrays),
        "metrics": metric_record,
        "config": config_record,
    })
    metric_record["profile.client.train_comms.ms"] = (t1 - t0) * 1000.0
    if send_complete:
        _cleanup_layer_files_for_context(context, layer_paths)

    return Message(content=content, reply_to=msg)
