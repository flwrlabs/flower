"""flowertune-llm: A Flower / FlowerTune app."""

from __future__ import annotations

import os
import pickle
import warnings
from contextlib import nullcontext
from time import perf_counter
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.profiling import (
    PROFILE_CLIENT_NAME_KEY,
    PROFILE_CONFIG_RECORD_NAME,
)
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.compression import add_compression_metrics, compress_if_enabled
from flowertune_llm.benchmark import run_instruction_training
from flowertune_llm.models import get_model, is_adapter_parameter
from flowertune_llm.pretraining import run_continued_pretraining
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


def _profiled_content(context: Context, records: dict[str, Any]) -> RecordDict:
    """Attach the optional human-readable client name to a reply."""
    configured_name = context.node_config.get(
        "client.name", context.run_config.get("client.name", "")
    )
    client_name = str(configured_name).strip() or str(context.node_id)
    records[PROFILE_CONFIG_RECORD_NAME] = ConfigRecord(
        {PROFILE_CLIENT_NAME_KEY: client_name}
    )
    return RecordDict(records)


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
    missing = [
        path
        for path in layer_paths
        if not os.path.isfile(path) or os.path.getsize(path) == 0
    ]
    if missing and require_all:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
        raise FileNotFoundError(
            "Layer-wise model files are incomplete for "
            f"run_id={context.run_id} node_id={context.node_id}: "
            f"{len(missing)}/{len(layer_paths)} expected layer files are missing "
            f"or empty: {preview}{suffix}. The operation was stopped before "
            "incomplete weights could be used."
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


def _run_minimal_training_step(
    model: torch.nn.Module, context: Context
) -> tuple[float, float]:
    """Run a small number of real SGD steps on one model parameter."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and 0 < parameter.numel() <= 1_000_000
    ]
    if not candidates:
        raise ValueError("Minimal training requires a floating parameter <= 1M values")
    _, parameter = candidates[-1]
    parameter.requires_grad_(True)

    device = parameter.device
    seq_length = int(context.run_config.get("train.minimal-seq-length", 8))
    num_steps = int(context.run_config.get("train.minimal-steps", 1))
    if num_steps < 1:
        raise ValueError("train.minimal-steps must be at least 1")
    learning_rate = float(
        context.run_config.get("train.minimal-learning-rate", 1e-4)
    )
    generator = torch.Generator(device=device).manual_seed(
        2026 + int(context.node_id)
    )
    vocab_size = int(getattr(model.config, "vocab_size"))
    before = parameter.detach().float().clone()
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)
    model.train()
    total_loss = 0.0
    for _ in range(num_steps):
        input_ids = torch.randint(
            0,
            vocab_size,
            (1, seq_length),
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().float().item())
    delta_norm = float((parameter.detach().float() - before).norm().item())
    optimizer.zero_grad(set_to_none=True)
    return total_loss / num_steps, delta_norm


_TRAINING_TEXT = """
Federated learning coordinates model improvement across independent participants
without collecting their original data in one place. Each participant trains on
local examples and returns an update which the server aggregates. Compression can
reduce network cost, but the reconstructed update must preserve model behavior.

Machine learning systems should be evaluated with repeatable experiments, explicit
metrics, and careful accounting of both numerical error and downstream predictions.
Reliable distributed systems also need bounded memory use, fault isolation, and
clear evidence that every expected participant contributed successfully.

Large language models learn statistical relationships between tokens. During
causal language modeling, each token is predicted from the tokens which precede it.
Optimization adjusts model parameters to increase the likelihood of the observed
continuation while validation checks that useful behavior remains stable.
""".strip()


def _run_substantial_training(
    model: torch.nn.Module,
    context: Context,
    *,
    server_round: int,
) -> tuple[float, float, int]:
    """Train the final transformer blocks for multiple natural-text steps."""
    if not torch.cuda.is_available():
        raise RuntimeError("Substantial training requires a CUDA device")

    num_steps = int(context.run_config.get("train.substantial-steps", 25))
    seq_length = int(context.run_config.get("train.substantial-seq-length", 128))
    learning_rate = float(
        context.run_config.get("train.substantial-learning-rate", 1e-3)
    )
    num_layers = int(context.run_config.get("train.substantial-last-layers", 2))
    if num_steps < 1 or seq_length < 2 or num_layers < 1:
        raise ValueError("Substantial training steps, sequence length, and layers > 0")

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) < num_layers:
        raise ValueError("Model does not expose enough transformer layers")

    lock_path = str(
        context.run_config.get("train.substantial-gpu-lock", "/tmp/flwr-gpu.lock")
    )
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")  # pylint: disable=R1732
    try:
        try:
            import fcntl

            lock_context = nullcontext()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            lock_context = nullcontext()

        with lock_context:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for layer in layers[-num_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            final_norm = getattr(backbone, "norm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad_(True)

            device = torch.device("cuda")
            model.to(device)
            model.train()
            model.config.use_cache = False
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            trainable_count = sum(parameter.numel() for parameter in trainable)
            before = [parameter.detach().clone() for parameter in trainable]

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            token_ids = tokenizer.encode(
                (_TRAINING_TEXT + "\n\n") * (num_steps + 8),
                add_special_tokens=True,
            )
            required = seq_length + 1
            if len(token_ids) < required:
                repeats = (required // max(1, len(token_ids))) + 1
                token_ids = (token_ids * repeats)[:required]

            optimizer = torch.optim.SGD(
                trainable, lr=learning_rate, momentum=0.9
            )
            total_loss = 0.0
            span = max(1, len(token_ids) - seq_length)
            offset = (int(context.node_id) + server_round * 97) % span
            for step in range(num_steps):
                start = (offset + step * seq_length) % span
                batch_ids = token_ids[start : start + seq_length]
                if len(batch_ids) < seq_length:
                    batch_ids += token_ids[: seq_length - len(batch_ids)]
                input_ids = torch.tensor(
                    batch_ids, dtype=torch.long, device=device
                ).unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                loss = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.detach().float().item())

            squared_delta = 0.0
            for parameter, original in zip(trainable, before, strict=True):
                difference = parameter.detach().float() - original.float()
                squared_delta += float(torch.sum(difference * difference).item())
            delta_norm = squared_delta**0.5
            optimizer.zero_grad(set_to_none=True)
            del optimizer, before, trainable
            model.to("cpu")
            torch.cuda.empty_cache()
            return total_loss / num_steps, delta_norm, trainable_count
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        lock_file.close()


@app.train("layer_wise_download")
def train_download(msg: Message, context: Context):
    """Receive layer chunks from the server and persist to disk."""
    t0 = perf_counter()
    if msg.content is None or "arrays" not in msg.content or "config" not in msg.content:
        return Message(
            content=_profiled_content(context, {"metrics": MetricRecord()}),
            reply_to=msg,
        )

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
        return Message(
            content=_profiled_content(context, {"metrics": MetricRecord()}),
            reply_to=msg,
        )

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
    return Message(
        content=_profiled_content(context, {"metrics": metrics}),
        reply_to=msg,
    )


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
            content=_profiled_content(
                context, {"arrays": ArrayRecord(), "metrics": metrics}
            ),
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
                content=_profiled_content(
                    context, {"arrays": arrays_out, "metrics": metrics}
                ),
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
            content=_profiled_content(
                context, {"arrays": ArrayRecord(), "metrics": metrics}
            ),
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
        if "layer_names" in config:
            _restore_layer_state_from_names(
                context,
                [str(layer_name) for layer_name in list(config["layer_names"])],
                require_all=True,
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
            content=_profiled_content(
                context, {"arrays": ArrayRecord(), "metrics": metrics}
            ),
            reply_to=msg,
        )

    # Load model
    model = get_model(cfg.model)
    if incoming_state is not None:
        adapter_only = trainer_backend in {
            "benchmark-lora",
            "benchmark-qlora",
            "benchmark-dora",
            "benchmark-qdora",
            "benchmark-rslora",
            "benchmark-qffa",
            "pretraining-lora",
            "pretraining-qlora",
            "pretraining-dora",
            "pretraining-qdora",
            "pretraining-rslora",
            "pretraining-ffa",
            "pretraining-qffa",
        }
        model.load_state_dict(incoming_state, strict=not adapter_only)
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

    minimal_step: tuple[float, float] | None = None
    substantial_step: tuple[float, float, int] | None = None
    benchmark_step: tuple[float, float, int, int, int, int] | None = None
    pretraining_step: tuple[float, int, int, int, int, int] | None = None
    if trainer_backend == "torchtitan":
        trained_state = run_torchtitan_training(
            cfg, context, model.state_dict(), server_round=server_round
        )
        model.load_state_dict(trained_state, strict=True)
    elif trainer_backend == "minimal":
        minimal_step = _run_minimal_training_step(model, context)
    elif trainer_backend == "substantial":
        substantial_step = _run_substantial_training(
            model, context, server_round=int(server_round or 0)
        )
    elif trainer_backend in {
        "benchmark-dense",
        "benchmark-lora",
        "benchmark-qlora",
        "benchmark-dora",
        "benchmark-qdora",
        "benchmark-rslora",
        "benchmark-qffa",
    }:
        benchmark_step = run_instruction_training(
            model,
            context,
            server_round=int(server_round or 0),
            method=trainer_backend.removeprefix("benchmark-"),
        )
    elif trainer_backend in {
        "pretraining-dense",
        "pretraining-lora",
        "pretraining-relora",
        "pretraining-qlora",
        "pretraining-dora",
        "pretraining-qdora",
        "pretraining-rslora",
        "pretraining-galore",
        "pretraining-ffa",
        "pretraining-qffa",
    }:
        method = trainer_backend.removeprefix("pretraining-")
        pretraining_step = run_continued_pretraining(
            model,
            context,
            server_round=int(server_round or 0),
            method=method,
        )
    elif trainer_backend != "none":
        raise ValueError(f"Unsupported trainer.backend: {trainer_backend}")

    trained_state_dict = model.state_dict()
    output_fingerprint = state_dict_fingerprint(trained_state_dict)
    updates_are_deltas = str(
        context.run_config.get("aggregation.updates", "weights")
    ).lower() == "delta"
    if updates_are_deltas:
        if incoming_state is None:
            raise ValueError("Delta updates require the downloaded base model state")
        names_to_send = (
            incoming_state.keys()
            if any(is_adapter_parameter(name) for name in incoming_state)
            else trained_state_dict.keys()
        )
        state_dict = {
            name: tensor.detach().cpu()
            - incoming_state[name].detach().cpu().to(dtype=tensor.dtype)
            for name in names_to_send
            for tensor in (trained_state_dict[name],)
        }
    else:
        state_dict = trained_state_dict
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
        "train.sent_delta": 1 if updates_are_deltas else 0,
    }
    if minimal_step is not None:
        loss, delta_norm = minimal_step
        metrics["train_loss"] = loss
        metrics["train.minimal_steps"] = int(
            context.run_config.get("train.minimal-steps", 1)
        )
        metrics["train.minimal_delta_norm"] = delta_norm
    if substantial_step is not None:
        loss, delta_norm, trainable_count = substantial_step
        metrics["train_loss"] = loss
        metrics["train.substantial_steps"] = int(
            context.run_config.get("train.substantial-steps", 25)
        )
        metrics["train.substantial_delta_norm"] = delta_norm
        metrics["train.trainable_parameters"] = trainable_count
    if benchmark_step is not None:
        (
            loss,
            delta_norm,
            trainable_count,
            supervised_tokens,
            peak_allocated,
            peak_reserved,
        ) = benchmark_step
        metrics["train_loss"] = loss
        metrics["train.benchmark_steps"] = int(
            context.run_config.get("train.benchmark.steps", 25)
        )
        if delta_norm >= 0:
            metrics["train.benchmark_delta_norm"] = delta_norm
        metrics["train.trainable_parameters"] = trainable_count
        metrics["train.supervised_tokens"] = supervised_tokens
        metrics["train.peak_cuda_allocated_bytes"] = peak_allocated
        metrics["train.peak_cuda_reserved_bytes"] = peak_reserved
    if pretraining_step is not None:
        (
            loss,
            trainable_count,
            trained_tokens,
            merged_modules,
            peak_allocated,
            peak_reserved,
        ) = pretraining_step
        metrics["train_loss"] = loss
        metrics["train.pretraining_steps"] = int(
            context.run_config.get("train.pretraining.steps", 50)
        )
        metrics["train.trainable_parameters"] = trainable_count
        metrics["train.pretraining_tokens"] = trained_tokens
        metrics["train.relora_merged_modules"] = merged_modules
        metrics["train.peak_cuda_allocated_bytes"] = peak_allocated
        metrics["train.peak_cuda_reserved_bytes"] = peak_reserved

    metric_record = MetricRecord(metrics)
    content = _profiled_content(
        context, {"arrays": ArrayRecord(), "metrics": metric_record}
    )

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
        if expected_layer_name:
            # The server-provided name is authoritative. Deriving the path from
            # it prevents stale or sparse context indices from selecting a
            # different parameter file.
            layer_path = _layer_file_path(context, expected_layer_name)
        elif layer_paths:
            if layer_idx >= len(layer_paths):
                layer_idx = len(layer_paths) - 1
            layer_path = layer_paths[layer_idx]
        if not expected_layer_name and STATE_LAYER_NAMES in context.state:
            layer_names = list(context.state[STATE_LAYER_NAMES]["names"])
            if layer_idx < len(layer_names):
                expected_layer_name = str(layer_names[layer_idx])
        if not layer_path:
            continue
        if not os.path.exists(layer_path):
            raise FileNotFoundError(
                "Layer-wise upload cannot continue because an expected layer "
                f"file is missing: run_id={context.run_id} "
                f"node_id={context.node_id} layer_idx={layer_idx} "
                f"layer_name={expected_layer_name!r} path={layer_path}. "
                "The preceding download or DCP conversion was incomplete."
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
    array_record, compression_stats, compression_ms = compress_if_enabled(
        ArrayRecord(arrays), config
    )
    metric_record = MetricRecord({"num-examples": num_examples})

    t1 = perf_counter()
    config_record = ConfigRecord({"send_complete": send_complete})
    content = RecordDict(
        {
            "arrays": array_record,
            "metrics": metric_record,
            "config": config_record,
        }
    )
    metric_record["profile.client.train_comms.ms"] = (t1 - t0) * 1000.0
    add_compression_metrics(
        metric_record,
        prefix="profile.client.upload_compression",
        stats=compression_stats,
        elapsed_ms=compression_ms,
    )
    if send_complete:
        _cleanup_layer_files_for_context(context, layer_paths)

    return Message(content=content, reply_to=msg)
