"""Configuration helper for TraceFL baseline.

This module handles:
1. Parsing Flower's run_config into TraceFL configuration
2. Detecting model architecture (CNN/ResNet/Transformer) from model name
3. Validating model-dataset compatibility (e.g., transformers for text)
4. Normalizing configuration values from CLI string overrides

The main entry point is create_tracefl_config() which returns a
SimpleNamespace object matching original TraceFL's configuration structure.
"""

import ast
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any


def _detect_model_architecture(model_name):
    """Detect the architecture type of a model based on its name.

    Parameters
    ----------
    model_name : str
        The name of the model

    Returns
    -------
    str
        The architecture type: 'cnn', 'resnet', 'densenet', 'transformer'
    """
    if model_name in [
        "squeezebert/squeezebert-uncased",
        "openai-community/openai-gpt",
        "Intel/dynamic_tinybert",
        "google-bert/bert-base-cased",
        "microsoft/MiniLM-L12-H384-uncased",
        "distilbert/distilbert-base-uncased",
    ]:
        return "transformer"
    if model_name.startswith("resnet"):
        return "resnet"
    if model_name.startswith("densenet"):
        return "densenet"
    if model_name == "cnn":
        return "cnn"
    return "unknown"


def _validate_model_dataset_compatibility(data_dist):
    """Validate that the model and dataset are compatible.

    Parameters
    ----------
    data_dist : SimpleNamespace
        Dataset configuration object

    Raises
    ------
    ValueError
        If the model and dataset are incompatible
    """
    dataset_arch = data_dist.architecture
    model_arch = data_dist.model_architecture

    # Check compatibility
    if dataset_arch == "text" and model_arch != "transformer":
        raise ValueError(
            f"Incompatible combination: Text dataset '{data_dist.dname}' "
            f"requires transformer model, but got '{data_dist.model_name}'"
        )

    if dataset_arch in ["vision", "medical"] and model_arch == "transformer":
        raise ValueError(
            f"Incompatible combination: Vision/Medical dataset '{data_dist.dname}' "
            f"requires CNN/ResNet/DenseNet model, but got transformer "
            f"'{data_dist.model_name}'"
        )

    # Log compatibility info
    print(
        f"✅ Model-Dataset Compatibility: {model_arch} model + {dataset_arch} dataset"
    )

    # Log channel adaptation info
    if dataset_arch == "medical" and model_arch in ["resnet", "densenet"]:
        print(
            f"🔧 Channel Adaptation: {data_dist.channels}-channel input "
            f"for {model_arch} model"
        )


def _normalize_run_config_value(value: Any) -> Any:
    """Return a normalized representation for values read from ``run_config``.

    Flower exposes CLI overrides through ``Context.run_config``. Depending on
    how the command was quoted these overrides might already be typed (e.g.,
    ``int``/``float``) or still be wrapped inside a quoted ``str``.  We try to
    coerce the string representation into the intended Python literal using
    :func:`ast.literal_eval`.  When parsing fails we simply strip matching outer
    quotes so commands such as ``tracefl.dataset='mnist'`` remain usable.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return stripped
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            if (stripped.startswith('"') and stripped.endswith('"')) or (
                stripped.startswith("'") and stripped.endswith("'")
            ):
                return stripped[1:-1]
            return stripped
    return value


def _as_int(value: Any, default: int) -> int:
    normalized = _normalize_run_config_value(value)
    if normalized is None:
        return int(default)
    try:
        return int(normalized)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    normalized = _normalize_run_config_value(value)
    if normalized is None:
        return float(default)
    try:
        return float(normalized)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    normalized = _normalize_run_config_value(value)
    if normalized is None:
        return bool(default)
    if isinstance(normalized, bool):
        return normalized
    if isinstance(normalized, str):
        lowered = normalized.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    try:
        return bool(int(normalized))
    except (TypeError, ValueError):
        return bool(default)


def _as_str(value: Any, default: str) -> str:
    normalized = _normalize_run_config_value(value)
    if normalized is None:
        return default
    return str(normalized)


def _as_int_list(value: Any, default: Sequence[int]) -> list[int]:
    normalized = _normalize_run_config_value(value)
    if normalized is None:
        return list(default)
    if isinstance(normalized, Iterable) and not isinstance(normalized, str | bytes):
        try:
            return [int(v) for v in normalized]
        except (TypeError, ValueError):
            pass
    if isinstance(normalized, str):
        if normalized == "":
            return []
        parts = [p.strip() for p in normalized.split(",") if p.strip()]
        try:
            return [int(p) for p in parts]
        except (TypeError, ValueError):
            return list(default)
    return list(default)


def _as_label_map(value: Any) -> dict[int, int]:
    normalized = _normalize_run_config_value(value)
    if normalized in (None, "", {}):
        return {}
    if isinstance(normalized, dict):
        result: dict[int, int] = {}
        for k, v in normalized.items():
            try:
                result[int(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def create_tracefl_config(context):
    """Create TraceFL configuration from Flower context.

    Parameters
    ----------
    context : Context
        Flower context containing run_config

    Returns
    -------
    SimpleNamespace
        TraceFL configuration object
    """
    # Configuration loaded successfully

    # Extract TraceFL config directly from run_config
    run_cfg = context.run_config

    data_dist = SimpleNamespace()
    data_dist.dname = _as_str(run_cfg.get("tracefl.dataset"), "cifar10")
    data_dist.dist_type = _as_str(
        run_cfg.get("tracefl.data-distribution"), "non_iid_dirichlet"
    )
    data_dist.num_clients = _as_int(run_cfg.get("tracefl.num-clients"), 3)
    data_dist.dirichlet_alpha = _as_float(run_cfg.get("tracefl.dirichlet-alpha"), 0.5)
    data_dist.max_per_client_data_size = _as_int(
        run_cfg.get("tracefl.max-per-client-data-size"), 1000
    )
    data_dist.max_server_data_size = _as_int(
        run_cfg.get("tracefl.max-server-data-size"), 500
    )
    data_dist.batch_size = _as_int(run_cfg.get("tracefl.batch-size"), 32)

    # Add model configuration
    model_name = _as_str(run_cfg.get("tracefl.model"), "cnn")
    data_dist.model_name = model_name

    # Set dataset-specific properties based on dataset name
    if data_dist.dname in ["cifar10", "cifar100"]:
        data_dist.num_classes = 10 if data_dist.dname == "cifar10" else 100
        data_dist.channels = 3
        data_dist.architecture = "vision"
    elif data_dist.dname in ["pathmnist", "organamnist"]:
        data_dist.num_classes = 9 if data_dist.dname == "pathmnist" else 11
        data_dist.channels = 1  # Grayscale medical images
        data_dist.architecture = "medical"
    elif data_dist.dname in ["dbpedia_14", "yahoo_answers_topics"]:
        data_dist.num_classes = 14 if data_dist.dname == "dbpedia_14" else 10
        data_dist.channels = 0  # Text data
        data_dist.architecture = "text"
    else:
        # Default values
        data_dist.num_classes = 10
        data_dist.channels = 3
        data_dist.architecture = "vision"

    # Auto-detect model architecture compatibility
    data_dist.model_architecture = _detect_model_architecture(model_name)

    # Validate model-dataset compatibility
    _validate_model_dataset_compatibility(data_dist)

    # Create main config
    cfg = SimpleNamespace()
    cfg.data_dist = data_dist
    cfg.device = _as_str(run_cfg.get("tracefl.device"), "cpu")
    cfg.enable_beta = _as_bool(run_cfg.get("tracefl.enable-beta"), True)

    # Parse provenance rounds from string to list
    cfg.provenance_rounds = _as_int_list(
        run_cfg.get("tracefl.provenance-rounds"), [1, 2, 3]
    )

    cfg.client_weights_normalization = _as_bool(
        run_cfg.get("tracefl.client-weights-normalization"), True
    )

    # Differential privacy parameters (defaults match original TraceFL)
    cfg.noise_multiplier = _as_float(run_cfg.get("tracefl.noise-multiplier"), -1.0)
    cfg.clipping_norm = _as_float(run_cfg.get("tracefl.clipping-norm"), -1.0)

    # Faulty client simulation parameters
    cfg.faulty_clients_ids = _as_int_list(run_cfg.get("tracefl.faulty-clients-ids"), [])
    cfg.label2flip = _as_label_map(run_cfg.get("tracefl.label2flip"))

    # Output directory for experiment results
    cfg.output_dir = _as_str(run_cfg.get("tracefl.output-dir"), "results")

    return cfg
