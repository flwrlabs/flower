"""TorchTitan DCP conversion worker used by scheduler jobs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from typing import Any

from flowertune_llm.task import (
    convert_dcp_to_layer_directory,
    convert_layer_directory_to_dcp,
)


def _max_rss_mb() -> float:
    """Return this converter process's peak resident memory in MiB."""
    max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return max_rss / (1024**2)
    return max_rss / 1024.0


def _append_profile_event(event: dict[str, Any]) -> None:
    """Append a conversion phase event when profiling is configured."""
    profile_path = os.environ.get("FLWR_TORCHTITAN_CONVERSION_PROFILE", "").strip()
    if not profile_path:
        return
    os.makedirs(os.path.dirname(profile_path) or ".", exist_ok=True)
    with open(profile_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True) + "\n")


def _run_profiled_phase(phase: str, function: Callable[[], None]) -> None:
    """Run one conversion phase and persist duration/peak-memory telemetry."""
    started_at = time.time() * 1000.0
    started = time.perf_counter()
    _append_profile_event(
        {
            "event": "start",
            "phase": phase,
            "timestamp_ms": started_at,
        }
    )
    try:
        function()
    except Exception as error:
        _append_profile_event(
            {
                "event": "end",
                "phase": phase,
                "timestamp_ms": time.time() * 1000.0,
                "duration_ms": (time.perf_counter() - started) * 1000.0,
                "max_rss_mb": _max_rss_mb(),
                "success": False,
                "error": str(error),
            }
        )
        raise
    _append_profile_event(
        {
            "event": "end",
            "phase": phase,
            "timestamp_ms": time.time() * 1000.0,
            "duration_ms": (time.perf_counter() - started) * 1000.0,
            "max_rss_mb": _max_rss_mb(),
            "success": True,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=("to-dcp", "to-layers"), required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--ready-marker", default="")
    parser.add_argument("--train-spec", default="llama3")
    parser.add_argument("--model-args", default="auto")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.direction == "to-dcp":
        _run_profiled_phase(
            "to_dcp",
            lambda: convert_layer_directory_to_dcp(
                args.input_dir,
                args.output_dir,
                train_spec_name=args.train_spec,
                model_args_key=args.model_args,
                dcp_threads=args.threads,
            ),
        )
        return

    if not args.reference_dir:
        raise ValueError("--reference-dir is required for --direction to-layers")
    _run_profiled_phase(
        "to_layers",
        lambda: convert_dcp_to_layer_directory(
            args.input_dir,
            args.reference_dir,
            args.output_dir,
            train_spec_name=args.train_spec,
            model_args_key=args.model_args,
            dcp_threads=args.threads,
            ready_marker=args.ready_marker or None,
        ),
    )


if __name__ == "__main__":
    main()
