"""TorchTitan DCP conversion worker used by scheduler jobs."""

from __future__ import annotations

import argparse
import os

from flowertune_llm.task import (
    convert_dcp_to_state_file,
    convert_dcp_to_layer_directory,
    convert_layer_directory_to_dcp,
    convert_state_file_to_dcp,
    run_profiled_conversion,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=("to-dcp", "to-layers", "state-to-dcp", "dcp-to-state"),
        required=True,
    )
    parser.add_argument("--input-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--input-state", default="")
    parser.add_argument("--output-state", default="")
    parser.add_argument("--reference-state", default="")
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--ready-marker", default="")
    parser.add_argument("--train-spec", default="llama3")
    parser.add_argument("--model-args", default="auto")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    profile_path = os.environ.get("FLWR_TORCHTITAN_CONVERSION_PROFILE", "").strip()
    if args.direction == "to-dcp":
        if not args.input_dir or not args.output_dir:
            raise ValueError("--input-dir and --output-dir are required for to-dcp")
        run_profiled_conversion(
            profile_path,
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

    if args.direction == "state-to-dcp":
        if not args.input_state or not args.output_dir:
            raise ValueError(
                "--input-state and --output-dir are required for state-to-dcp"
            )
        run_profiled_conversion(
            profile_path,
            "to_dcp",
            lambda: convert_state_file_to_dcp(
                args.input_state,
                args.output_dir,
                train_spec_name=args.train_spec,
                model_args_key=args.model_args,
                dcp_threads=args.threads,
            ),
        )
        return

    if args.direction == "dcp-to-state":
        if not args.input_dir or not args.reference_state or not args.output_state:
            raise ValueError(
                "--input-dir, --reference-state, and --output-state are required "
                "for dcp-to-state"
            )
        run_profiled_conversion(
            profile_path,
            "to_state",
            lambda: convert_dcp_to_state_file(
                args.input_dir,
                args.reference_state,
                args.output_state,
                train_spec_name=args.train_spec,
                model_args_key=args.model_args,
            ),
        )
        return

    if not args.input_dir or not args.output_dir or not args.reference_dir:
        raise ValueError("--reference-dir is required for --direction to-layers")
    run_profiled_conversion(
        profile_path,
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
