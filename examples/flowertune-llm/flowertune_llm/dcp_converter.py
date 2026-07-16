"""TorchTitan DCP conversion worker used by scheduler jobs."""

from __future__ import annotations

import argparse

from flowertune_llm.task import (
    convert_dcp_to_layer_directory,
    convert_layer_directory_to_dcp,
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
        convert_layer_directory_to_dcp(
            args.input_dir,
            args.output_dir,
            train_spec_name=args.train_spec,
            model_args_key=args.model_args,
            dcp_threads=args.threads,
        )
        return

    if not args.reference_dir:
        raise ValueError("--reference-dir is required for --direction to-layers")
    convert_dcp_to_layer_directory(
        args.input_dir,
        args.reference_dir,
        args.output_dir,
        train_spec_name=args.train_spec,
        model_args_key=args.model_args,
        dcp_threads=args.threads,
        ready_marker=args.ready_marker or None,
    )


if __name__ == "__main__":
    main()
