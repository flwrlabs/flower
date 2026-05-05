#!/usr/bin/env python3

# Copyright 2026 Flower Labs GmbH. All Rights Reserved.

"""Resolve Docker matrices for parameterized framework image publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker-image-namespace", required=True)
    parser.add_argument("--copy-path", default="framework/dist")
    parser.add_argument("--tag")
    parser.add_argument("--strip-flwr-version-ref", action="store_true")
    args = parser.parse_args()

    matrix: dict[str, Any] = json.loads(args.input.read_text())
    for item in matrix["base"]["images"]:
        item["namespace_repository"] = f"{args.docker_image_namespace}/base"
        if args.tag is not None:
            item["tags_encoded"] = args.tag
        build_args = _lines(item.get("build_args_encoded", ""))
        if args.strip_flwr_version_ref:
            build_args = [arg for arg in build_args if not arg.startswith("FLWR_VERSION_REF=")]
        build_args.extend([f"COPY_PATH={args.copy_path}", "FLWR_WHEEL=__FLWR_WHEEL__"])
        item["build_args_encoded"] = "\n".join(build_args)

    for item in matrix["binary"]["images"]:
        repository = item["namespace_repository"].split("/")[-1]
        item["namespace_repository"] = f"{args.docker_image_namespace}/{repository}"
        if args.tag is not None:
            item["tags_encoded"] = args.tag
            item["base_image"] = args.tag

    args.output.write_text(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
