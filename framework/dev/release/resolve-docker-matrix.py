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


def _rewrite_matrix(
    matrix: dict[str, Any],
    docker_image_namespace: str,
    copy_path: str,
    tag: str | None,
    strip_flwr_version_ref: bool,
    build_local_wheel: bool,
) -> dict[str, Any]:
    for item in matrix["base"]["images"]:
        item["namespace_repository"] = f"{docker_image_namespace}/base"
        if tag is not None:
            item["tags_encoded"] = tag
        build_args = _lines(item.get("build_args_encoded", ""))
        if strip_flwr_version_ref:
            build_args = [arg for arg in build_args if not arg.startswith("FLWR_VERSION_REF=")]
        if build_local_wheel:
            build_args.extend([f"COPY_PATH={copy_path}", "FLWR_WHEEL=__FLWR_WHEEL__"])
        item["build_args_encoded"] = "\n".join(build_args)

    for item in matrix["binary"]["images"]:
        repository = item["namespace_repository"].split("/")[-1]
        item["namespace_repository"] = f"{docker_image_namespace}/{repository}"
        if tag is not None:
            item["tags_encoded"] = tag
            item["base_image"] = tag

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker-image-namespace", required=True)
    parser.add_argument("--copy-path", default="framework/dist")
    parser.add_argument("--tag")
    parser.add_argument("--strip-flwr-version-ref", action="store_true")
    parser.add_argument("--build-local-wheel", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.input.read_text())
    matrix = _rewrite_matrix(
        matrix=matrix,
        docker_image_namespace=args.docker_image_namespace,
        copy_path=args.copy_path,
        tag=args.tag,
        strip_flwr_version_ref=args.strip_flwr_version_ref,
        build_local_wheel=args.build_local_wheel,
    )
    args.output.write_text(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
