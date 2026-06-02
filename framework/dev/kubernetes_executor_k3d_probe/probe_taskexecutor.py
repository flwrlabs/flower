#!/usr/local/bin/python
# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dev-only KubernetesExecutor probe TaskExecutor command."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

APPIO_TOKEN_FILE_PATH = "/run/flwr/appio/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = "/run/flwr/appio/ca.crt"
SMOKE_TOKEN_PATTERN = re.compile(r"^smoke-token-[0-9a-f]{32}-[0-9]+$")
COMMAND_TO_ADDRESS_ARG = {
    "flwr-serverapp": "--serverappio-api-address",
    "flwr-clientapp": "--clientappio-api-address",
    "flwr-simulation": "--serverappio-api-address",
}


class ProbeFailure(RuntimeError):
    """Raised when the probe command surface is not rendered as expected."""


def main(argv: list[str] | None = None) -> int:
    """Validate the mounted Secret and rendered TaskExecutor argv."""
    if argv is None:
        argv = sys.argv

    try:
        _run_probe(argv)
    except ProbeFailure as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    print("probe passed: mounted credentials and rendered argv were valid")
    return 0


def _run_probe(argv: list[str]) -> None:
    """Run the probe checks."""
    command = os.path.basename(argv[0])
    args = argv[1:]
    address_arg = COMMAND_TO_ADDRESS_ARG.get(command)
    if address_arg is None:
        raise ProbeFailure(f"unsupported command {command!r}")

    appio_address = _required_arg_value(args, address_arg)
    if not appio_address.strip():
        raise ProbeFailure(f"{address_arg} value must not be empty")
    print(f"probe: verified {address_arg}")

    token_path = _required_arg_value(args, "--token-file")
    if token_path != APPIO_TOKEN_FILE_PATH:
        raise ProbeFailure(
            f"--token-file must point at {APPIO_TOKEN_FILE_PATH}, got {token_path!r}"
        )
    _verify_token_file(Path(token_path))
    print("probe: verified mounted token file")

    _verify_tls_args(args)


def _required_arg_value(args: list[str], flag: str) -> str:
    """Return the value following a required argv flag."""
    try:
        index = args.index(flag)
    except ValueError as exc:
        raise ProbeFailure(f"missing required argument {flag}") from exc
    value_index = index + 1
    if value_index >= len(args):
        raise ProbeFailure(f"missing value for {flag}")
    value = args[value_index]
    if value.startswith("--"):
        raise ProbeFailure(f"missing value for {flag}")
    return value


def _verify_token_file(token_path: Path) -> None:
    """Verify token file existence and smoke-token shape without printing it."""
    if not token_path.is_file():
        raise ProbeFailure(f"token file does not exist at {token_path}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not SMOKE_TOKEN_PATTERN.fullmatch(token):
        raise ProbeFailure("token file content did not match smoke token pattern")


def _verify_tls_args(args: list[str]) -> None:
    """Verify AppIo TLS mode arguments are internally consistent."""
    has_insecure = "--insecure" in args
    has_root_certificates = "--root-certificates" in args
    if has_insecure and has_root_certificates:
        raise ProbeFailure(
            "--insecure and --root-certificates must not both be rendered"
        )
    if has_insecure:
        print("probe: verified insecure AppIo flag")
        return
    if has_root_certificates:
        root_certificates_path = _required_arg_value(args, "--root-certificates")
        if root_certificates_path != APPIO_ROOT_CERTIFICATES_FILE_PATH:
            raise ProbeFailure(
                "--root-certificates must point at "
                f"{APPIO_ROOT_CERTIFICATES_FILE_PATH}, got {root_certificates_path!r}"
            )
        if not Path(root_certificates_path).is_file():
            raise ProbeFailure(
                f"root certificates file does not exist at {root_certificates_path}"
            )
        print("probe: verified root certificates file")
        return
    raise ProbeFailure("expected either --insecure or --root-certificates")


if __name__ == "__main__":
    sys.exit(main())
