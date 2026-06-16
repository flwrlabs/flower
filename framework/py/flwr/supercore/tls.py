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
"""TLS helpers for SuperExec/AppIO-style gRPC connections."""


import argparse
from pathlib import Path

from flwr.common.exit import ExitCode, flwr_exit

ServerCertificates = tuple[bytes, bytes, bytes]


def get_client_tls_args(
    insecure: bool,
    root_certificates_path: str | None,
) -> list[str]:
    """Return TLS flags for a Flower client process."""
    if insecure:
        return ["--insecure"]
    if root_certificates_path is None:
        return []
    return ["--root-certificates", root_certificates_path]


def try_obtain_optional_appio_server_certificates(
    args: argparse.Namespace,
) -> ServerCertificates | None:
    """Load AppIO server certificates from `appio_ssl_*` args when provided."""
    if (
        args.appio_ssl_certfile
        and args.appio_ssl_keyfile
        and args.appio_ssl_ca_certfile
    ):
        appio_ssl_ca_certfile = Path(args.appio_ssl_ca_certfile).expanduser()
        appio_ssl_certfile = Path(args.appio_ssl_certfile).expanduser()
        appio_ssl_keyfile = Path(args.appio_ssl_keyfile).expanduser()
        if not appio_ssl_ca_certfile.is_file():
            flwr_exit(
                ExitCode.COMMON_PATH_INVALID,
                "Path argument `--appio-ssl-ca-certfile` does not point to a file.",
            )
        if not appio_ssl_certfile.is_file():
            flwr_exit(
                ExitCode.COMMON_PATH_INVALID,
                "Path argument `--appio-ssl-certfile` does not point to a file.",
            )
        if not appio_ssl_keyfile.is_file():
            flwr_exit(
                ExitCode.COMMON_PATH_INVALID,
                "Path argument `--appio-ssl-keyfile` does not point to a file.",
            )
        return (
            appio_ssl_ca_certfile.read_bytes(),
            appio_ssl_certfile.read_bytes(),
            appio_ssl_keyfile.read_bytes(),
        )
    if args.appio_ssl_certfile or args.appio_ssl_keyfile or args.appio_ssl_ca_certfile:
        flwr_exit(
            ExitCode.COMMON_TLS_SERVER_CERTIFICATES_INVALID,
            "You need to provide valid file paths to `--appio-ssl-certfile`, "
            "`--appio-ssl-keyfile`, and `--appio-ssl-ca-certfile` to create a "
            "secure AppIO connection.",
        )
    return None


def validate_and_resolve_root_certificates(
    root_cert_path: str | None,
    insecure: bool,
) -> bytes | None:
    """Validate and return root certificate bytes for gRPC connections."""
    if insecure:
        if root_cert_path is not None:
            flwr_exit(
                ExitCode.COMMON_TLS_ROOT_CERTIFICATES_INCOMPATIBLE,
                "Conflicting options: The '--insecure' flag disables TLS, but "
                "'--root-certificates' was also specified.",
            )
        return None

    if root_cert_path is None:
        return None  # None in gRPC means the default system root certificates

    if not Path(root_cert_path).expanduser().is_file():
        flwr_exit(
            ExitCode.COMMON_PATH_INVALID,
            "Path argument `--root-certificates` does not point to a file.",
        )

    try:
        return Path(root_cert_path).expanduser().read_bytes()
    except OSError as e:
        flwr_exit(
            ExitCode.COMMON_PATH_INVALID,
            f"Failed to read root certificates from '{root_cert_path}': {e}",
        )
