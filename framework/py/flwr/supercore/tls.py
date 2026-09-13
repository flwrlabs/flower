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
"""TLS helpers for Runtime API connections."""


from pathlib import Path

from flwr.supercore.exit import ExitCode, flwr_exit


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


def validate_and_resolve_root_certificates(
    root_cert_path: str | None,
    insecure: bool,
) -> bytes | None:
    """Validate and return root certificate bytes for Runtime API clients."""
    if insecure:
        if root_cert_path is not None:
            flwr_exit(
                ExitCode.COMMON_TLS_ROOT_CERTIFICATES_INCOMPATIBLE,
                "Conflicting options: The '--insecure' flag disables TLS, but "
                "'--root-certificates' was also specified.",
            )
        return None

    if root_cert_path is None:
        return None  # Use the default system root certificates

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
