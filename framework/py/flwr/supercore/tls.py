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


from pathlib import Path

from flwr.common.exit import ExitCode, flwr_exit


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
