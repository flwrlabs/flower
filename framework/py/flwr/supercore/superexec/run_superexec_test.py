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
"""Tests for SuperExec runtime dependency installation wiring."""


from unittest.mock import patch

from flwr.supercore.superexec.run_superexec import run_with_deprecation_warning


def test_run_with_deprecation_warning_logs_and_forwards_runtime_flags() -> None:
    """Ensure deprecation path includes runtime dependency install flags."""
    with (
        patch("flwr.supercore.superexec.run_superexec.log") as log,
        patch(
            "flwr.supercore.superexec.run_superexec.run_superexec"
        ) as run_superexec_fn,
    ):
        run_with_deprecation_warning(
            cmd="flwr-serverapp",
            plugin_type="serverapp",
            plugin_class=object,  # type: ignore[arg-type]
            stub_class=object,  # type: ignore[arg-type]
            appio_api_address="127.0.0.1:9091",
            parent_pid=321,
            warn_run_once=False,
            runtime_dependency_install=True,
        )

    assert (
        "flower-superexec --insecure --plugin-type serverapp "
        "--appio-api-address 127.0.0.1:9091 --parent-pid 321 "
        "--allow-runtime-dependency-installation"
    ) in [call.args[1] for call in log.call_args_list]
    run_superexec_fn.assert_called_once_with(
        plugin_class=object,
        stub_class=object,
        appio_api_address="127.0.0.1:9091",
        superexec_auth_secret=None,
        parent_pid=321,
        runtime_dependency_install=True,
    )
