# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower ClientApp loading utils."""


from collections.abc import Callable
from logging import DEBUG
from pathlib import Path

from flwr.clientapp.client_app import ClientApp, LoadClientAppError
from flwr.common.config import (
    get_metadata_from_config,
    get_project_config,
    get_project_dir,
)
from flwr.common.logger import log
from flwr.common.object_ref import load_app, validate
from flwr.supercore.utils import get_flwr_home


def get_load_client_app_fn(
    default_app_ref: str,
    app_path: str | None,
    multi_app: bool,
) -> Callable[[str, str, str], ClientApp]:
    """Get the load_client_app_fn function.

    If `multi_app` is True, this function loads the specified ClientApp
    based on `fab_id` and `fab_version`. If `fab_id` is empty, a default
    ClientApp will be loaded.

    If `multi_app` is False, it ignores `fab_id` and `fab_version` and
    loads a default ClientApp.
    """
    if not multi_app:
        log(
            DEBUG,
            "Flower SuperNode will load and validate ClientApp `%s`",
            default_app_ref,
        )

        valid, error_msg = validate(default_app_ref, project_dir=app_path)
        if not valid and error_msg:
            raise LoadClientAppError(error_msg) from None

    def _load(fab_id: str, fab_version: str, fab_hash: str) -> ClientApp:
        runtime_app_dir = Path(app_path if app_path else "").absolute()
        # If multi-app feature is disabled
        if not multi_app:
            # Set app reference
            client_app_ref = default_app_ref
        # If multi-app feature is enabled but app directory is provided.
        # `fab_hash` is not required since the app is loaded from `runtime_app_dir`.
        elif app_path is not None:
            config = get_project_config(runtime_app_dir)
            this_fab_id, this_fab_version = get_metadata_from_config(config)

            if this_fab_version != fab_version or this_fab_id != fab_id:
                raise LoadClientAppError(
                    f"FAB ID or version mismatch: Expected FAB ID '{this_fab_id}' and "
                    f"FAB version '{this_fab_version}', but received FAB ID '{fab_id}' "
                    f"and FAB version '{fab_version}'.",
                ) from None

            # log(WARN, "FAB ID is not provided; the default ClientApp will be loaded.")

            # Set app reference
            client_app_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]
        # If multi-app feature is enabled
        else:
            try:
                runtime_app_dir = get_project_dir(fab_id, fab_version, fab_hash)
                config = get_project_config(runtime_app_dir)
            except Exception as e:
                raise LoadClientAppError(
                    "Failed to load ClientApp."
                    "Possible reasons for error include mismatched "
                    "`fab_id`, `fab_version`, or `fab_hash` in "
                    f"{str(get_flwr_home().resolve())}."
                ) from e

            # Set app reference
            client_app_ref = config["tool"]["flwr"]["app"]["components"]["clientapp"]

        # Load ClientApp
        log(
            DEBUG,
            "Loading ClientApp `%s`",
            client_app_ref,
        )
        client_app = load_app(client_app_ref, LoadClientAppError, runtime_app_dir)

        if not isinstance(client_app, ClientApp):
            raise LoadClientAppError(
                f"Attribute {client_app_ref} is not of type {ClientApp}",
            ) from None

        return client_app

    return _load
