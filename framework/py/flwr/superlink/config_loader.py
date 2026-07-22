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
"""Load SuperLink configuration and config-driven components."""


import sys
from collections.abc import Callable
from logging import WARN
from pathlib import Path
from typing import TypeVar, cast

import yaml

from flwr.common.constant import (
    AUTHN_TYPE_YAML_KEY,
    AUTHZ_TYPE_YAML_KEY,
    AuthnType,
    AuthzType,
)
from flwr.common.logger import log
from flwr.superlink.auth_plugin import (
    ControlAuthnPlugin,
    ControlAuthzPlugin,
    NoOpControlAuthnPlugin,
    NoOpControlAuthzPlugin,
)

P = TypeVar("P", ControlAuthnPlugin, ControlAuthzPlugin)

try:
    from flwr.ee import get_control_authn_ee_plugins, get_control_authz_ee_plugins
except ImportError:

    def get_control_authn_ee_plugins() -> dict[str, type[ControlAuthnPlugin]]:
        """Return all Control API authentication plugins for EE."""
        return {}

    def get_control_authz_ee_plugins() -> dict[str, type[ControlAuthzPlugin]]:
        """Return all Control API authorization plugins for EE."""
        return {}


def get_control_authn_plugins() -> dict[str, type[ControlAuthnPlugin]]:
    """Return all Control API authentication plugins."""
    ee_dict: dict[str, type[ControlAuthnPlugin]] = get_control_authn_ee_plugins()
    return ee_dict | {AuthnType.NOOP: NoOpControlAuthnPlugin}


def get_control_authz_plugins() -> dict[str, type[ControlAuthzPlugin]]:
    """Return all Control API authorization plugins."""
    ee_dict: dict[str, type[ControlAuthzPlugin]] = get_control_authz_ee_plugins()
    return ee_dict | {AuthzType.NOOP: NoOpControlAuthzPlugin}


def load_control_auth_plugins(
    config_path: str | None, verify_tls_cert: bool
) -> tuple[ControlAuthnPlugin, ControlAuthzPlugin]:
    """Obtain Control API authentication and authorization plugins."""
    # Load NoOp plugins if no config path is provided
    if config_path is None:
        config_path = ""
        config = {
            "authentication": {AUTHN_TYPE_YAML_KEY: AuthnType.NOOP},
            "authorization": {AUTHZ_TYPE_YAML_KEY: AuthzType.NOOP},
        }
    # Load YAML file
    else:
        with Path(config_path).expanduser().open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    def _load_plugin(
        section: str, yaml_key: str, loader: Callable[[], dict[str, type[P]]]
    ) -> P:
        section_cfg = config.get(section, {})
        auth_plugin_name = section_cfg.get(yaml_key, "")
        try:
            plugins: dict[str, type[P]] = loader()
            plugin_cls: type[P] = plugins[auth_plugin_name]
            return plugin_cls(Path(cast(str, config_path)), verify_tls_cert)
        except KeyError:
            if auth_plugin_name:
                sys.exit(
                    f"{yaml_key}: {auth_plugin_name} is not supported. "
                    f"Please provide a valid {section} type in the configuration."
                )
            sys.exit(f"No {section} type is provided in the configuration.")

    # Warn deprecated auth_type key
    if authn_type := config["authentication"].pop("auth_type", None):
        log(
            WARN,
            "The `auth_type` key in the authentication configuration is deprecated. "
            "Use `%s` instead.",
            AUTHN_TYPE_YAML_KEY,
        )
        config["authentication"][AUTHN_TYPE_YAML_KEY] = authn_type

    authn_plugin = _load_plugin(
        section="authentication",
        yaml_key=AUTHN_TYPE_YAML_KEY,
        loader=get_control_authn_plugins,
    )
    authz_plugin = _load_plugin(
        section="authorization",
        yaml_key=AUTHZ_TYPE_YAML_KEY,
        loader=get_control_authz_plugins,
    )

    return authn_plugin, authz_plugin
