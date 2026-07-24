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
from dataclasses import dataclass
from logging import WARN
from pathlib import Path
from typing import TypeVar, cast

import yaml

from flwr.common.constant import (
    AUTHN_TYPE_YAML_KEY,
    AUTHZ_TYPE_YAML_KEY,
    AuthnType,
    AuthzType,
    EventLogWriterType,
)
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.logger import log
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import (
    ControlAuthnPlugin,
    ControlAuthzPlugin,
    NoOpControlAuthnPlugin,
    NoOpControlAuthzPlugin,
)
from flwr.superlink.federation import FederationManager, NoOpFederationManager

P = TypeVar("P", ControlAuthnPlugin, ControlAuthzPlugin)

try:
    from flwr.ee import (
        get_control_authn_ee_plugins,
        get_control_authz_ee_plugins,
        get_ee_federation_manager,
        get_ee_linkstate_factory,
        get_ee_objectstore_factory,
    )
except ImportError:

    def get_control_authn_ee_plugins() -> dict[str, type[ControlAuthnPlugin]]:
        """Return all Control API authentication plugins for EE."""
        return {}

    def get_control_authz_ee_plugins() -> dict[str, type[ControlAuthzPlugin]]:
        """Return all Control API authorization plugins for EE."""
        return {}

    def get_ee_federation_manager() -> FederationManager:
        """Return the EE FederationManager."""
        raise NotImplementedError("No federation manager is currently supported.")

    def get_ee_objectstore_factory(database: str) -> ObjectStoreFactory:
        """Return an EE ObjectStoreFactory for supported non-SQLite database URLs."""
        raise NotImplementedError("No additional state backends are supported.")

    def get_ee_linkstate_factory(
        database: str,
        federation_manager: FederationManager,
        objectstore_factory: ObjectStoreFactory,
    ) -> LinkStateFactory:
        """Return an EE LinkStateFactory for supported non-SQLite database URLs."""
        raise NotImplementedError("No additional state backends are supported.")


@dataclass
class SuperLinkLifespanConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration needed to start the SuperLink lifespan."""

    serverappio_address: str
    control_address: str
    health_server_address: str | None
    enable_http_api: bool
    disable_grpc_api: bool
    host: str
    port: int
    insecure: bool
    certificates: tuple[bytes, bytes, bytes] | None
    appio_certificates: tuple[bytes, bytes, bytes] | None
    superexec_auth_secret: bytes | None
    authn_plugin: ControlAuthnPlugin
    authz_plugin: ControlAuthzPlugin
    event_log_plugin: EventLogWriterPlugin | None
    enable_event_log: bool
    artifact_provider: ArtifactProvider | None
    enable_supernode_auth: bool
    fleet_api_type: str
    fleet_api_address: str | None
    simulation: bool
    ssl_keyfile: str | None
    ssl_certfile: str | None
    database: str
    isolation: str
    appio_ssl_ca_certfile: str | None
    runtime_dependency_install: bool


def get_license_plugin() -> LicensePlugin | None:
    """Return the license plugin when Flower Enterprise is installed."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee import get_license_plugin as get_ee_license_plugin
    except ImportError:
        return None

    ret: LicensePlugin | None = get_ee_license_plugin()
    return ret


def load_control_event_log_plugin() -> EventLogWriterPlugin:
    """Load the configured Control API event log writer plugin."""
    try:
        # pylint: disable-next=import-outside-toplevel
        from flwr.ee import get_control_event_log_writer_plugins
    except ImportError:
        sys.exit("No event log writer plugins are currently supported.")

    try:
        plugins: dict[str, type[EventLogWriterPlugin]] = (
            get_control_event_log_writer_plugins()
        )
        return plugins[EventLogWriterType.STDOUT]()
    except KeyError:
        sys.exit("No event log writer plugin is provided.")
    except NotImplementedError:
        sys.exit("No event log writer plugins are currently supported.")


def get_federation_manager(is_simulation: bool = False) -> FederationManager:
    """Return the FederationManager."""
    try:
        federation_manager: FederationManager = get_ee_federation_manager()
        return federation_manager
    except NotImplementedError:
        return NoOpFederationManager(simulation=is_simulation)


def _is_non_sqlite_database_url(database: str) -> bool:
    """Return whether the database argument is a non-SQLite URL."""
    normalized = database.strip().lower()
    return "://" in normalized and not normalized.startswith("sqlite://")


def get_objectstore_linkstate_factories(
    database: str,
    federation_manager: FederationManager,
) -> tuple[ObjectStoreFactory, LinkStateFactory]:
    """Return ObjectStore and LinkState factories for the selected DB backend."""
    if _is_non_sqlite_database_url(database):
        try:
            objectstore_factory = get_ee_objectstore_factory(database)
            state_factory = get_ee_linkstate_factory(
                database, federation_manager, objectstore_factory
            )
            return objectstore_factory, state_factory
        except NotImplementedError as exc:
            raise ValueError(
                "Unsupported value for `--database`. The Flower framework supports "
                "`:flwr-in-memory:`, `:memory:`, SQLite file paths, and `sqlite://` "
                "URLs (including `sqlite:///:memory:`)."
            ) from exc

    objectstore_factory = ObjectStoreFactory(database)
    state_factory = LinkStateFactory(database, federation_manager, objectstore_factory)
    return objectstore_factory, state_factory


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
