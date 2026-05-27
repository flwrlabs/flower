# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""Node configuration loading from YAML / TOML files.

Supported YAML schema
---------------------

.. code-block:: yaml

    context: classification
    address: 0.0.0.0
    port: 9100
    tcp: true
    udp: false
    bootnodes:
      - "127.0.0.1:9001"

    topology:
      mode: dynamic           # "dynamic" | "static"
      node_name: node_0       # required when mode=static

      # Option A – path to an existing topology YAML file
      file: /path/to/topology.yaml

      # Option B – generate the file automatically (only when file is absent)
      generate:
        node_count: 5
        kind: ring            # random | star | ring | fullconnected
        output_path: /tmp/generated_topology.yaml
        # only when kind=random:
        random:
          mode: exact         # "exact" | "range"
          send_to: 3
          receive_from: 2

    sampling:
      algorithm: gbps         # gbps | brahams | basalt
      config_file: /tmp/config_sampling.json
      params:
        view_size: 10
        delay: 5
        # gbps only
        heal: 2
        swap: 3
        selection_policy: rand        # rand | old
        propagation_policy: pushpull  # push | pushpull
        age: 1
        # brahams only
        sampler_size: 5
        alpha: 0.45
        beta: 0.45
        # basalt only
        refresh: 3

    network:
      idle_connection_timeout_secs: 60
      max_negotiating_inbound_streams: 25
      per_connection_event_buffer_size: 7
      dial_concurrency_factor: 8
      enable_mdns: true
      enable_kad: true
      yamux:
        max_buffer_size: 1048576
        receive_window: 262144
        max_num_streams: 8192
      request_response:
        request_timeout_secs: 30
        max_concurrent_streams: 100
        max_requests_size_bytes: 10485760
        max_responses_size_bytes: 31457280
      kademlia:
        query_timeout_secs: 60
        kbucket_size: 20
      mdns:
        query_interval_secs: 300
        ttl_secs: 360
        enable_ipv6: false
        idle_discovery_timeout_secs: -1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from flwr.decentralized.common.graph import (
    generate_deploy_topology_yaml,
    topology_mode_dynamic,
    topology_mode_static,
)
from flwr.decentralized.common.network import (
    KademliaSettings,
    MdnsSettings,
    NetworkSettings,
    ResquestResponseSettings,
    YamuxSettings,
)
from flwr.decentralized.common.runtime_node import RuntimeNode
from flwr.decentralized.common.sampling import (
    Basalt,
    Brahams,
    Configuration,
    Gbps,
    PropagationPolicy,
    SelectionPolicy,
)
from flwr.decentralized.common.typing import DEFAULT_IP_ADDRESS, DEFAULT_PORT

# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_sampling(sampling_cfg: Dict[str, Any]) -> Configuration:
    """Build a :class:`~flwr.decentralized.common.sampling.Configuration`
    from the ``sampling`` section of a config file.

    Parameters
    ----------
    sampling_cfg : dict
        The ``sampling`` dict extracted from the config file.

    Returns
    -------
    Configuration
        A fully configured :class:`Configuration` instance.

    Raises
    ------
    ValueError
        If the algorithm name is unknown or required params are missing.
    """
    algo = sampling_cfg.get("algorithm", "").lower()
    params: Dict[str, Any] = sampling_cfg.get("params", {})
    config_file: Optional[str] = sampling_cfg.get("config_file", None)

    if algo == "gbps":
        sampling_obj = Gbps(
            view_size=params["view_size"],
            heal=params["heal"],
            swap=params["swap"],
            selection_policy=SelectionPolicy(params.get("selection_policy", "rand")),
            propagation_policy=PropagationPolicy(
                params.get("propagation_policy", "pushpull")
            ),
            delay=params["delay"],
            age=params.get("age", 1),
        )
    elif algo == "brahams":
        sampling_obj = Brahams(
            view_size=params["view_size"],
            sampler_size=params["sampler_size"],
            alpha=params["alpha"],
            beta=params["beta"],
            delay=params["delay"],
        )
    elif algo == "basalt":
        sampling_obj = Basalt(
            view_size=params["view_size"],
            refresh=params["refresh"],
            delay=params["delay"],
        )
    else:
        raise ValueError(
            f"Unknown sampling algorithm '{algo}'. "
            "Expected one of: 'gbps', 'brahams', 'basalt'."
        )

    return Configuration(config=sampling_obj, config_file=config_file)


def _build_network_settings(net_cfg: Dict[str, Any]) -> NetworkSettings:
    """Build a :class:`~flwr.decentralized.common.network.NetworkSettings`
    from the ``network`` section of a config file.

    Parameters
    ----------
    net_cfg : dict
        The ``network`` dict extracted from the config file.

    Returns
    -------
    NetworkSettings
    """
    yamux_cfg = net_cfg.get("yamux", {})
    rr_cfg = net_cfg.get("request_response", {})
    kad_cfg = net_cfg.get("kademlia", {})
    mdns_cfg = net_cfg.get("mdns", {})

    return NetworkSettings(
        idle_connection_timeout_secs=net_cfg.get("idle_connection_timeout_secs", 60),
        max_negotiating_inbound_streams=net_cfg.get(
            "max_negotiating_inbound_streams", 25
        ),
        per_connection_event_buffer_size=net_cfg.get(
            "per_connection_event_buffer_size", 7
        ),
        dial_concurrency_factor=net_cfg.get("dial_concurrency_factor", 8),
        enable_mdns=net_cfg.get("enable_mdns", True),
        enable_kad=net_cfg.get("enable_kad", True),
        yamux=YamuxSettings(
            max_buffer_size=yamux_cfg.get("max_buffer_size", 1024 * 1024),
            receive_window=yamux_cfg.get("receive_window", 256 * 1024),
            max_num_streams=yamux_cfg.get("max_num_streams", 8192),
        ),
        request_response=ResquestResponseSettings(
            request_timeout_secs=rr_cfg.get("request_timeout_secs", 30),
            max_concurrent_streams=rr_cfg.get("max_concurrent_streams", 100),
            max_requests_size_bytes=rr_cfg.get(
                "max_requests_size_bytes", 10 * 1024 * 1024
            ),
            max_responses_size_bytes=rr_cfg.get(
                "max_responses_size_bytes", 30 * 1024 * 1024
            ),
        ),
        kademlia=KademliaSettings(
            query_timeout_secs=kad_cfg.get("query_timeout_secs", 60),
            kbucket_size=kad_cfg.get("kbucket_size", 20),
        ),
        mdns=MdnsSettings(
            query_interval_secs=mdns_cfg.get("query_interval_secs", 300),
            ttl_secs=mdns_cfg.get("ttl_secs", 360),
            enable_ipv6=mdns_cfg.get("enable_ipv6", False),
            idle_discovery_timeout_secs=mdns_cfg.get("idle_discovery_timeout_secs", -1),
        ),
    )


def _resolve_topology(
    topology_cfg: Dict[str, Any],
    node_name: Optional[str] = None,
) -> Any:
    """Resolve the topology mode from the ``topology`` section.

    For ``mode=static``, if ``file`` is provided it is used directly. If
    ``generate`` is provided instead, the YAML file is generated first.

    Parameters
    ----------
    topology_cfg : dict
        The ``topology`` dict extracted from the config file.
    node_name : str | None
        Node name override from CLI. Takes precedence over the value in
        the config file.

    Returns
    -------
    Any
        A ``TopologyMode`` object.

    Raises
    ------
    ValueError
        If mode is ``static`` but neither ``file`` nor ``generate`` is
        provided, or if ``node_name`` is missing for static mode.
    """
    mode = topology_cfg.get("mode", "dynamic")
    resolved_node_name: Optional[str] = node_name or topology_cfg.get("node_name")

    if mode == "dynamic":
        return topology_mode_dynamic()

    # --- static ---
    if resolved_node_name is None:
        raise ValueError(
            "topology.node_name is required when topology.mode is 'static'."
        )

    topology_file: Optional[str] = topology_cfg.get("file")

    if topology_file is None:
        generate_cfg = topology_cfg.get("generate")
        if generate_cfg is None:
            raise ValueError(
                "topology.file or topology.generate must be provided when "
                "topology.mode is 'static'."
            )
        output_path = generate_cfg["output_path"]
        generate_deploy_topology_yaml(
            node_count=generate_cfg["node_count"],
            kind=generate_cfg["kind"],
            output_path=output_path,
            random=generate_cfg.get("random"),
        )
        topology_file = output_path

    return topology_mode_static(
        config_path=topology_file,
        node_name=resolved_node_name,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_node_config_yaml(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> RuntimeNode:
    """Load a node configuration from a YAML file and return a
    :class:`~flwr.decentralized.common.runtime_node.RuntimeNode`.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.
    overrides : dict | None
        Flat key-value pairs from the CLI that override values from the
        file.  Supported keys: ``context``, ``address``, ``port``,
        ``topology_mode``, ``topology_file``, ``node_name``, ``tcp``,
        ``udp``, ``bootnodes``.

    Returns
    -------
    RuntimeNode

    Raises
    ------
    ImportError
        If PyYAML (``pyyaml``) is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML config files. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}

    return _build_runtime_node(data, overrides or {})


def load_node_config_toml(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> RuntimeNode:
    """Load a node configuration from a TOML file and return a
    :class:`~flwr.decentralized.common.runtime_node.RuntimeNode`.

    Parameters
    ----------
    path : str | Path
        Path to the TOML configuration file.
    overrides : dict | None
        Flat key-value pairs from the CLI that override values from the
        file. Same supported keys as :func:`load_node_config_yaml`.

    Returns
    -------
    RuntimeNode

    Raises
    ------
    ImportError
        If ``tomllib`` / ``tomli`` is not available (Python < 3.11).
    FileNotFoundError
        If the file does not exist.
    """
    try:
        import tomllib  # type: ignore[import-untyped]  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import-untyped,no-redef]
        except ImportError as exc:
            raise ImportError(
                "tomli is required to load TOML config files on Python < 3.11. "
                "Install it with: pip install tomli"
            ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as fh:
        data: Dict[str, Any] = tomllib.load(fh)

    return _build_runtime_node(data, overrides or {})


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def _build_runtime_node(
    data: Dict[str, Any],
    overrides: Dict[str, Any],
) -> RuntimeNode:
    """Construct a :class:`RuntimeNode` from a raw config dict plus CLI overrides.

    Parameters
    ----------
    data : dict
        Parsed YAML / TOML content.
    overrides : dict
        CLI values that take precedence over ``data``.

    Returns
    -------
    RuntimeNode
    """

    def _get(key: str, default: Any = None) -> Any:
        """Override > file > default."""
        return (
            overrides.get(key)
            if overrides.get(key) is not None
            else data.get(key, default)
        )

    context: str = _get("context", "")
    address: str = _get("address", DEFAULT_IP_ADDRESS)
    port: int = int(_get("port", DEFAULT_PORT))
    tcp: bool = bool(_get("tcp", True))
    udp: bool = bool(_get("udp", False))
    bootnodes = _get("bootnodes", None)

    # --- topology ---
    topology_cfg: Dict[str, Any] = dict(data.get("topology", {}))

    # CLI overrides for topology
    cli_topo_mode = overrides.get("topology_mode")
    cli_topo_file = overrides.get("topology_file")
    cli_node_name = overrides.get("node_name")

    if cli_topo_mode:
        topology_cfg["mode"] = cli_topo_mode
    if cli_topo_file:
        topology_cfg["file"] = str(cli_topo_file)

    topology_mode_obj = _resolve_topology(topology_cfg, node_name=cli_node_name)

    # --- sampling ---
    sampling_cfg = data.get("sampling")
    sampling_conf: Optional[Configuration] = (
        _build_sampling(sampling_cfg) if sampling_cfg else None
    )

    # --- network ---
    network_cfg = data.get("network")
    network_settings: Optional[NetworkSettings] = (
        _build_network_settings(network_cfg) if network_cfg else None
    )

    return RuntimeNode(
        context=context,
        address=address,
        port=port,
        topology_mode=topology_mode_obj,
        sampling_conf=sampling_conf,
        tcp=tcp,
        udp=udp,
        network_settings=network_settings,
        bootnodes=bootnodes,
    )
