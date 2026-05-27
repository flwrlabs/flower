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
"""RuntimeNode: resolved configuration container for a DNode instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from flwr.decentralized.common.network import NetworkSettings
from flwr.decentralized.common.sampling import Configuration


@dataclass
class RuntimeNode:
    """Resolved configuration for a :class:`~flwr.decentralized.node.DNode`.

    All fields have been validated and are ready to be forwarded directly to
    the ``DNode`` constructor.

    Parameters
    ----------
    context : str
        Runtime context name used by the underlying node.
        Only nodes sharing the same context can discover each other and
        communicate.
    address : str
        Host / IP address used by the node.
    port : int
        Port used by the node for incoming communications.
    topology_mode : Any
        Resolved ``TopologyMode`` object (dynamic or static) produced by
        :func:`~flwr.decentralized.common.graph.topology_mode_dynamic` or
        :func:`~flwr.decentralized.common.graph.topology_mode_static`.
    sampling_conf : Configuration | None
        Sampling configuration for dynamic topology mode.
    tcp : bool
        Enable TCP transport (default ``True``).
    udp : bool
        Enable UDP transport (default ``False``).
    network_settings : NetworkSettings | None
        Low-level network runtime settings.
    bootnodes : list | None
        Initial bootstrap peers for network discovery.

    Examples
    --------
    Instantiate manually:

    >>> from flwr.decentralized.common.graph import topology_mode_dynamic
    >>> node = RuntimeNode(
    ...     context="classification",
    ...     address="0.0.0.0",
    ...     port=9100,
    ...     topology_mode=topology_mode_dynamic(),
    ... )

    Build from CLI arguments via :func:`~flwr.decentralized.common.args.get_args_nodes`:

    >>> from flwr.decentralized.common.args import get_args_nodes
    >>> runtime_node = get_args_nodes()
    >>> dnode = DNode(
    ...     context=runtime_node.context,
    ...     address=runtime_node.address,
    ...     port=runtime_node.port,
    ...     topology_mode=runtime_node.topology_mode,
    ...     sampling_conf=runtime_node.sampling_conf,
    ...     tcp=runtime_node.tcp,
    ...     udp=runtime_node.udp,
    ...     network_settings=runtime_node.network_settings,
    ...     bootnodes=runtime_node.bootnodes,
    ... )
    """

    context: str
    address: str
    port: int
    topology_mode: Any
    sampling_conf: Optional[Configuration] = None
    tcp: bool = True
    udp: bool = False
    network_settings: Optional[NetworkSettings] = None
    bootnodes: Optional[List[str]] = field(default=None)

    def to_dnode_kwargs(self) -> dict:
        """Return a dict of keyword arguments ready to unpack into ``DNode()``.

        Returns
        -------
        dict
            Keyword arguments matching the ``DNode.__init__`` signature.

        Examples
        --------
        >>> dnode = DNode(**runtime_node.to_dnode_kwargs())
        """
        return {
            "context": self.context,
            "address": self.address,
            "port": self.port,
            "topology_mode": self.topology_mode,
            "sampling_conf": self.sampling_conf,
            "tcp": self.tcp,
            "udp": self.udp,
            "network_settings": self.network_settings,
            "bootnodes": self.bootnodes,
        }
