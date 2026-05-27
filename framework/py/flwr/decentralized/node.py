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
"""Decentralized node abstraction for app registration and execution."""

from __future__ import annotations
import logging
from typing import Optional

from flwr.common.logger import log
from flwr.decentralized.common.graph import TopologyMode, topology_mode_dynamic
from flwr.decentralized.common.network import NetworkSettings
from flwr.decentralized.common.sampling import Configuration
from nodemanager.application._application import App
from nodemanager.node._node import Node


class DNode(Node):
    """Decentralized Federated Learning node orchestration class.

    This class wraps a :class:`nodemanager.node.Node` to simplify node startup
    and app execution in decentralized FL scenarios. It optionally creates and
    injects a network configuration file when running in dynamic topology mode,
    then delegates all networking and runtime behavior to the parent class.

    Attributes
    ----------
    logger : logging.Logger
        Instance logger named after the class.
    context : str
        Runtime context identifier.
        Only the nodes sharing the same context will be able to discover each
        other and communicate.
        For example, nodes with context "classification" will form a separate
        network from those with context "llm".
    address : str
        Bind or advertised host address used by the node.
    port : int
        Listening port used by the node.
    tcp : bool
        Whether TCP transport is enabled.
    udp : bool
        Whether UDP transport is enabled.
    topology_mode : TopologyMode
        Topology management mode for peer discovery and connectivity.
    config_path : str | None
        Path to the generated network configuration file passed to the parent
        node when available.
    network_settings : NetworkSettings | None
        Optional low-level network runtime settings forwarded to
        :class:`Node`.
    bootnodes : list | None
        Optional list of bootstrap peers used for initial network discovery.

    Notes
    -----
    In ``topology_mode_dynamic()``, a ``netwk`` configuration object is
    expected. If configuration creation fails, a warning is logged and node
    initialization continues with ``config_path=None``.
    """

    # one too-many attribute; pylint: disable=too-many-instance-attributes
    # pylint: disable-next=too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        context: str,
        address: str,
        port: int,
        topology_mode: Optional[TopologyMode] = None,
        sampling_conf: Optional[Configuration] = None,
        tcp: bool = True,
        udp: bool = False,
        network_settings: Optional[NetworkSettings] = None,
        bootnodes: Optional[list] = None,
    ) -> None:
        """Initialize a decentralized node and optional network configuration.

        Parameters
        ----------
        context : str
            Runtime context name used by the underlying node.
            Only the nodes sharing the same context will be able to discover
            each other and communicate.
            For example, nodes with context "classification" will form a
            separate network from those with context "llm".
        address : str
            Host/IP address used by the node.
        port : int
            Port used by the node for incoming communications.
        topology_mode : TopologyMode | None, default=None
            Topology mode controlling peer discovery strategy.
        sampling_conf : Configuration, optional
            Sampling configuration builder. When provided, ``create()`` is
            called and the generated ``config_file`` path is forwarded to the
            base node.
        tcp : bool, default=True
            Enable TCP transport.
        udp : bool, default=False
            Enable UDP transport.
        network_settings : NetworkSettings, optional
            Additional network settings forwarded to the parent node.
        bootnodes : list, optional
            Initial peers for bootstrap/discovery.

        Raises
        ------
        ValueError
            Raised internally when dynamic topology is requested without a
            network configuration object. The error is logged as a warning and
            initialization then proceeds.

        Examples
        --------
        >>> from declearn.decentralized.graph import topology_mode_dynamic
        >>> node = DNode(
        ...     context="eval",
        ...     address="0.0.0.0",
        ...     port=9100,
        ...     topology_mode=topology_mode_dynamic(),
        ...     sampling_conf=Configuration(...),
        ...     tcp=True,
        ...     udp=False,
        ... )
        """

        if topology_mode is None:
            topology_mode = topology_mode_dynamic()

        config_path = None

        try:
            if topology_mode == TopologyMode.dynamic() and sampling_conf is None:
                raise ValueError(
                    "A sampling configuration must be provided for dynamic "
                    "topology mode."
                )
            elif sampling_conf is not None:
                # Attempt to create the sampling configuration file and extract
                # the path.
                sampling_conf.create()
                config_path = sampling_conf.config_file
            else:
                config_path = None
        except Exception as e:
            log(
                logging.WARNING,
                "The sampling configuration could not be created: %s",
                e,
            )

        # Initialize the parent node with the generated config path when
        # available.
        super().__init__(
            context=context,
            address=address,
            port=port,
            tcp=tcp,
            udp=udp,
            topology_mode=topology_mode,
            config_path=config_path,
            network_settings=network_settings,
            bootnodes=bootnodes,
        )

    def create_node(self) -> None:
        """Start the node runtime.

        This is a convenience wrapper around :meth:`Node.start`.

        Examples
        --------
        >>> node.create_node()
        """
        self.start()

    def run_node(self, app: App, timeout: int = 500) -> None:
        """Attach an application to the node and run the event loop.

        The provided app is linked to the current node instance, registered by
        its name, then the node run loop is started.

        Parameters
        ----------
        app : App
            Application instance to register and execute on this node.
        timeout : int, default=500
            Maximum run duration or polling timeout (as interpreted by
            :meth:`Node.run`).

        Examples
        --------
        >>> app = App(name="my_fl_app")
        >>> node.run_node(app=app, timeout=500)
        """

        app.node = self

        self.register(app_name=app.name, app=app)

        self.run(timeout=timeout)


def start_node(
    node: DNode,
    applications: list[App],
    timeout: int = 500,
) -> None:
    """Helper function to start a decentralized node with a given applications.

    This function is a simple wrapper around the
    :meth:`DecentralizedNode.run_node` method for convenience.

    Parameters
    ----------
    node : DNode
        The decentralized node instance to run.
    applications : list[App]
        The list of application instances to execute on the node.
    timeout : int, default=500
        Maximum run duration or polling timeout (as interpreted by
        :meth:`Node.run`).

    Examples
    --------
    >>> node = DNode(...)
    >>> app1 = App(name="app1", ...)
    >>> app2 = App(name="app2", ...)

    >>> run_node(node, applications=[app1, app2], timeout=500)
    """

    for app in applications:
        # Link the app to the node and register it by name.
        # This allows the app to access node functionalities
        # and be discoverable by other peers. In the rust core there is
        # not a strict link between apps and nodes, but in the python wrapper
        # we need to set the node reference in the app for it to be able
        # to call node methods and access network features.
        app.node = node

        # Register the app with the node using its name as the identifier.
        # This makes the app discoverable and allows it to receive messages
        # sent to its name.
        node.register(app_name=app.name, app=app)

    # After all applications are registered, start the node's event loop to
    # begin processing messages and executing app logic.
    node.run(timeout=timeout)

    # Unregister the applications after the node run loop ends (optional
    # cleanup).
    for app in applications:
        node.unregister(app_name=app.name)
