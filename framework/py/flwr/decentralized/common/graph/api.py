"""Graph and topology APIs for decentralized deploy and simulation contexts.

Deploy-oriented APIs
--------------------
* ``topology_mode_dynamic``
* ``topology_mode_static``
* ``generate_deploy_topology_yaml``

Simulation-oriented APIs
------------------------
* ``generate_simulation_graph``
* ``generate_simulation_csr``
* ``convert_graph_to_csr``
* ``convert_graph_to_static_topology``
* ``write_static_topology_yaml``
"""

from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple, Union

from nodemanager.graph._graph import (  # type: ignore[import-untyped]
    GraphMapping,
    generate_csr_matrix,
    generate_topology_graph,
    graph_to_csr_matrix,
    graph_to_static_topology,
    write_static_topology_yaml,
)
from nodemanager.network._topology import (  # type: ignore[import-untyped]
    TopologyMode,
    generate_topology_yaml_file,
)
from nodemanager.network._topology_specs import (  # type: ignore[import-untyped]
    RandomExact,
    RandomInput,
    RandomRange,
    TopologyKind,
)
from nodemanager.utils._utils import CSRMatrix  # type: ignore[import-untyped]

__all__ = [
    "TopologyMode",
    "TopologyKind",
    "RandomExact",
    "RandomRange",
    "RandomInput",
    "GraphMapping",
    "topology_mode_dynamic",
    "topology_mode_static",
    "generate_deploy_topology_yaml",
    "generate_simulation_graph",
    "generate_simulation_csr",
    "convert_graph_to_csr",
    "convert_graph_to_static_topology",
    "write_static_topology_yaml",
]


def topology_mode_dynamic() -> Any:
    """Return a dynamic topology mode object for deploy runtime."""
    return TopologyMode.dynamic()


def topology_mode_static(
    config_path: Union[str, Path],
    node_name: str,
) -> Any:
    """Return a static topology mode object for deploy runtime.

    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the static topology configuration file.
    node_name : str
        Name of the node for which the static topology is being created.

    Returns
    -------
    Any
        A static topology mode object.
    """
    return TopologyMode.static(str(config_path), node_name)


def generate_deploy_topology_yaml(
    node_count: int,
    kind: Union[str, TopologyKind],
    output_path: Union[str, Path],
    random: Optional[RandomInput] = None,
) -> None:
    """Generate a topology YAML file for deploy static topology mode.

    Parameters
    ----------
    node_count : int
        The number of nodes in the topology.
    kind : Union[str, TopologyKind]
        The kind of topology to generate. Can be a string identifier
        or a TopologyKind enum value.
    output_path : Union[str, Path]
        The path where the generated YAML file will be saved.
    random : Optional[RandomInput], optional
        Optional random input configuration for topology generation,
        by default None.
    """
    generate_topology_yaml_file(
        node_count=node_count,
        kind=kind,
        output_path=output_path,
        random=random,
    )


def generate_simulation_graph(
    node_count: int,
    kind: Union[str, TopologyKind],
    random: Optional[RandomInput] = None,
    seed: Optional[int] = None,
) -> Any:
    """Generate a graph for simulation from a topology specification.

    Parameters
    ----------
    node_count : int
        The number of nodes in the topology.
    kind : Union[str, TopologyKind]
        The kind of topology to generate. Can be a string identifier or a
        TopologyKind enum value.
    random : Optional[RandomInput], optional
        Optional random input configuration for topology generation, by
        default None.
    seed : Optional[int], optional
        Optional random seed for reproducibility, by default None.

    Returns
    -------
    Any
        A graph object suitable for simulation.
    """
    return generate_topology_graph(
        node_count=node_count,
        kind=kind,
        random=random,
        seed=seed,
    )


def generate_simulation_csr(
    node_count: int,
    kind: Union[str, TopologyKind],
    random: Optional[RandomInput] = None,
    sampling: Any = None,
    seed: Optional[int] = None,
) -> Tuple[CSRMatrix, GraphMapping]:
    """Generate a CSR matrix for simulation from a topology specification.

    Parameters
    ----------
    node_count : int
        The number of nodes in the topology.
    kind : Union[str, TopologyKind]
        The kind of topology to generate. Can be a string identifier or a
        TopologyKind enum value.
    random : Optional[RandomInput], optional
        Optional random input configuration for topology generation, by
        default None.
    sampling : Any, optional
        Optional sampling configuration for topology generation, by
        default None.
    seed : Optional[int], optional
        Optional random seed for reproducibility, by default None.

    Returns
    -------
    Tuple[CSRMatrix, GraphMapping]
        A tuple containing the CSR matrix and the graph mapping.
    """
    return generate_csr_matrix(
        node_count=node_count,
        kind=kind,
        random=random,
        sampling=sampling,
        seed=seed,
    )


def convert_graph_to_csr(
    graph: Any,
    node_to_name: Optional[Mapping[Hashable, str]] = None,
    sampling: Any = None,
) -> Tuple[CSRMatrix, GraphMapping]:
    """Convert an existing graph into a simulation-ready CSR matrix.

    Parameters
    ----------
    graph : Any
        The graph to convert.
        This graph must be a networkx graph.
    node_to_name : Optional[Mapping[Hashable, str]], optional
        Mapping from node identifiers to names, by default None.
    sampling : Any, optional
        Optional sampling configuration to apply during conversion, by
        default None.

    Returns
    -------
    Tuple[CSRMatrix, GraphMapping]
        A tuple containing the CSR matrix and the graph mapping.
    """

    return graph_to_csr_matrix(
        graph=graph,
        node_to_name=node_to_name,
        sampling=sampling,
    )


def convert_graph_to_static_topology(
    graph: Any,
    node_to_name: Optional[Mapping[Hashable, str]] = None,
) -> Tuple[Dict[str, Any], GraphMapping]:
    """Convert an existing graph into a static topology dictionary in
    deployment settings.

    Parameters
    ----------
    graph : Any
        The graph to convert.
        This graph must be a networkx graph.
    node_to_name : Optional[Mapping[Hashable, str]], optional
        Mapping from node identifiers to names, by default None.

    Returns
    -------
    Tuple[Dict[str, Any], GraphMapping]
        A tuple containing the static topology dictionary and the graph
        mapping.
    """
    return graph_to_static_topology(graph=graph, node_to_name=node_to_name)
