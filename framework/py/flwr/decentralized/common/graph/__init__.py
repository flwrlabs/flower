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
"""Graph APIs for decentralized deploy and simulation topologies."""

from .api import (
    GraphMapping,
    RandomExact,
    RandomInput,
    RandomRange,
    TopologyKind,
    TopologyMode,
    convert_graph_to_csr,
    convert_graph_to_static_topology,
    generate_deploy_topology_yaml,
    generate_simulation_csr,
    generate_simulation_graph,
    topology_mode_dynamic,
    topology_mode_static,
    write_static_topology_yaml,
)
