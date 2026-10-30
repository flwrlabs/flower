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
"""Tests for Ray Actor pool sizing."""


from unittest import TestCase
from unittest.mock import patch

from flwr.simulation.ray_transport.ray_actor import pool_size_from_resources


class TestPoolSizeFromResources(TestCase):
    """Tests for the per-node ActorPool size calculation."""

    def test_gpus_requested_but_none_detected(self) -> None:
        """Test that an unsatisfiable GPU request names both possible causes.

        A cluster reporting GPUs=0 cannot satisfy any non-zero `num_gpus`, so the
        generic "lowering these resources could help" advice does not describe the
        fix. The cluster may genuinely have no GPU, or Ray may have failed to
        discover one, and the warning must not assert either.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with patch(
            "flwr.simulation.ray_transport.ray_actor.ray.nodes",
            return_value=[{"Resources": {"CPU": 8}}],  # CPUs aplenty, no GPU
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("Set `num_gpus=0`", output)
        self.assertIn("init-args-num-gpus", output)
        self.assertNotIn("Lowering these resources could help", output)

    def test_insufficient_cpus_keeps_generic_warning(self) -> None:
        """Test that a genuine resource shortage still advises lowering resources."""
        client_resources: dict[str, int | float] = {"num_cpus": 16, "num_gpus": 0}
        with patch(
            "flwr.simulation.ray_transport.ray_actor.ray.nodes",
            return_value=[{"Resources": {"CPU": 8}}],  # Too few CPUs per client
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("Lowering these resources could help", output)
        self.assertNotIn("set `num_gpus=0`", output)

    def test_gpu_on_another_node_keeps_generic_warning(self) -> None:
        """Test that a GPU on any node rules out the zero-GPU warning.

        The per-node counts only retain the last node's values, so a cluster whose GPU
        sits on an earlier node must not be reported as having no GPU at all. Here the
        GPU exists but is smaller than requested, which lowering `num_gpus` would fix.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 4}
        with patch(
            "flwr.simulation.ray_transport.ray_actor.ray.nodes",
            return_value=[
                {"Resources": {"CPU": 8, "GPU": 1}},  # Has a GPU, fewer than requested
                {"Resources": {"CPU": 8}},  # No GPU, and it is the last node
            ],
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("Lowering these resources could help", output)
        self.assertNotIn("set `num_gpus=0`", output)

    def test_case_with_no_usable_node(self) -> None:
        """Test that a cluster with no usable node raises ValueError.

        The reported CPU/GPU counts are only bound inside the per-node loop, so an
        empty node list must not fall through to a NameError.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0}
        node_lists: list[list[dict[str, dict[str, float]]]] = [
            [],  # Ray reports no nodes at all
            [{"Resources": {}}],  # A node that has detached, reporting no resources
        ]
        for nodes in node_lists:
            with self.subTest(nodes=nodes):
                with patch(
                    "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                    return_value=nodes,
                ):
                    with self.assertRaises(ValueError):
                        pool_size_from_resources(client_resources)
