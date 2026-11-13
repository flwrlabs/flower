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
from unittest.mock import mock_open, patch

from flwr.simulation.ray_transport.ray_actor import (
    _is_amd_render_node,
    pool_size_from_resources,
)


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

    def test_zero_gpu_branch_also_reports_cpu_shortage(self) -> None:
        """Test that a concurrent CPU shortage is surfaced, not just the GPU one.

        When no node has enough CPUs either, telling the user to set `num_gpus=0`
        would still leave the pool empty -- the CPU shortage must be named too.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with patch(
            "flwr.simulation.ray_transport.ray_actor.ray.nodes",
            return_value=[{"Resources": {"CPU": 1}}],  # Too few CPUs, and no GPU
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("Set `num_gpus=0`", output)
        self.assertIn("no node has enough CPUs for even one client", output)

    def test_cpu_shortage_warning_reports_the_best_node_not_the_last(self) -> None:
        """Test that the reported CPU count is the cluster-wide best, not the last.

        `num_cpus` is only ever assigned the last node's value inside the per-node
        loop, so reporting it directly would understate the best node whenever a
        smaller-CPU node happens to come last in `ray.nodes()`.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with patch(
            "flwr.simulation.ray_transport.ray_actor.ray.nodes",
            # The best node (CPU=1) comes first, the smaller one (CPU=0.5) last.
            return_value=[{"Resources": {"CPU": 1}}, {"Resources": {"CPU": 0.5}}],
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("CPUs=1", output)
        self.assertNotIn("CPUs=0.5", output)

    def test_kfd_present_but_inaccessible_reports_permissions(self) -> None:
        """Test that an unopenable /dev/kfd is diagnosed as a permissions issue.

        A non-root host or container missing render-group access still has
        `/dev/kfd` on disk, but the process cannot open it. The hint must not point
        at the SMI library path in that case -- it would not fix anything.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}}],  # CPUs aplenty, no GPU
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.os.path.exists",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._can_open_device",
                return_value=False,
            ),
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("cannot open it", output)
        self.assertIn("render", output)
        self.assertNotIn("python -c 'import amdsmi'", output)

    def test_kfd_present_and_accessible_points_at_amdsmi(self) -> None:
        """Test that an openable /dev/kfd with a usable render node points at the
        `amdsmi` package rather than the pre-2.56 `librocm_smi64.so` path."""
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}}],  # CPUs aplenty, no GPU
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.os.path.exists",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._can_open_device",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.glob.glob",
                return_value=["/dev/dri/renderD128"],  # A usable render node exists
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
                return_value=True,
            ),
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("python -c 'import amdsmi'", output)
        self.assertIn("libamd_smi.so", output)
        self.assertNotIn("cannot open it", output)
        self.assertNotIn("no accessible `/dev/dri/renderD*` node", output)

    def test_kfd_accessible_render_node_non_amd_reports_vendor_mismatch(self) -> None:
        """Test that an accessible but non-AMD render node is not mistaken for a
        usable one.

        A hybrid host or container can expose an accessible render node (e.g. an
        integrated Intel GPU's `renderD128`) while the AMD-backed node is missing or
        permission-denied. The diagnostic must neither claim device access is
        complete nor point at ROCM_PATH in that case -- the real problem is that the
        AMD device itself is not accessible.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}}],  # CPUs aplenty, no GPU
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.os.path.exists",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._can_open_device",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.glob.glob",
                return_value=["/dev/dri/renderD128"],  # Accessible, but not AMD
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
                return_value=False,
            ),
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("not AMD-backed", output)
        self.assertNotIn("python -c 'import amdsmi'", output)
        self.assertNotIn("no accessible `/dev/dri/renderD*` node", output)

    def test_kfd_accessible_but_no_usable_render_node_reports_render_node(self) -> None:
        """Test that an openable /dev/kfd without a usable render node does not point
        at the SMI library -- it names the missing/inaccessible render node instead."""
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}}],  # CPUs aplenty, no GPU
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.os.path.exists",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._can_open_device",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.glob.glob",
                return_value=[],  # No /dev/dri/renderD* node found
            ),
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertNotIn("python -c 'import amdsmi'", output)
        self.assertNotIn("cannot open it", output)
        self.assertIn("no accessible `/dev/dri/renderD*` node", output)
        self.assertIn("render", output)

    def test_multi_node_cluster_skips_driver_host_diagnostics(self) -> None:
        """Test that a multi-node cluster skips the driver-only KFD diagnostic.

        `_log_kfd_diagnostics` only inspects the driver host's own `/dev/kfd` and
        render nodes. On a multi-node cluster that is misleading -- the driver's
        device state says nothing about which worker node lacks access -- so the
        single-node diagnostics must not run, and a generic per-node check message
        must be logged instead.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[
                    {"Resources": {"CPU": 8}},
                    {"Resources": {"CPU": 8}},
                ],
            ),
            # These would normally trigger the "points at amdsmi" branch on a
            # single-node cluster -- proving the gate suppresses it, not just that
            # it adds extra text.
            patch(
                "flwr.simulation.ray_transport.ray_actor.os.path.exists",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._can_open_device",
                return_value=True,
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.glob.glob",
                return_value=["/dev/dri/renderD128"],
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
                return_value=True,
            ),
        ):
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        output = "\n".join(logs.output)
        self.assertIn("multi-node Ray cluster", output)
        self.assertIn("each node individually", output)
        self.assertNotIn("python -c 'import amdsmi'", output)
        self.assertNotIn("cannot open it", output)
        self.assertNotIn("not AMD-backed", output)
        self.assertNotIn("no accessible `/dev/dri/renderD*` node", output)

    def test_single_remote_node_skips_local_kfd_probe(self) -> None:
        """Test that a single-node cluster not co-located with the driver skips the
        local `/dev/kfd` probe.

        `init_args` can hand `ray.init()` an `address` that joins this driver to a
        single-node cluster running on another host. `active_node_count == 1` alone
        does not mean the driver shares a host with that node, so the driver's own
        `/dev/kfd`/render-node state must not be probed or named in that case.
        """
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}, "NodeID": "remote-node"}],
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.get_runtime_context"
            ) as mock_get_runtime_context,
            patch(
                "flwr.simulation.ray_transport.ray_actor._log_kfd_diagnostics"
            ) as mock_log_kfd_diagnostics,
        ):
            mock_get_runtime_context.return_value.get_node_id.return_value = (
                "driver-node"
            )
            with self.assertLogs("flwr", level="WARNING") as logs:
                with self.assertRaises(ValueError):
                    pool_size_from_resources(client_resources)

        mock_log_kfd_diagnostics.assert_not_called()
        output = "\n".join(logs.output)
        self.assertIn("connected to a remote cluster", output)
        self.assertIn("remote-node", output)
        self.assertIn("driver-node", output)

    def test_single_local_node_runs_kfd_probe_when_colocated(self) -> None:
        """Test that a single-node cluster co-located with the driver still runs the
        local `/dev/kfd` probe."""
        client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
        with (
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.nodes",
                return_value=[{"Resources": {"CPU": 8}, "NodeID": "same-node"}],
            ),
            patch(
                "flwr.simulation.ray_transport.ray_actor.ray.get_runtime_context"
            ) as mock_get_runtime_context,
            patch(
                "flwr.simulation.ray_transport.ray_actor._log_kfd_diagnostics"
            ) as mock_log_kfd_diagnostics,
        ):
            mock_get_runtime_context.return_value.get_node_id.return_value = "same-node"
            with self.assertRaises(ValueError):
                pool_size_from_resources(client_resources)

        mock_log_kfd_diagnostics.assert_called_once()

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


class TestIsAmdRenderNode(TestCase):
    """Tests for the render-node vendor check used by the KFD diagnostic."""

    def test_amd_vendor_id_is_amd(self) -> None:
        """Test that a node whose sysfs vendor file reads AMD's PCI ID is AMD."""
        with patch("builtins.open", mock_open(read_data="0x1002\n")):
            self.assertTrue(_is_amd_render_node("/dev/dri/renderD128"))

    def test_non_amd_vendor_id_is_not_amd(self) -> None:
        """Test that an Intel-backed render node's sysfs vendor ID is rejected.

        This is the hybrid-host case: an accessible non-AMD render node must not be
        mistaken for a ROCm-capable one.
        """
        with patch("builtins.open", mock_open(read_data="0x8086\n")):
            self.assertFalse(_is_amd_render_node("/dev/dri/renderD128"))

    def test_missing_sysfs_vendor_file_is_unknown(self) -> None:
        """Test that an unreadable sysfs vendor file is reported as unknown.

        Some restricted containers do not mount sysfs; an unreadable vendor file
        says nothing about which GPU backs the node, so it must be neither
        certified as AMD nor rejected as non-AMD.
        """
        with patch("builtins.open", side_effect=OSError):
            self.assertIsNone(_is_amd_render_node("/dev/dri/renderD128"))
