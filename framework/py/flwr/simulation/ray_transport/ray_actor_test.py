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


import logging
from unittest.mock import mock_open, patch

import pytest

from flwr.simulation.ray_transport.ray_actor import (
    _is_amd_render_node,
    pool_size_from_resources,
)


def test_gpus_requested_but_none_detected(caplog: pytest.LogCaptureFixture) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "Set `num_gpus=0`" in output
    assert "init-args-num-gpus" in output
    assert "Lowering these resources could help" not in output


def test_insufficient_cpus_keeps_generic_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a genuine resource shortage still advises lowering resources."""
    client_resources: dict[str, int | float] = {"num_cpus": 16, "num_gpus": 0}
    with patch(
        "flwr.simulation.ray_transport.ray_actor.ray.nodes",
        return_value=[{"Resources": {"CPU": 8}}],  # Too few CPUs per client
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "Lowering these resources could help" in output
    assert "set `num_gpus=0`" not in output


def test_gpu_on_another_node_keeps_generic_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "Lowering these resources could help" in output
    assert "set `num_gpus=0`" not in output


def test_zero_gpu_branch_also_reports_cpu_shortage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a concurrent CPU shortage is surfaced, not just the GPU one.

    When no node has enough CPUs either, telling the user to set `num_gpus=0`
    would still leave the pool empty -- the CPU shortage must be named too.
    """
    client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}
    with patch(
        "flwr.simulation.ray_transport.ray_actor.ray.nodes",
        return_value=[{"Resources": {"CPU": 1}}],  # Too few CPUs, and no GPU
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "Set `num_gpus=0`" in output
    assert "no node has enough CPUs for even one client" in output


def test_cpu_shortage_warning_reports_the_best_node_not_the_last(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "CPUs=1" in output
    assert "CPUs=0.5" not in output


def test_kfd_present_but_inaccessible_reports_permissions(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "cannot open it" in output
    assert "render" in output
    assert "python -c 'import amdsmi'" not in output


def test_kfd_present_and_accessible_points_at_amdsmi(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "python -c 'import amdsmi'" in output
    assert "libamd_smi.so" in output
    assert "cannot open it" not in output
    assert "no accessible `/dev/dri/renderD*` node" not in output


def test_kfd_accessible_amd_node_with_denied_sibling_reports_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that an accessible AMD render node does not hide a denied AMD
    sibling, instead of discarding it before `_amd_access_message` can run.

    A single-node multi-GPU host can have one confirmed-AMD render node that
    opens while `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` allocates a
    `ClientApp` to a different, permission-denied AMD node. Pointing only at
    `amdsmi` would leave that allocated device's own failure unreported.
    """
    client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}

    def _can_open(path: str) -> bool | None:
        if path == "/dev/kfd":
            return True
        return path == "/dev/dri/renderD128"  # accessible; renderD129 is denied

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
            side_effect=_can_open,
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor.glob.glob",
            return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=True,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "python -c 'import amdsmi'" in output
    assert "renderD129" in output
    assert "permission-denied" in output
    assert "ROCR_VISIBLE_DEVICES" in output


def test_kfd_accessible_render_node_non_amd_reports_vendor_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "not AMD-backed" in output
    assert "python -c 'import amdsmi'" not in output
    assert "no accessible `/dev/dri/renderD*` node" not in output


def test_kfd_accessible_but_no_usable_render_node_reports_render_node(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "python -c 'import amdsmi'" not in output
    assert "cannot open it" not in output
    assert "no `/dev/dri/renderD*` node was found at all" in output
    assert "render" in output


def test_kfd_accessible_unknown_vendor_node_denied_reports_both_issues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a render node with unknown vendor and denied access reports
    both issues, rather than letting the vendor-undetermined message hide a
    concrete, actionable permission failure.

    `_is_amd_render_node` returning `None` (sysfs missing/unreadable) does not
    mean the node's access state is also unknown -- `_can_open_device` can
    still return a definite `False` for it, and that must be surfaced.
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
            # /dev/kfd itself opens fine; only the render node is denied.
            side_effect=lambda path: False if path != "/dev/kfd" else True,
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor.glob.glob",
            return_value=["/dev/dri/renderD128"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "this process cannot open it" in output
    assert "cannot determine its PCI vendor ID" in output
    assert "opens successfully" not in output
    assert "undetermined too" not in output


def test_kfd_accessible_unknown_vendor_node_inconclusive_reports_undetermined(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a render node with unknown vendor and inconclusive access
    reports both as undetermined, rather than falsely claiming a concrete
    permission denial that did not occur.
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
            # /dev/kfd itself opens fine; only the render node is inconclusive.
            side_effect=lambda path: None if path != "/dev/kfd" else True,
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor.glob.glob",
            return_value=["/dev/dri/renderD128"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "undetermined too" in output
    assert "this process cannot open it" not in output
    assert "opens successfully" not in output


def test_kfd_accessible_unknown_vendor_node_accessible_reports_vendor_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that a render node with unknown vendor but confirmed access only
    reports the vendor as undetermined, and is not mistaken for the
    confirmed-AMD-and-accessible branch that points at `amdsmi`.
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
            return_value=["/dev/dri/renderD128"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "opens successfully" in output
    assert "this process cannot open it" not in output
    assert "undetermined too" not in output
    assert "python -c 'import amdsmi'" not in output


def test_kfd_mixed_unknown_vendor_accessible_and_denied_reports_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that an accessible unknown-vendor node's message also surfaces a
    denied unknown-vendor node on the same host, instead of suppressing it.

    A restricted hybrid host can expose two render nodes with unreadable
    sysfs: one opens (e.g. an unrelated Intel GPU) while the other is
    permission-denied -- and the denied one could be the actual AMD GPU.
    Reporting only the accessible node would leave no path to the real
    access problem once the user finds the open node is not AMD's.
    """
    client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}

    def _can_open(path: str) -> bool | None:
        if path == "/dev/kfd":
            return True
        return path == "/dev/dri/renderD128"  # accessible; renderD129 is denied

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
            side_effect=_can_open,
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor.glob.glob",
            return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "opens successfully" in output
    assert "renderD128" in output
    assert "this process cannot open it" in output
    assert "undetermined too" not in output


def test_kfd_mixed_unknown_vendor_denied_and_inconclusive_reports_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that unknown-vendor denied and inconclusive nodes are reported
    together, rather than a plain `elif` silently dropping one of them.

    Neither outcome rules out the other: a permission fix for the denied
    node would leave the inconclusive node still failing, and vice versa.
    """
    client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0.25}

    def _can_open(path: str) -> bool | None:
        if path == "/dev/kfd":
            return True
        if path == "/dev/dri/renderD128":
            return False  # denied
        return None  # renderD129 inconclusive

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
            side_effect=_can_open,
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor.glob.glob",
            return_value=["/dev/dri/renderD128", "/dev/dri/renderD129"],
        ),
        patch(
            "flwr.simulation.ray_transport.ray_actor._is_amd_render_node",
            return_value=None,
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "Opening them also failed for two different reasons" in output
    assert "renderD128" in output
    assert "renderD129" in output
    assert "opens successfully" not in output


def test_multi_node_cluster_skips_driver_host_diagnostics(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    output = caplog.text
    assert "multi-node Ray cluster" in output
    assert "each node individually" in output
    assert "python -c 'import amdsmi'" not in output
    assert "cannot open it" not in output
    assert "not AMD-backed" not in output
    assert "no accessible `/dev/dri/renderD*` node" not in output


def test_single_remote_node_skips_local_kfd_probe(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        mock_get_runtime_context.return_value.get_node_id.return_value = "driver-node"
        with caplog.at_level(logging.WARNING, logger="flwr"):
            with pytest.raises(ValueError):
                pool_size_from_resources(client_resources)

    mock_log_kfd_diagnostics.assert_not_called()
    output = caplog.text
    assert "connected to a remote cluster" in output
    assert "remote-node" in output
    assert "driver-node" in output


def test_single_local_node_runs_kfd_probe_when_colocated() -> None:
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
        with pytest.raises(ValueError):
            pool_size_from_resources(client_resources)

    mock_log_kfd_diagnostics.assert_called_once()


@pytest.mark.parametrize(
    "nodes",
    [
        [],  # Ray reports no nodes at all
        [{"Resources": {}}],  # A node that has detached, reporting no resources
    ],
)
def test_case_with_no_usable_node(
    nodes: list[dict[str, dict[str, float]]],
) -> None:
    """Test that a cluster with no usable node raises ValueError.

    The reported CPU/GPU counts are only bound inside the per-node loop, so an
    empty node list must not fall through to a NameError.
    """
    client_resources: dict[str, int | float] = {"num_cpus": 2, "num_gpus": 0}
    with patch(
        "flwr.simulation.ray_transport.ray_actor.ray.nodes",
        return_value=nodes,
    ):
        with pytest.raises(ValueError):
            pool_size_from_resources(client_resources)


def test_amd_vendor_id_is_amd() -> None:
    """Test that a node whose sysfs vendor file reads AMD's PCI ID is AMD."""
    with patch("builtins.open", mock_open(read_data="0x1002\n")):
        assert _is_amd_render_node("/dev/dri/renderD128") is True


def test_non_amd_vendor_id_is_not_amd() -> None:
    """Test that an Intel-backed render node's sysfs vendor ID is rejected.

    This is the hybrid-host case: an accessible non-AMD render node must not be
    mistaken for a ROCm-capable one.
    """
    with patch("builtins.open", mock_open(read_data="0x8086\n")):
        assert _is_amd_render_node("/dev/dri/renderD128") is False


def test_missing_sysfs_vendor_file_is_unknown() -> None:
    """Test that an unreadable sysfs vendor file is reported as unknown.

    Some restricted containers do not mount sysfs; an unreadable vendor file
    says nothing about which GPU backs the node, so it must be neither
    certified as AMD nor rejected as non-AMD.
    """
    with patch("builtins.open", side_effect=OSError):
        assert _is_amd_render_node("/dev/dri/renderD128") is None
