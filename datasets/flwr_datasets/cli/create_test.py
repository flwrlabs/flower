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
"""Test for Flower Datasets command line interface `create` command."""


import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
import typer

from datasets.load import DatasetNotFoundError

from . import create as create_module
from .create import create


class _FakePartition:
    """Fake dataset partition used to capture save-to-disk calls."""

    def __init__(self, saved_dirs: list[Path]) -> None:
        """Initialize the fake partition."""
        self._saved_dirs = saved_dirs

    def save_to_disk(self, out_dir: Path) -> None:
        """Record the output directory instead of writing to disk."""
        self._saved_dirs.append(out_dir)


class _FakeFederatedDataset:
    """Fake FederatedDataset that records partition loading behavior."""

    def __init__(self, calls: dict[str, Any]) -> None:
        """Initialize the fake federated dataset."""
        self._calls = calls

    def load_partition(self, *, partition_id: int) -> _FakePartition:
        """Simulate loading a partition and record calls."""
        self._calls.setdefault("loaded_ids", []).append(partition_id)
        return _FakePartition(self._calls.setdefault("saved_dirs", []))


def test_create_raises_on_non_positive_num_partitions(tmp_path: Path) -> None:
    """Ensure `create` fails when `num_partitions` is not a positive integer."""
    with pytest.raises(click.ClickException, match="positive integer"):
        create(dataset_name="user/ds", num_partitions=0, out_dir=tmp_path)


def test_create_raises_click_exception_when_dataset_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure `create` raises a user-friendly error when the dataset is.

    missing/unreachable.
    """
    # Create a unique dataset name
    out_dir = tmp_path / "out"
    dataset_name = "does-not-exist/dataset"

    # Avoid overwrite prompt path
    monkeypatch.setattr(Path, "exists", lambda _self: False)

    # Avoid touching the real filesystem in this unit test
    monkeypatch.setattr(Path, "mkdir", lambda _self, **_kwargs: None)

    # Mock partitioner
    monkeypatch.setattr(
        create_module,
        "IidPartitioner",
        lambda *, num_partitions: SimpleNamespace(num_partitions=num_partitions),
    )

    # Ensure the command handles DatasetNotFoundError specifically
    def _raise_fds(
        *, dataset: str, partitioners: dict[str, object]
    ) -> _FakeFederatedDataset:
        raise DatasetNotFoundError()

    monkeypatch.setattr(create_module, "FederatedDataset", _raise_fds)

    expected_msg = (
        f"Dataset '{dataset_name}' could not be found on the Hugging Face Hub, "
        "or network access is unavailable. "
        "Please verify the dataset identifier and your internet connection."
    )

    with pytest.raises(click.ClickException, match=re.escape(expected_msg)):
        create(dataset_name=dataset_name, num_partitions=2, out_dir=out_dir)


@dataclass(frozen=True)
class _CreateCase:
    """Single parametrized case for `create` output-directory behavior tests."""

    out_dir_exists: bool
    user_overwrite: bool | None
    expect_runs: bool
    expect_confirm_calls: int
    num_partitions: int = 3


@pytest.mark.parametrize(
    "case",
    [
        _CreateCase(
            out_dir_exists=False,
            user_overwrite=None,
            expect_runs=True,
            expect_confirm_calls=0,
        ),
        _CreateCase(
            out_dir_exists=True,
            user_overwrite=False,
            expect_runs=False,
            expect_confirm_calls=1,
        ),
        _CreateCase(
            out_dir_exists=True,
            user_overwrite=True,
            expect_runs=True,
            expect_confirm_calls=1,
        ),
    ],
)
def test_create_partitions_save_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: _CreateCase,
) -> None:
    """Test `create` behavior depending on whether the output directory exists."""
    out_dir = tmp_path / "out"
    calls: dict[str, Any] = {}
    confirm_calls: list[str] = []
    mkdir_calls: list[Path] = []

    def _exists(self: Path) -> bool:
        """Simulate existence of the output directory."""
        return case.out_dir_exists and self == out_dir

    def _confirm(message: str, default: bool = False) -> bool:
        """Simulate user response to overwrite confirmation."""
        del default  # unused
        confirm_calls.append(message)
        assert (
            case.user_overwrite is not None
        ), "confirm should not be called in this scenario"
        return case.user_overwrite

    def _mkdir(self: Path, parents: bool = False, exist_ok: bool = False) -> None:
        """Record directory creation attempts."""
        del parents, exist_ok  # unused
        mkdir_calls.append(self)

    monkeypatch.setattr(Path, "exists", _exists)
    monkeypatch.setattr(typer, "confirm", _confirm)
    monkeypatch.setattr(Path, "mkdir", _mkdir)

    if case.expect_runs:

        def _fake_partitioner(*, num_partitions: int) -> SimpleNamespace:
            """Record partitioner initialization."""
            calls["partitioner_num_partitions"] = num_partitions
            return SimpleNamespace(num_partitions=num_partitions)

        def _fake_fds(
            *, dataset: str, partitioners: dict[str, object]
        ) -> _FakeFederatedDataset:
            """Record dataset creation and return a fake federated dataset."""
            calls["dataset"] = dataset
            calls["partitioners"] = partitioners
            return _FakeFederatedDataset(calls)

        monkeypatch.setattr(create_module, "IidPartitioner", _fake_partitioner)
        monkeypatch.setattr(create_module, "FederatedDataset", _fake_fds)
    else:

        def _fail_partitioner(**_: object) -> None:
            raise AssertionError("IidPartitioner should not be called")

        def _fail_fds(**_: object) -> None:
            raise AssertionError("FederatedDataset should not be called")

        monkeypatch.setattr(create_module, "IidPartitioner", _fail_partitioner)
        monkeypatch.setattr(create_module, "FederatedDataset", _fail_fds)

    create(dataset_name="user/ds", num_partitions=case.num_partitions, out_dir=out_dir)

    assert len(confirm_calls) == case.expect_confirm_calls

    if not case.expect_runs:
        assert not mkdir_calls
        return

    assert mkdir_calls == [out_dir]
    assert calls["partitioner_num_partitions"] == case.num_partitions
    assert calls["dataset"] == "user/ds"
    assert "train" in calls["partitioners"]
    assert calls["loaded_ids"] == list(range(case.num_partitions))
    assert calls["saved_dirs"] == [
        out_dir / f"partition_{i}" for i in range(case.num_partitions)
    ]
