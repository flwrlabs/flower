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
"""Tests for simulation startup order."""


from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest

from flwr.app import Context, RecordDict
from flwr.app.user_config import UserConfig
from flwr.clientapp import ClientApp
from flwr.common.seeding import RANDOM_SEED_KEY
from flwr.serverapp import ServerApp
from flwr.supercore.run import Run
from flwr.supercore.telemetry import EventType

# The package __init__ binds this name to the public function, which hides the module.
run_simulation_module = importlib.import_module("flwr.simulation.run_simulation")


def _run_with_stubs(
    monkeypatch: pytest.MonkeyPatch,
    run_config: UserConfig,
) -> list[str]:
    """Run ``_main_loop`` with the runtime calls replaced by ordered stubs."""
    order: list[str] = []
    context = Context(
        run_id=7,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )
    run = Run.create_empty(run_id=7)
    run.primary_task_id = 9

    def prepare(**_kwargs: object) -> Mock:
        order.append("prepare")
        return Mock()

    def start_server(**kwargs: object) -> Mock:
        order.append("server")
        ctx_queue = kwargs["ctx_queue"]
        assert hasattr(ctx_queue, "put")
        ctx_queue.put(context)
        return Mock()

    def run_prepared(*_args: object, **_kwargs: object) -> None:
        order.append("run")

    def start_vce(**_kwargs: object) -> None:
        order.append("start_vce")

    monkeypatch.setattr(run_simulation_module.vce, "prepare_seeded_vce", prepare)
    monkeypatch.setattr(run_simulation_module, "run_serverapp_th", start_server)
    monkeypatch.setattr(run_simulation_module.vce, "run_prepared_vce", run_prepared)
    monkeypatch.setattr(run_simulation_module.vce, "start_vce", start_vce)

    run_simulation_module._main_loop(  # pylint: disable=protected-access
        num_supernodes=4,
        backend_name="ray",
        backend_config_stream="{}",
        app_dir=".",
        is_app=False,
        enable_tf_gpu_growth=False,
        run=run,
        exit_event=EventType.PYTHON_API_RUN_SIMULATION_LEAVE,
        client_app=ClientApp(),
        server_app=ServerApp(),
        server_app_context=context,
    )
    return order


def test_seeded_simulation_prepares_before_the_server_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node registration and Ray setup finish before the ServerApp thread."""
    order = _run_with_stubs(monkeypatch, {RANDOM_SEED_KEY: 3})
    assert order == ["prepare", "server", "run"]


def test_unseeded_simulation_starts_the_server_thread_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs without ``random-seed`` keep the previous startup order."""
    order = _run_with_stubs(monkeypatch, {})
    assert order == ["server", "start_vce"]
