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
"""Tests for ServerApp."""


from collections.abc import Iterator
from unittest.mock import MagicMock, Mock

import pytest

from flwr.app import Context, RecordDict
from flwr.server import ServerConfig
from flwr.serverapp import Grid

from .server_app import ServerApp


def test_server_app_custom_mode() -> None:
    """Test sampling w/o criterion."""
    # Prepare
    app = ServerApp()
    grid = MagicMock()
    context = Context(
        run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config={}
    )

    called = {"called": False}

    # pylint: disable=unused-argument
    @app.main()
    def custom_main(grid: Grid, context: Context) -> None:
        called["called"] = True

    # pylint: enable=unused-argument

    # Execute
    app(grid, context)

    # Assert
    assert called["called"]


def test_server_app_exception_when_both_modes() -> None:
    """Test ServerApp error when both compat mode and custom fns are used."""
    # Prepare
    app = ServerApp(config=ServerConfig(num_rounds=3))

    # Execute and assert
    with pytest.raises(ValueError):
        # pylint: disable=unused-argument
        @app.main()
        def custom_main(grid: Grid, context: Context) -> None:
            pass

        # pylint: enable=unused-argument


def test_lifespan_success() -> None:
    """Test the lifespan decorator with success."""
    # Prepare
    app = ServerApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
        pass

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifespan_failure() -> None:
    """Test the lifespan decorator with failure."""
    # Prepare
    app = ServerApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifespan_no_yield() -> None:
    """Test the lifespan decorator with no yield."""
    # Prepare
    app = ServerApp()
    enter_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:  # type: ignore
        enter_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()


def test_lifespan_multiple_yields() -> None:
    """Test the lifespan decorator with multiple yields."""
    # Prepare
    app = ServerApp()
    enter_code = Mock()
    middle_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        middle_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    middle_code.assert_called_once()
    exit_code.assert_not_called()
