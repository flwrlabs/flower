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
"""Tests for `flwr run` command."""


from collections.abc import Iterator
from contextlib import contextmanager
import unittest
from unittest.mock import MagicMock, patch

from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions

from .run import run


@contextmanager
def _mock_cli_output_handler(*_args: object, **_kwargs: object) -> Iterator[bool]:
    yield False


class TestRun(unittest.TestCase):
    """Tests for the run command."""

    @patch("flwr.cli.run.run._run_with_control_api")
    @patch("flwr.cli.run.run.load_and_validate")
    @patch("flwr.cli.run.run.read_superlink_connection")
    @patch("flwr.cli.run.run.migrate")
    def test_run_options_only_connection_uses_control_api(
        self,
        mock_migrate: MagicMock,
        mock_read_superlink: MagicMock,
        mock_load_and_validate: MagicMock,
        mock_run_with_control_api: MagicMock,
    ) -> None:
        """`flwr run` should use Control API even for options-only local profiles."""
        del mock_migrate
        connection = SuperLinkConnection(
            name="local",
            options=SuperLinkSimulationOptions(num_supernodes=2),
        )
        mock_read_superlink.return_value = connection
        mock_load_and_validate.return_value = ({}, [])

        with patch("flwr.cli.run.run.cli_output_handler", _mock_cli_output_handler):
            run(app=".", superlink="local")

        mock_run_with_control_api.assert_called_once()
        assert mock_run_with_control_api.call_args.args[3] == connection
