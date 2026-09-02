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
"""Tests for backwards-compatible Flower logger exports."""

from importlib import import_module

from flwr.common import logger as common_public_logger
from flwr.common import configure as common_configure
from flwr.common import log as common_log
from flwr.supercore import log as supercore_log

common_logger = import_module("flwr.common.logger")
supercore_logger = import_module("flwr.supercore.logger")


def test_common_logger_reexports_supercore_implementation() -> None:
    """Verify legacy logger exports refer to the SuperCore implementation."""
    assert common_logger.__all__ == supercore_logger.__all__
    for name in supercore_logger.__all__:
        assert getattr(common_logger, name) is getattr(supercore_logger, name)


def test_common_log_reexports_supercore_log() -> None:
    """Verify the legacy package-level log export remains compatible."""
    assert common_log is supercore_log
    assert common_configure is supercore_logger.configure


def test_common_logger_reexports_supercore_logger() -> None:
    """Verify the supported package-level logger export remains compatible."""
    assert common_public_logger is supercore_logger.logger
