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
"""Flower ServerApp exceptions."""


from flwr.app.exception import AppExitException
from flwr.supercore.exit import ExitCode


class InconsistentMessageReplies(AppExitException):
    """Exception triggered when replies are inconsistent and therefore aggregation must
    be skipped."""

    exit_code = ExitCode.SERVERAPP_STRATEGY_PRECONDITION_UNMET

    def __init__(self, reason: str):
        super().__init__(reason)


class AggregationError(AppExitException):
    """Exception triggered when aggregation fails."""

    exit_code = ExitCode.SERVERAPP_STRATEGY_AGGREGATION_ERROR

    def __init__(self, reason: str):
        super().__init__(reason)
