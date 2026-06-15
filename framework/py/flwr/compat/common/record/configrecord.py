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
"""Deprecated ConfigRecord compatibility APIs."""

from logging import WARN

from flwr.app.message import ConfigRecord
from flwr.app.typing import ConfigRecordValues
from flwr.common.logger import log


class ConfigsRecord(ConfigRecord):
    """Deprecated class ``ConfigsRecord``, use ``ConfigRecord`` instead."""

    _warning_logged = False

    def __init__(
        self,
        config_dict: dict[str, ConfigRecordValues] | None = None,
        keep_input: bool = True,
    ) -> None:
        if not ConfigsRecord._warning_logged:
            ConfigsRecord._warning_logged = True
            log(
                WARN,
                "The `ConfigsRecord` class has been renamed to `ConfigRecord`. "
                "Support for `ConfigsRecord` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(config_dict, keep_input)
