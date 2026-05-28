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
"""Deprecated ArrayRecord compatibility APIs."""

from logging import WARN
from typing import Any

from flwr.app.message import ArrayRecord
from flwr.common.logger import log


class ParametersRecord(ArrayRecord):
    """Deprecated class ``ParametersRecord``, use ``ArrayRecord`` instead."""

    _warning_logged = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not ParametersRecord._warning_logged:
            ParametersRecord._warning_logged = True
            log(
                WARN,
                "The `ParametersRecord` class has been renamed to `ArrayRecord`. "
                "Support for `ParametersRecord` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(*args, **kwargs)
