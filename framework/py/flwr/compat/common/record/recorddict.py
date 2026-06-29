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
"""Deprecated RecordDict compatibility APIs."""

from logging import WARN

from flwr.app.message import ArrayRecord, ConfigRecord, MetricRecord, RecordDict
from flwr.common.logger import log

RecordType = ArrayRecord | MetricRecord | ConfigRecord


class RecordSet(RecordDict):
    """Deprecated class ``RecordSet``, use ``RecordDict`` instead."""

    _warning_logged = False

    def __init__(
        self,
        records: dict[str, RecordType] | None = None,
        *,
        parameters_records: dict[str, ArrayRecord] | None = None,
        metrics_records: dict[str, MetricRecord] | None = None,
        configs_records: dict[str, ConfigRecord] | None = None,
    ) -> None:
        if not RecordSet._warning_logged:
            RecordSet._warning_logged = True
            log(
                WARN,
                "The `RecordSet` class has been renamed to `RecordDict`. "
                "Support for `RecordSet` will be removed in a future release. "
                "Please update your code accordingly.",
            )
        super().__init__(
            records,
            parameters_records=parameters_records,
            metrics_records=metrics_records,
            configs_records=configs_records,
        )
