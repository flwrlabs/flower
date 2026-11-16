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
"""Deprecated record compatibility APIs."""

from .arrayrecord import ParametersRecord
from .configrecord import ConfigsRecord
from .conversion_utils import array_from_numpy
from .metricrecord import MetricsRecord
from .recorddict import RecordSet

__all__ = [
    "ConfigsRecord",
    "MetricsRecord",
    "ParametersRecord",
    "RecordSet",
    "array_from_numpy",
]
