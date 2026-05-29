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
"""Metrics collected by the Virtual Client Engine."""


import threading
from dataclasses import dataclass, field


@dataclass
class VceMetrics:
    """Thread-safe runtime accumulator for Simulation Runtime runs."""

    clientapp_runtime: float = 0.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def add_clientapp_runtime(self, runtime: float) -> None:
        """Add ClientApp execution runtime in seconds."""
        with self._lock:
            self.clientapp_runtime += runtime
