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
"""Shared configuration loading for Flower services."""


import math
import os


def load_poll_interval(
    env_var: str, default: float, *, value: float | None = None
) -> float:
    """Load a polling interval in seconds, preferring an explicit value.

    Parameters
    ----------
    env_var : str
        Environment variable to read when no explicit value is provided.
    default : float
        Interval to use when the environment variable is absent.
    value : float | None
        Explicit interval overriding the environment, if provided.

    Returns
    -------
    float
        A finite interval between 0.01 and 60 seconds, inclusive.

    Raises
    ------
    ValueError
        If the selected value is not a number or is outside the accepted range.
    """
    raw_interval = value if value is not None else os.getenv(env_var, str(default))
    try:
        interval = float(raw_interval)
        if not math.isfinite(interval) or not 0.01 <= interval <= 60.0:
            raise ValueError
    except ValueError as err:
        raise ValueError(
            f"{env_var} must be a finite number of seconds between 0.01 and 60, "
            f"got {raw_interval!r}."
        ) from err
    return interval
