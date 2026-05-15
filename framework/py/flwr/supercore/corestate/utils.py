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
"""Utility functions for CoreState."""


from os import urandom


def generate_rand_int_from_bytes(
    num_bytes: int, exclude: set[int] | None = None
) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes.

    If `exclude` is set, this function guarantees such number is not returned.
    """
    num = int.from_bytes(urandom(num_bytes), "little", signed=False)

    if exclude:
        while num in exclude:
            num = int.from_bytes(urandom(num_bytes), "little", signed=False)
    return num
