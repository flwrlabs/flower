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
"""Shared FlowerError formatting utilities."""


from .base import FlowerError


def format_flower_error(err: FlowerError) -> str:
    """Return a user-facing message for a FlowerError."""
    msg = err.message
    if err.public_details:
        msg += f"\n{err.public_details}"
    return msg
