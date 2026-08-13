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
"""Utilities shared by protobuf-over-HTTP client interceptors."""

import requests


def add_headers(
    request: requests.PreparedRequest,
    headers: dict[str, str],
) -> None:
    """Add headers while rejecting values already provided by another layer."""
    duplicates = {name for name in headers if name in request.headers}
    if duplicates:
        raise RuntimeError(
            f"HTTP request already contains headers: {', '.join(sorted(duplicates))}"
        )
    request.headers.update(headers)
