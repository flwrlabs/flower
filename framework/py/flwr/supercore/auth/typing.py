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
"""Flower account authentication type definitions."""


from dataclasses import dataclass


@dataclass
class AccountAuthLoginDetails:
    """Account authentication login details."""

    authn_type: str
    device_code: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass
class AccountAuthCredentials:
    """Account authentication tokens."""

    access_token: str
    refresh_token: str


@dataclass
class AccountInfo:
    """User information for event log."""

    flwr_aid: str
    account_name: str
