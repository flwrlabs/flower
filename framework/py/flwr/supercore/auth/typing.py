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
class AccountAuthLoginDetails:  # pylint: disable=too-many-instance-attributes
    """Account authentication login details."""

    authn_type: str
    device_code: str
    verification_uri_complete: str
    expires_in: int
    interval: int
    verification_uri: str = ""
    user_code: str = ""
    supports_cli_signup: bool = False


@dataclass
class AccountAuthCredentials:
    """Account authentication tokens."""

    access_token: str
    refresh_token: str


@dataclass(frozen=True)
class AuthenticatedUser:
    """User identity within the deployment's single configured issuer."""

    # Keep the existing subject-based ID. The issuer is deployment configuration,
    # not part of a new user-identity key or account-linking schema.
    user_id: str
    roles: tuple[str, ...] = ()
    # Profile hints are optional and do not define the local account identity.
    provider_username: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    email: str | None = None
    # Account resolution supplies this only after local admission succeeds.
    handle: str | None = None
