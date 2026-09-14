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
"""Versioned local Guardian protocol for the capability proof of concept."""

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from flwr.supercore.run import Run

GUARDIAN_PROTOCOL_VERSION = "v1"
GUARDIAN_URL_ENV = "FLWR_GUARDIAN_URL"
GUARDIAN_TIMEOUT_ENV = "FLWR_GUARDIAN_TIMEOUT"
DEFAULT_GUARDIAN_TIMEOUT = 5.0


class GuardianVerificationError(Exception):
    """Raised when a capability cannot be verified by the Guardian."""


def verify_capability(run: Run) -> None:
    """Fail closed unless the local Guardian authorizes the expected binding."""
    if run.capability_required is not True:
        return
    if not run.capability_package:
        raise GuardianVerificationError("no capability was routed to this SuperNode")

    guardian_url = os.getenv(GUARDIAN_URL_ENV, "").strip()
    if not guardian_url:
        raise GuardianVerificationError(f"{GUARDIAN_URL_ENV} is not configured")
    try:
        parsed_url = urllib.parse.urlsplit(guardian_url)
    except ValueError as err:
        raise GuardianVerificationError(
            f"{GUARDIAN_URL_ENV} is not a valid URL"
        ) from err
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise GuardianVerificationError(f"{GUARDIAN_URL_ENV} must be an HTTP(S) URL")
    try:
        timeout = float(os.getenv(GUARDIAN_TIMEOUT_ENV, str(DEFAULT_GUARDIAN_TIMEOUT)))
    except ValueError as err:
        raise GuardianVerificationError(
            f"{GUARDIAN_TIMEOUT_ENV} must be a number"
        ) from err
    if timeout <= 0:
        raise GuardianVerificationError(f"{GUARDIAN_TIMEOUT_ENV} must be positive")

    try:
        body = json.dumps(
            {
                "version": GUARDIAN_PROTOCOL_VERSION,
                "capability": base64.b64encode(run.capability_package).decode("ascii"),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            guardian_url.rstrip("/") + "/v1/verify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise GuardianVerificationError(
            f"Guardian request failed: {type(err).__name__}"
        ) from err
    try:
        result = json.loads(response_body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as err:
        raise GuardianVerificationError("Guardian returned malformed JSON") from err

    if (
        not isinstance(result, dict)
        or result.get("version") != GUARDIAN_PROTOCOL_VERSION
    ):
        raise GuardianVerificationError("Guardian returned an unsupported response")
    if result.get("allowed") is not True:
        raise GuardianVerificationError("Guardian denied the capability")
    if result.get("binding") != run.capability_binding:
        raise GuardianVerificationError("Guardian returned a mismatched job binding")
