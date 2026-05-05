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
"""Concrete API error types."""


import json

from .base import ApiErrorCode, FlowerError


class EntitlementError(FlowerError):
    """API error for actions blocked by entitlement checks.

    Parameters
    ----------
    details : str
        Client-visible explanation of why the entitlement check failed.
    entitlement_code : int
        Service-defined entitlement code included in the serialized payload
        for programmatic handling.
    """

    def __init__(self, details: str, entitlement_code: int):
        super().__init__(
            message=details,
            public_details=details,
            code=ApiErrorCode.ENTITLEMENT_ERROR,
        )
        self.entitlement_code = entitlement_code

    def to_json(self, public_message: str) -> str:
        """Serialize the entitlement error payload as JSON.

        Parameters
        ----------
        public_message : str
            Sanitized message that should be exposed to the client together with
            the entitlement-specific details.

        Returns
        -------
        str
            A JSON string containing the base client-visible error fields plus
            the entitlement code used for programmatic handling on the client.
        """
        base_dict = json.loads(super().to_json(public_message))
        base_dict["entitlement_code"] = self.entitlement_code
        return json.dumps(base_dict)
