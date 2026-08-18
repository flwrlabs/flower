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
"""Runtime API HTTP client helpers."""

import ssl
from collections.abc import Sequence
from typing import TypeVar

from flwr.supercore.protobuf.client import ProtobufClient, ProtobufClientInterceptor

RuntimeHttpStubT = TypeVar("RuntimeHttpStubT", bound=ProtobufClient)


def create_runtime_http_stub(
    stub_class: type[RuntimeHttpStubT],
    runtime_api_address: str,
    insecure: bool,
    root_certificates: bytes | str | None,
    interceptors: Sequence[ProtobufClientInterceptor],
) -> RuntimeHttpStubT:
    """Create a protobuf-over-HTTP Runtime API stub."""
    scheme = "http" if insecure else "https"
    verify: ssl.SSLContext | str | bool = not insecure
    if not insecure and root_certificates is not None:
        if isinstance(root_certificates, str):
            verify = root_certificates
        else:
            verify = ssl.create_default_context()
            verify.load_verify_locations(cadata=root_certificates.decode("ascii"))

    return stub_class(
        f"{scheme}://{runtime_api_address}",
        interceptors=interceptors,
        verify=verify,
    )
