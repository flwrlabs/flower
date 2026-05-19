# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower IP address utils."""


import socket
from ipaddress import ip_address

IPV6: int = 6


def parse_address(address: str) -> tuple[str, int, bool | None] | None:
    """Parse an IP address into host, port, and version.

    Parameters
    ----------
    address : str
        The string representation of a domain, an IPv4, or an IPV6 address
        with the port number.

        For example, '127.0.0.1:8080', or [::1]:8080.

    Returns
    -------
    Optional[Tuple[str, int, Optional[bool]]]
        If the port provided is incorrect,
        the function will return None, otherwise it will return the host,
        (str), port number (int), and version (bool).
    """
    try:
        raw_host, _, raw_port = address.rpartition(":")

        port = int(raw_port)

        if port > 65535 or port < 0:
            raise ValueError("Port number is invalid.")

        try:
            host = raw_host.translate({ord(i): None for i in "[]"})
            version = ip_address(host).version == IPV6
        except ValueError:
            host = raw_host
            version = None

        return host, port, version

    except ValueError:
        return None


def is_port_in_use(address: str) -> bool:
    """Check if the port specified in address is in use.

    Parameters
    ----------
    address : str
        The string representation of a domain, an IPv4, or an IPV6 address
        with the port number.

        For example, '127.0.0.1:8080', or `[::1]:8080`.

    Returns
    -------
    bool
        If the port provided is in use or can't be parsed,
        the function will return True, otherwise it will return False.
    """
    parsed_address = parse_address(address)
    if not parsed_address:
        return True
    host, port, is_v6 = parsed_address

    if port == 0:  # OS will assign an unused port, no need to check for availability
        return False

    if is_v6:
        protocol = socket.AF_INET6
    else:
        protocol = socket.AF_INET

    with socket.socket(protocol, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            if is_v6:
                # For IPv6, provide `flowinfo` and `scopeid` as 0
                s.bind((host, port, 0, 0))
            else:
                # For IPv4
                s.bind((host, port))
        except OSError:
            return True

        return False


def resolve_bind_address(address: str) -> str:
    """Replace bind-all addresses (0.0.0.0, ::) with localhost (127.0.0.1, ::1)."""
    if address.startswith("[::]"):
        return address.replace("[::]", "[::1]", 1)
    if address.startswith("0.0.0.0"):
        return address.replace("0.0.0.0", "127.0.0.1", 1)
    return address
