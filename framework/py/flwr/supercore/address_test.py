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


import pytest

from .address import parse_address, resolve_bind_address


@pytest.mark.parametrize(
    "address, expected",
    [
        ("127.0.0.1:8080", ("127.0.0.1", 8080, False)),
        ("0.0.0.0:12", ("0.0.0.0", 12, False)),
        ("0.0.0.0:65535", ("0.0.0.0", 65535, False)),
        ("0.0.0.0:0", ("0.0.0.0", 0, False)),
    ],
)
def test_ipv4_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct IPv4 address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",  # Missing port
        "42.1.1.0:9988898",  # Port number out of range
        "0.0.0.0:-999999",  # Negative port number
        "0.0.0.0:-1",  # Negative port number
        "0.0.0.0:65536",  # Port number out of range
    ],
)
def test_ipv4_incorrect(address: str) -> None:
    """Test if an incorrect IPv4 address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


@pytest.mark.parametrize(
    "address, expected",
    [
        ("[::1]:8080", ("::1", 8080, True)),
        ("[::]:12", ("::", 12, True)),
        (
            "2001:db8:3333:4444:5555:6666:7777:8888:12",
            ("2001:db8:3333:4444:5555:6666:7777:8888", 12, True),
        ),
        (
            "2001:db8:3333:4444:5555:6666:7777:8888:65535",
            ("2001:db8:3333:4444:5555:6666:7777:8888", 65535, True),
        ),
        (
            "[0000:0000:0000:0000:0000:0000:0000:0001]:443",
            ("0000:0000:0000:0000:0000:0000:0000:0001", 443, True),
        ),
        ("[::]:123", ("::", 123, True)),
        ("[0:0:0:0:0:0:0:1]:80", ("0:0:0:0:0:0:0:1", 80, True)),
        ("[::1]:80", ("::1", 80, True)),
        (
            "[2001:db8:3333:4444:5555:6666:7777:8888]:0",
            ("2001:db8:3333:4444:5555:6666:7777:8888", 0, True),
        ),
    ],
)
def test_ipv6_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct IPv6 address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "[2001:db8:3333:4444:5555:6666:7777:8888]:9988898",  # Port number out of range
        "[2001:db8:3333:4444:5555:6666:7777:8888]:-9988898",  # Negative port number
        "[2001:db8:3333:4444:5555:6666:7777:8888]:-1",  # Negative port number
        "[2001:db8:3333:4444:5555:6666:7777:8888]:65536",  # Port number out of range
    ],
)
def test_ipv6_incorrect(address: str) -> None:
    """Test if an incorrect IPv6 address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


@pytest.mark.parametrize(
    "address, expected",
    [
        ("flower.ai:123", ("flower.ai", 123, None)),
        ("sub.flower.ai:123", ("sub.flower.ai", 123, None)),
        ("sub2.sub1.flower.ai:123", ("sub2.sub1.flower.ai", 123, None)),
        ("s5.s4.s3.s2.s1.flower.ai:123", ("s5.s4.s3.s2.s1.flower.ai", 123, None)),
        ("localhost:123", ("localhost", 123, None)),
        ("https://localhost:123", ("https://localhost", 123, None)),
        ("http://localhost:123", ("http://localhost", 123, None)),
    ],
)
def test_domain_correct(address: str, expected: tuple[str, int, bool]) -> None:
    """Test if a correct domain address is correctly parsed."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual == expected


@pytest.mark.parametrize(
    "address",
    [
        "flower.ai",  # Missing port
        "flower.ai:65536",  # Port number out of range
    ],
)
def test_domain_incorrect(address: str) -> None:
    """Test if an incorrect domain address returns None."""
    # Execute
    actual = parse_address(address)

    # Assert
    assert actual is None


@pytest.mark.parametrize(
    "address, expected",
    [
        ("0.0.0.0:8080", "127.0.0.1:8080"),
        ("[::]:8080", "[::1]:8080"),
        ("192.168.1.1:8080", "192.168.1.1:8080"),
        ("[2001:db8::1]:8080", "[2001:db8::1]:8080"),
        ("localhost:8080", "localhost:8080"),
        ("example.com:443", "example.com:443"),
    ],
)
def test_resolve_bind_address(address: str, expected: str) -> None:
    """Test conversion of bind-all addresses to localhost."""
    assert resolve_bind_address(address) == expected
