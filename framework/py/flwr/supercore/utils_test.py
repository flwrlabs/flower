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
"""Tests for utility functions for the infrastructure."""


import json
from typing import Any

import pytest
import requests
from parameterized import parameterized

from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611

from .utils import (
    get_metadata_bytes,
    get_metadata_str,
    humanize_bytes,
    humanize_duration,
    int64_to_uint64,
    mask_string,
    parse_app_spec,
    request_download_link,
    simulation_config_from_json,
    simulation_config_to_json,
    uint64_to_int64,
)


def test_mask_string() -> None:
    """Test the `mask_string` function."""
    assert mask_string("abcdefghi") == "abcd...fghi"
    assert mask_string("abcdefghijklm") == "abcd...jklm"
    assert mask_string("abc") == "abc"
    assert mask_string("a") == "a"
    assert mask_string("") == ""
    assert mask_string("1234567890", head=2, tail=3) == "12...890"
    assert mask_string("1234567890", head=5, tail=4) == "12345...7890"


@pytest.mark.parametrize(
    ("metadata", "key", "expected"),
    [
        ([("x-token", "value")], "x-token", "value"),
        (
            [
                ("x-token", ""),
            ],
            "x-token",
            None,
        ),
        ([("x-token", "value"), ("x-token", "other")], "x-token", None),
        ([("x-token", b"value")], "x-token", None),
    ],
)
def test_get_metadata_str(
    metadata: list[tuple[str, str | bytes]], key: str, expected: str | None
) -> None:
    """Return exactly one non-empty string value of the expected type."""
    assert get_metadata_str(metadata, key) == expected


@pytest.mark.parametrize(
    ("metadata", "key", "expected"),
    [
        ([("x-bin", b"value")], "x-bin", b"value"),
        ([("x-bin", b"")], "x-bin", None),
        ([("x-bin", b"value"), ("x-bin", b"other")], "x-bin", None),
        ([("x-bin", "value")], "x-bin", None),
    ],
)
def test_get_metadata_bytes(
    metadata: list[tuple[str, str | bytes]], key: str, expected: bytes | None
) -> None:
    """Return exactly one non-empty bytes value of the expected type."""
    assert get_metadata_bytes(metadata, key) == expected


@parameterized.expand(  # type: ignore
    [
        # Test values within the positive range of sint64 (below 2^63)
        (0, 0),  # Minimum positive value
        (1, 1),  # 1 remains 1 in both uint64 and sint64
        (2**62, 2**62),  # Mid-range positive value
        (2**63 - 1, 2**63 - 1),  # Maximum positive value for sint64
        # Test values at or above 2^63 (become negative in sint64)
        (2**63, -(2**63)),  # Minimum negative value for sint64
        (2**63 + 1, -(2**63) + 1),  # Slightly above the boundary
        (9223372036854775811, -9223372036854775805),  # Some value > sint64 max
        (2**64 - 1, -1),  # Maximum uint64 value becomes -1 in sint64
    ]
)
def test_convert_uint64_to_sint64(before: int, after: int) -> None:
    """Test conversion from uint64 to sint64."""
    assert uint64_to_int64(before) == after


@parameterized.expand(  # type: ignore
    [
        # Test values within the negative range of sint64
        (-(2**63), 2**63),  # Minimum sint64 value becomes 2^63 in uint64
        (-(2**63) + 1, 2**63 + 1),  # Slightly above the minimum
        (-9223372036854775805, 9223372036854775811),  # Some value > sint64 max
        # Test zero-adjacent inputs
        (-1, 2**64 - 1),  # -1 in sint64 becomes 2^64 - 1 in uint64
        (0, 0),  # 0 remains 0 in both sint64 and uint64
        (1, 1),  # 1 remains 1 in both sint64 and uint64
        # Test values within the positive range of sint64
        (2**63 - 1, 2**63 - 1),  # Maximum positive value in sint64
        # Test boundary and maximum uint64 value
        (2**63, 2**63),  # Exact boundary value for sint64
        (2**64 - 1, 2**64 - 1),  # Maximum uint64 value, stays the same
    ]
)
def test_sint64_to_uint64(before: int, after: int) -> None:
    """Test conversion from sint64 to uint64."""
    assert int64_to_uint64(before) == after


@parameterized.expand(  # type: ignore
    [
        (0),
        (1),
        (2**62),
        (2**63 - 1),
        (2**63),
        (2**63 + 1),
        (9223372036854775811),
        (2**64 - 1),
    ]
)
def test_uint64_to_sint64_to_uint64(expected: int) -> None:
    """Test conversion from sint64 to uint64."""
    actual = int64_to_uint64(uint64_to_int64(expected))
    assert actual == expected


@pytest.mark.parametrize(
    "value",
    [
        "user/app==1.2.3",  # missing '@'
        "@accountapp==1.2.3",  # missing slash
        "@account/app==1.2",  # bad version
        "@account/app==1.2.3.4",  # bad version
        "@account*/app==1.2.3",  # bad user id chars
        "@account/app*==1.2.3",  # bad app id chars
    ],
)
def test_parse_app_spec_rejects_invalid_formats(value: str) -> None:
    """For an invalid string, the function should fail fast with ValueError."""
    with pytest.raises(ValueError):
        parse_app_spec(value)


def test_request_download_link_all_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single table-driven test covering all major outcomes for
    request_download_link."""
    app_id: str = "@user/app"
    app_version: str | None = "1.0.0"
    in_url = "https://example.ai/hub/fetch-fab"
    out_url = "fab_url"

    # Table of scenarios
    scenarios: list[dict[str, Any]] = [
        {
            "name": "success_with_verifications",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {
                    "fab_url": "https://example.ai/fab.fab",
                    "verifications": [
                        {"public_key_id": "key1", "sig": "abc", "algo": "ed25519"}
                    ],
                    "note": "Compatibility fallback applied.",
                },
            },
            "assert": lambda out: (
                out[0] == "https://example.ai/fab.fab"
                and isinstance(out[1], list)
                and out[1][0]["public_key_id"] == "key1"
                and out[2] == "Compatibility fallback applied."
            ),
        },
        {
            "name": "success_without_verifications",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {"fab_url": "https://example.ai/fab.fab"},
            },
            "assert": lambda out: out[0] == "https://example.ai/fab.fab"
            and out[1] is None
            and out[2] is None,
        },
        {
            "name": "http_404_not_found",
            "fake_resp": {
                "ok": False,
                "status": 404,
                "text": "not found",
                # No JSON body -> json() returns {}
            },
            "raises": "not found in Platform API",
        },
        {
            "name": "http_503_unavailable",
            "fake_resp": {
                "ok": False,
                "status": 503,
                "text": "service unavailable",
            },
            "raises": "status 503",
        },
        {
            "name": "network_error",
            "fake_exc": requests.RequestException("network down"),
            "raises": "Unable to connect to Platform API",
        },
        {
            "name": "missing_fab_url",
            "fake_resp": {
                "ok": True,
                "status": 200,
                "json": {"verifications": []},
            },
            "raises": "Invalid response from Platform API",
        },
    ]

    current_case: dict[str, Any] = {"data": None}

    class _FakeResp:
        ok: bool
        status_code: int
        _json: dict[str, Any]
        text: str

        def __init__(
            self,
            ok: bool,
            status: int,
            json_data: dict[str, Any] | None = None,
            text: str = "",
        ) -> None:
            self.ok = ok
            self.status_code = status
            self._json = json_data or {}
            self.text = text

        def json(self) -> dict[str, Any]:
            """Return JSON data."""
            return self._json

    def fake_post(url: str, data: str | None = None, **_: Any) -> _FakeResp:
        case_data: dict[str, Any] | None = current_case.get("data")

        # Basic payload sanity check for the success-like cases
        if isinstance(case_data, dict) and "fake_resp" in case_data:
            assert url == in_url
            assert data is not None
            payload: dict[str, Any] = json.loads(data)
            assert payload["app_id"] == app_id
            assert payload["app_version"] == app_version
            assert "flwr_version" in payload

        # Simulate network error if requested
        if isinstance(case_data, dict) and "fake_exc" in case_data:
            raise case_data["fake_exc"]

        fr: dict[str, Any] = case_data["fake_resp"]  # type: ignore[index]
        return _FakeResp(
            ok=fr["ok"],
            status=fr["status"],
            json_data=fr.get("json"),
            text=fr.get("text", ""),
        )

    monkeypatch.setattr(requests, "post", fake_post)

    for case in scenarios:
        current_case["data"] = case

        if "raises" in case:
            with pytest.raises(ValueError) as exc:
                _ = request_download_link(app_id, app_version, in_url, out_url)
            msg: str = str(exc.value)
            assert case["raises"] in msg, f"Expected '{case['raises']}' in '{msg}'"
            if case["name"] == "http_404_not_found":
                assert app_id in msg
        else:
            # Expect a (fab_url, verifications, note) tuple
            result: tuple[str, list[dict[str, str]] | None, str | None] = (
                request_download_link(app_id, app_version, in_url, out_url)
            )
            assert case["assert"](result), f"Assertion failed for {case['name']}"


def test_simulation_config_to_json_includes_unset_optional_fields() -> None:
    """Serialize simulation config with unset optional fields as null."""
    config = SimulationConfig(
        num_supernodes=10,
        client_resources_num_cpus=2,
        client_resources_num_gpus=0.25,
        backend="ray",
        verbose=True,
    )

    assert simulation_config_to_json(config) == {
        "num_supernodes": 10,
        "client_resources_num_cpus": 2,
        "client_resources_num_gpus": 0.25,
        "backend": "ray",
        "verbose": True,
        "init_args_num_cpus": None,
        "init_args_num_gpus": None,
        "init_args_logging_level": None,
        "init_args_log_to_driver": None,
    }


def test_simulation_config_from_json_preserves_falsey_optional_values() -> None:
    """Deserialize optional values without dropping explicit zero/false settings."""
    payload = {
        "num_supernodes": 5,
        "client_resources_num_cpus": 1,
        "client_resources_num_gpus": 0.0,
        "backend": "ray",
        "verbose": False,
        "init_args_num_cpus": 4,
        "init_args_num_gpus": 0,
        "init_args_logging_level": "",
        "init_args_log_to_driver": False,
    }

    config = simulation_config_from_json(payload)

    assert config.num_supernodes == 5
    assert config.client_resources_num_cpus == 1
    assert config.client_resources_num_gpus == 0.0
    assert config.backend == "ray"
    assert not config.verbose
    assert config.init_args_num_cpus == 4
    assert config.init_args_num_gpus == 0
    assert config.init_args_logging_level == ""
    assert not config.init_args_log_to_driver
    assert config.HasField("init_args_num_cpus")
    assert config.HasField("init_args_num_gpus")
    assert config.HasField("init_args_logging_level")
    assert config.HasField("init_args_log_to_driver")


def test_simulation_config_json_round_trip_preserves_presence() -> None:
    """Round-trip simulation config JSON without inventing optional fields."""
    original = SimulationConfig(
        num_supernodes=12,
        client_resources_num_cpus=3,
        client_resources_num_gpus=1.5,
        backend="ray",
        verbose=False,
        init_args_num_gpus=2,
    )

    restored = simulation_config_from_json(simulation_config_to_json(original))

    assert restored == original
    assert not restored.HasField("init_args_num_cpus")
    assert restored.HasField("init_args_num_gpus")
    assert not restored.HasField("init_args_logging_level")
    assert not restored.HasField("init_args_log_to_driver")


def test_simulation_config_from_json_rejects_unknown_fields() -> None:
    """Reject unknown JSON keys when deserializing simulation configs."""
    with pytest.raises(ValueError, match="unknown_field"):
        simulation_config_from_json({"unknown_field": 1})


@parameterized.expand(  # type: ignore
    [
        (24, "24s"),  # seconds
        (90, "1m 30s"),  # min + sec
        (3723, "1h 2m"),  # hour + min
        (90000, "1d 1h"),  # day + hour
    ]
)
def test_humanize_duration(seconds, expected) -> None:
    """Test the humanize_duration function."""
    assert humanize_duration(seconds) == expected


@parameterized.expand(  # type: ignore
    [
        (800, "800 B"),  # bytes
        (2048, "2.0 KB"),  # KB < 10 → 1 decimal
        (10 * 1024, "10 KB"),  # KB >= 10 → no decimal
        (5 * 1024**2, "5.0 MB"),  # MB < 10
        (3 * 1024**3, "3.0 GB"),  # GB < 10
    ]
)
def test_humanize_bytes(num_bytes, expected) -> None:
    """Test the humanize_bytes function."""
    assert humanize_bytes(num_bytes) == expected
