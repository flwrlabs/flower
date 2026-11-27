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
"""Tests for protobuf Swagger UI integration."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from .docs import add_protobuf_docs


def test_docs_registers_plugin_and_preserves_swagger_configuration() -> None:
    """Serve the plugin with configured Swagger UI options and proxy paths."""
    app = FastAPI(
        title="Test API",
        docs_url=None,
        root_path="/proxy",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
        swagger_ui_init_oauth={"clientId": "test-client"},
    )
    add_protobuf_docs(app)
    client = TestClient(app)

    response = client.get("/docs")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    html = response.text
    assert "Test API - Swagger UI" in html
    assert html.count("wrapComponents:") == 1
    assert html.index("plugins:") < html.index("presets:")
    assert '"url": "/proxy/openapi.json"' in html
    assert '"defaultModelsExpandDepth": -1' in html
    assert "window.location.origin + '/proxy/docs/oauth2-redirect'" in html
    assert 'ui.initOAuth({"clientId": "test-client"})' in html
    assert client.get("/docs/oauth2-redirect").status_code == 200
    assert "/docs" not in app.openapi()["paths"]


def test_docs_escapes_title_and_proxy_prefix() -> None:
    """Keep title and proxy prefixes from becoming executable page content."""
    app = FastAPI(
        title="<script>alert(1)</script>",
        root_path="/</script><script>alert(2)</script>",
        docs_url=None,
    )
    add_protobuf_docs(app)

    html = TestClient(app).get("/docs").text

    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(2)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_docs_without_oauth_redirect() -> None:
    """Honor applications that disable the OAuth redirect route."""
    app = FastAPI(docs_url=None, swagger_ui_oauth2_redirect_url=None)
    add_protobuf_docs(app)
    client = TestClient(app)

    assert "oauth2RedirectUrl" not in client.get("/docs").text
    assert client.get("/docs/oauth2-redirect").status_code == 404
