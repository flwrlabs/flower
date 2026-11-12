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
"""Swagger UI support for documented binary protobuf request bodies."""

from html import escape
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from starlette.responses import HTMLResponse

# Swagger UI's binary-body renderer skips the request body description. Wrap that
# component instead of changing the binary schema or its file upload behavior.
_PROTOBUF_REQUEST_BODY_PLUGIN = """
    plugins: [() => ({
        wrapComponents: {
            RequestBody: (Original, system) => (props) => {
                const body = props.requestBody;
                const contentType = props.contentType ||
                    body?.get("content")?.keySeq().first();
                const description = body?.get("description");
                const schema = body?.getIn(["content", contentType, "schema"]);
                const React = system.React;
                if (contentType !== "application/protobuf" || !description ||
                    !props.fn.isFileUploadIntended(schema, contentType)) {
                    return React.createElement(Original, props);
                }
                const Markdown = props.getComponent("Markdown", true);
                return React.createElement("div", null,
                    React.createElement(Markdown, {source: description}),
                    props.isExecute ? React.createElement(Original, props) : null
                );
            }
        }
    })],
"""


def add_protobuf_docs(app: FastAPI) -> None:
    """Register Swagger UI with descriptions for binary protobuf request bodies.

    Configure the application with ``docs_url=None`` before calling this helper.
    """

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui(request: Request) -> HTMLResponse:
        root_path = request.scope.get("root_path", "").rstrip("/")
        oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url
        response = get_swagger_ui_html(
            # Pass the URL as an HTML-safe JSON parameter so proxy prefixes cannot
            # introduce executable HTML or JavaScript into the generated page.
            openapi_url="",
            title=escape(app.title) + " - Swagger UI",
            oauth2_redirect_url=(
                quote(root_path + oauth2_redirect_url, safe="/")
                if oauth2_redirect_url
                else None
            ),
            init_oauth=app.swagger_ui_init_oauth,
            swagger_ui_parameters={
                **(app.swagger_ui_parameters or {}),
                "url": root_path + (app.openapi_url or "/openapi.json"),
            },
        )
        # FastAPI serializes configuration values as JSON, which cannot contain
        # plugin functions. Insert only this static plugin at its presets entry.
        html = (
            bytes(response.body)
            .decode()
            .replace("presets: [", _PROTOBUF_REQUEST_BODY_PLUGIN + "presets: [", 1)
        )
        return HTMLResponse(html)

    if app.swagger_ui_oauth2_redirect_url:
        app.add_api_route(
            app.swagger_ui_oauth2_redirect_url,
            get_swagger_ui_oauth2_redirect_html,
            include_in_schema=False,
        )
