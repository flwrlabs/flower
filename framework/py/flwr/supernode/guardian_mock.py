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
"""Local Guardian mock for the capability POC.

Run with ``python -m flwr.supernode.guardian_mock``. The mock treats the opaque
capability bytes as the binding it should return. A capability beginning with
``deny:`` produces a denial response.
"""

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flwr.common.capability import CAPABILITY_LOG_PREFIX, safe_digest_prefix

from .guardian import GUARDIAN_PROTOCOL_VERSION


class GuardianMockHandler(BaseHTTPRequestHandler):
    """Serve the v1 local Guardian verification endpoint."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Return the binding carried by a valid mock capability."""
        if self.path != "/v1/verify":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if request.get("version") != GUARDIAN_PROTOCOL_VERSION:
                raise ValueError("unsupported protocol")
            capability = base64.b64decode(request["capability"], validate=True).decode(
                "utf-8"
            )
        except (KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        denied = capability.startswith("deny:")
        presented_binding = capability.removeprefix("deny:")
        print(
            f"{CAPABILITY_LOG_PREFIX} GuardianMock verify "
            f"protocol={GUARDIAN_PROTOCOL_VERSION} "
            f"decision={'deny' if denied else 'allow'} "
            f"returned_binding={safe_digest_prefix(presented_binding)}",
            flush=True,
        )
        response = {
            "version": GUARDIAN_PROTOCOL_VERSION,
            "allowed": not denied,
            "binding": capability,
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(  # pylint: disable=redefined-builtin
        self, format: str, *args: object
    ) -> None:
        """Suppress default request logging in the tiny local mock."""


def main() -> None:
    """Run the local Guardian mock server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), GuardianMockHandler)
    print(f"Guardian mock listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
