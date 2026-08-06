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
"""Scaffold connector providers and generate their static registry."""

from __future__ import annotations

import argparse
from pathlib import Path

_FRAMEWORK_DIR = Path(__file__).resolve().parents[1]
_PROVIDERS_DIR = _FRAMEWORK_DIR / "py/flwr/supercore/task_process/connector/providers"
_REGISTRY_PATH = _PROVIDERS_DIR / "registry_generated.py"
_PACKAGE_PREFIX = "flwr.supercore.task_process.connector.providers"
_LICENSE = """# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""


def main() -> None:
    """Scaffold an optional provider and update its generated registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", nargs="?", help="Lowercase snake-case provider ID")
    parser.add_argument("--display-name", help="Provider name shown to users")
    parser.add_argument(
        "--check", action="store_true", help="Fail if the registry is stale"
    )
    args = parser.parse_args()
    if args.check and args.provider:
        parser.error("--check cannot be combined with a provider")
    if args.provider:
        scaffold_provider(args.provider, args.display_name)
    update_registry(check=args.check)


def scaffold_provider(provider: str, display_name: str | None = None) -> None:
    """Create a minimal provider package without overwriting existing files."""
    _validate_identifier(provider)
    target = _PROVIDERS_DIR / provider
    if target.exists():
        raise FileExistsError(f"Connector provider already exists: {provider}")
    target.mkdir()
    shown_name = display_name or provider.replace("_", " ").title()
    for filename, content in _provider_templates(provider, shown_name).items():
        (target / filename).write_text(content, encoding="utf-8")
    print(f"Created connector provider: {target.relative_to(_FRAMEWORK_DIR)}")


def update_registry(*, check: bool = False) -> None:
    """Write or verify the deterministic provider package registry."""
    content = render_registry(_provider_ids())
    current = _REGISTRY_PATH.read_text(encoding="utf-8")
    if check:
        if current != content:
            raise RuntimeError(
                "Connector provider registry is stale; run "
                "`python -m dev.generate_connector`."
            )
        return
    if current != content:
        _REGISTRY_PATH.write_text(content, encoding="utf-8")
        print(f"Updated {_REGISTRY_PATH.relative_to(_FRAMEWORK_DIR)}")


def render_registry(providers: list[str]) -> str:
    """Render the generated Python provider registry."""
    packages = "\n".join(
        f'    "{_PACKAGE_PREFIX}.{provider}",' for provider in providers
    )
    if packages:
        value = f"(\n{packages}\n)"
    else:
        value = "()"
    return (
        _LICENSE
        + '"""Generated connector provider package registry. Do not edit."""\n\n'
        + f"PROVIDER_PACKAGES: tuple[str, ...] = {value}\n"
    )


def _provider_ids() -> list[str]:
    """Return provider package directory names in deterministic order."""
    providers = [
        path.name
        for path in _PROVIDERS_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    ]
    for provider in providers:
        _validate_identifier(provider)
        required = {"actions.py", "definition.py", "executors.py"}
        missing = required.difference(
            path.name for path in (_PROVIDERS_DIR / provider).iterdir()
        )
        if missing:
            raise RuntimeError(
                f"Provider '{provider}' is missing: {', '.join(sorted(missing))}."
            )
    return sorted(providers)


def _validate_identifier(provider: str) -> None:
    """Require the same provider identifier accepted by definitions."""
    if not provider or not provider.isidentifier() or provider.lower() != provider:
        raise ValueError("Provider ID must be a lowercase snake-case identifier.")


def _provider_templates(provider: str, display_name: str) -> dict[str, str]:
    """Return minimal source files for one standard OAuth provider."""
    env_prefix = provider.upper()
    init = _LICENSE + f'"""{display_name} connector provider."""\n'
    actions = (
        _LICENSE
        + f'''"""{display_name} action definitions."""

from ...definition import ActionAccess, ActionDefinition

READ = ActionDefinition(
    name="read",
    description="Read resources from {display_name}.",
    access=ActionAccess.READ,
    input_schema={{
        "type": "object",
        "properties": {{}},
        "additionalProperties": False,
    }},
)

ACTIONS = (READ,)
'''
    )
    definition = (
        _LICENSE
        + f'''"""{display_name} provider definition."""

from ...definition import OAuth2Definition, ProviderDefinition
from .actions import ACTIONS

PROVIDER = ProviderDefinition(
    ref="{provider}",
    display_name="{display_name}",
    description="Connect to {display_name}.",
    actions=ACTIONS,
    oauth=OAuth2Definition(
        authorization_url="https://provider.example/oauth/authorize",
        token_url="https://provider.example/oauth/token",
        client_id_env="FLWR_{env_prefix}_CLIENT_ID",
        client_secret_env="FLWR_{env_prefix}_CLIENT_SECRET",
        redirect_uri_env="FLWR_{env_prefix}_REDIRECT_URI",
    ),
    api_base_url="https://api.provider.example/v1",
)
'''
    )
    executors = (
        _LICENSE
        + f'''"""{display_name} action executors."""

from flwr.supercore.typing import JSONObject

from ...runtime import ConnectorContext, ConnectorExecutor


def read(arguments: JSONObject, context: ConnectorContext) -> JSONObject:
    """Read resources from {display_name}."""
    del arguments
    if context.http is None:
        raise RuntimeError("{display_name} HTTP client is not configured.")
    return context.http.request("GET", "/resources")


EXECUTORS: dict[str, ConnectorExecutor] = {{"read": read}}
'''
    )
    return {
        "__init__.py": init,
        "actions.py": actions,
        "definition.py": definition,
        "executors.py": executors,
    }


if __name__ == "__main__":
    main()
