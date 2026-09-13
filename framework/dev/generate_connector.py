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
"""Scaffold connector packages and generate their static registry."""

from __future__ import annotations

import argparse
from pathlib import Path

_FRAMEWORK_DIR = Path(__file__).resolve().parents[1]
_CONNECTOR_DIR = _FRAMEWORK_DIR / "py/flwr/supercore/task_process/connector"
_REGISTRY_PATH = _CONNECTOR_DIR / "registry_generated.py"
_PACKAGE_PREFIX = "flwr.supercore.task_process.connector"
_REQUIRED_MODULES = {"actions.py", "definition.py", "executors.py"}
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
    """Scaffold an optional connector and update its generated registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connector", nargs="?", help="Lowercase connector ID")
    parser.add_argument("--display-name", help="Connector name shown to users")
    parser.add_argument(
        "--check", action="store_true", help="Fail if the registry is stale"
    )
    args = parser.parse_args()
    if args.check and args.connector:
        parser.error("--check cannot be combined with a connector")
    if args.connector:
        scaffold_connector(args.connector, args.display_name)
    update_registry(check=args.check)


def scaffold_connector(connector: str, display_name: str | None = None) -> None:
    """Create a minimal connector package without overwriting existing files."""
    _validate_identifier(connector)
    target = _CONNECTOR_DIR / connector
    if target.exists():
        raise FileExistsError(f"Connector already exists: {connector}")
    target.mkdir()
    shown_name = display_name or connector.replace("_", " ").title()
    if not shown_name.strip():
        raise ValueError("Connector display name must not be empty.")
    for filename, content in _connector_templates(connector, shown_name).items():
        (target / filename).write_text(content, encoding="utf-8")
    print(f"Created connector: {target.relative_to(_FRAMEWORK_DIR)}")


def update_registry(*, check: bool = False) -> None:
    """Write or verify the deterministic connector package registry."""
    content = render_registry(_connector_ids())
    current = _REGISTRY_PATH.read_text(encoding="utf-8")
    if check:
        if current != content:
            raise RuntimeError(
                "Connector registry is stale; run `python -m dev.generate_connector`."
            )
        return
    if current != content:
        _REGISTRY_PATH.write_text(content, encoding="utf-8")
        print(f"Updated {_REGISTRY_PATH.relative_to(_FRAMEWORK_DIR)}")


def render_registry(connectors: list[str]) -> str:
    """Render the generated connector package registry."""
    packages = "\n".join(
        f'    "{_PACKAGE_PREFIX}.{connector}",' for connector in connectors
    )
    value = f"(\n{packages}\n)" if packages else "()"
    return (
        _LICENSE
        + '"""Generated connector package registry. Do not edit."""\n\n'
        + f"CONNECTOR_PACKAGES: tuple[str, ...] = {value}\n"
    )


def _connector_ids() -> list[str]:
    """Return complete connector package names in deterministic order."""
    connectors: list[str] = []
    for path in _CONNECTOR_DIR.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        filenames = {entry.name for entry in path.iterdir() if entry.is_file()}
        if not filenames.intersection(_REQUIRED_MODULES):
            continue
        _validate_identifier(path.name)
        missing = _REQUIRED_MODULES.difference(filenames)
        if missing:
            raise RuntimeError(
                f"Connector '{path.name}' is missing: {', '.join(sorted(missing))}."
            )
        connectors.append(path.name)
    return sorted(connectors)


def _validate_identifier(connector: str) -> None:
    """Require a stable lowercase snake-case connector identifier."""
    if not connector or not connector.isidentifier() or connector.lower() != connector:
        raise ValueError("Connector ID must be a lowercase snake-case identifier.")


def _connector_templates(connector: str, display_name: str) -> dict[str, str]:
    """Return the minimal files for one connector package."""
    display = repr(display_name)
    description = repr(f"Connect to {display_name}.")
    not_implemented = repr(f"Implement the {display_name} read action.")
    init = (
        _LICENSE
        + '"""Connector package."""\n\n'
        + "from .definition import CONNECTOR\n\n"
        + '__all__ = ["CONNECTOR"]\n'
    )
    actions = (
        _LICENSE
        + f'''"""Connector action definitions."""

from ..definition import ActionAccess, ActionDefinition

READ = ActionDefinition(
    name="read",
    description={description},
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
    executors = (
        _LICENSE
        + f'''"""Connector action executors."""

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor


def read(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read resources from the provider."""
    del arguments, context
    raise NotImplementedError({not_implemented})


EXECUTORS: dict[str, ConnectorExecutor] = {{"read": read}}
'''
    )
    definition = (
        _LICENSE
        + f'''"""Connector definition."""

from ..definition import ConnectorDefinition, ProviderDefinition
from .actions import ACTIONS
from .executors import EXECUTORS

PROVIDER = ProviderDefinition(
    ref={connector!r},
    display_name={display},
    description={description},
    actions=ACTIONS,
)

CONNECTOR = ConnectorDefinition(provider=PROVIDER, executors=EXECUTORS)
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
