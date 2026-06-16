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
"""Helpers for import errors raised while running apps."""


def format_app_import_error_message(
    app_name: str, exc: ImportError, runtime_dependency_install: bool
) -> str:
    """Return user guidance for an app import error."""
    import_error_details = str(exc)
    if runtime_dependency_install:
        guidance = (
            "Automatic runtime dependency installation is enabled. Add the missing "
            "package to the app's `pyproject.toml` dependencies and run the app again."
        )
    else:
        guidance = (
            f"Ensure the Python environment where the {app_name} runs has all "
            "required dependencies installed. Alternatively, enable automatic runtime "
            "dependency installation. See "
            "https://flower.ai/docs/framework/how-to-install-app-dependencies-at-runtime.html"
            " for details."
        )
    return f"{app_name} failed to import a module: {import_error_details}. {guidance}"
