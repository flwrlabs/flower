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
"""Helper functions to load objects from a reference."""


import ast
import importlib
import sys
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any

OBJECT_REF_HELP_STR = """
\n\nThe object reference string should have the form <module>:<attribute>. Valid
examples include `client:app` and `project.package.module:wrapper.app`. It must
refer to a module on the PYTHONPATH and the module needs to have the specified
attribute.
"""


_import_lock = Lock()


def validate(
    module_attribute_str: str,
    check_module: bool = True,
    project_dir: str | Path | None = None,
) -> tuple[bool, str | None]:
    """Validate object reference.

    Parameters
    ----------
    module_attribute_str : str
        The reference to the object. It should have the form `<module>:<attribute>`.
        Valid examples include `client:app` and `project.package.module:wrapper.app`.
        It must refer to a module on the PYTHONPATH or in the provided `project_dir`
        and the module needs to have the specified attribute.
    check_module : bool (default: True)
        Flag indicating whether to verify the existence of the module and the
        specified attribute within it.
    project_dir : Optional[Union[str, Path]] (default: None)
        The directory containing the module. If None, the current working directory
        is used. If `check_module` is True, the `project_dir` will be temporarily
        inserted into the system path and then removed after the validation is complete.

    Returns
    -------
    Tuple[bool, Optional[str]]
        A boolean indicating whether an object reference is valid and
        the reason why it might not be.

    Note
    ----
    This function will temporarily modify `sys.path` by inserting the provided
    `project_dir`, which will be removed after the validation is complete.
    """
    module_str, _, attributes_str = module_attribute_str.partition(":")
    if not module_str:
        return (
            False,
            f"Missing module in {module_attribute_str}{OBJECT_REF_HELP_STR}",
        )
    if not attributes_str:
        return (
            False,
            f"Missing attribute in {module_attribute_str}{OBJECT_REF_HELP_STR}",
        )

    if not check_module:
        return (True, None)

    if project_dir is None:
        project_dir = Path.cwd()
    project_dir = Path(project_dir).absolute()

    # Temporarily include the app directory for module lookup without importing it.
    sys.path.insert(0, str(project_dir))
    try:
        module = find_spec(module_str)
    finally:
        sys.path.remove(str(project_dir))

    if module and module.origin:
        # For dotted attributes (e.g. "wrapper.app"), static AST analysis cannot
        # reliably verify the full chain without importing the module, so only
        # the first segment is checked against the module's top-level names.
        root_attribute = attributes_str.split(".")[0]
        if not _find_attribute_in_module(module.origin, root_attribute):
            return (
                False,
                f"Unable to find attribute {attributes_str} in module {module_str}"
                f"{OBJECT_REF_HELP_STR}",
            )
        return (True, None)

    return (
        False,
        f"Unable to load module {module_str}{OBJECT_REF_HELP_STR}",
    )


def load_app(  # pylint: disable= too-many-branches
    module_attribute_str: str,
    error_type: type[Exception],
    project_dir: str | Path | None = None,
) -> Any:
    """Return the object specified in a module attribute string.

    Parameters
    ----------
    module_attribute_str : str
        The reference to the object. It should have the form `<module>:<attribute>`.
        Valid examples include `client:app` and `project.package.module:wrapper.app`.
        It must refer to a module on the PYTHONPATH or in the provided `project_dir`
        and the module needs to have the specified attribute.
    error_type : Type[Exception]
        The type of exception to be raised if the provided `module_attribute_str` is
        in an invalid format.
    project_dir : Optional[Union[str, Path]], optional (default=None)
        The directory containing the module. If None, the current working directory
        is used. The `project_dir` will be inserted into the system path.

    Returns
    -------
    Any
        The object specified by the module attribute string.

    Note
    ----
    This function will modify `sys.path` by inserting the provided `project_dir`.
    """
    with _import_lock:
        valid, error_msg = validate(module_attribute_str, check_module=False)
        if not valid and error_msg:
            raise error_type(error_msg) from None

        module_str, _, attributes_str = module_attribute_str.partition(":")

        if project_dir is None:
            project_dir = Path.cwd()
        project_dir = Path(project_dir).absolute()

        _ensure_sys_path(project_dir)

        if module_str not in sys.modules:
            module = importlib.import_module(module_str)
        else:
            module = sys.modules[module_str]

        # Recursively load attribute
        attribute = module
        try:
            for attribute_str in attributes_str.split("."):
                attribute = getattr(attribute, attribute_str)
        except AttributeError as err:
            raise error_type(
                f"Unable to load attribute {attributes_str} from module {module_str}"
                f"{OBJECT_REF_HELP_STR}",
            ) from err

        return attribute


def _ensure_sys_path(directory: str | Path) -> None:
    """Ensure the directory is available on `sys.path`."""
    directory = Path(directory).absolute()
    if str(directory) in sys.path:
        return
    sys.path.insert(0, str(directory))


def _find_attribute_in_module(file_path: str, attribute_name: str) -> bool:
    """Check if attribute_name exists in module's abstract symbolic tree."""
    with open(file_path, encoding="utf-8") as file:
        node = ast.parse(file.read(), filename=file_path)

    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id == attribute_name:
                    return True
                if _is_module_in_all(attribute_name, target, n):
                    return True
    return False


def _is_module_in_all(attribute_name: str, target: ast.expr, n: ast.Assign) -> bool:
    """Now check if attribute_name is in __all__."""
    if isinstance(target, ast.Name) and target.id == "__all__":
        if isinstance(n.value, ast.List):
            for elt in n.value.elts:
                if _is_string_constant(elt, attribute_name):
                    return True
        elif isinstance(n.value, ast.Tuple):
            for elt in n.value.elts:
                if _is_string_constant(elt, attribute_name):
                    return True
    return False


def _is_string_constant(node: ast.expr, value: str) -> bool:
    """Return True if node is a string constant matching value."""
    return isinstance(node, ast.Constant) and node.value == value
