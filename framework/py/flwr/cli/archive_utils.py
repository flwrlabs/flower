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
"""Archive helpers for CLI commands."""

import zipfile
from pathlib import Path

import click


def safe_extract_zip(
    zf: zipfile.ZipFile,
    dest_dir: Path,
) -> None:
    """Extract ZIP contents safely into the destination directory.

    This prevents path traversal (zip-slip) by validating that each member path resolves
    within ``dest_dir`` before extraction.
    """
    base_dir = dest_dir.resolve()

    for member in zf.infolist():
        target = (base_dir / member.filename).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            raise click.ClickException(
                f"Unsafe path in FAB archive: {member.filename}"
            ) from None

    zf.extractall(base_dir)
