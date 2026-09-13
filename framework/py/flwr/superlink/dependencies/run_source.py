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
"""FastAPI dependency for Control API run-source attribution."""

from typing import Annotated

from fastapi import Depends, Header

from flwr.supercore.constant import FLWR_CLIENT_METADATA_KEY
from flwr.superlink.run_source import RunSource, resolve_source


def get_run_source(
    run_source: Annotated[str | None, Header(alias=FLWR_CLIENT_METADATA_KEY)] = None,
) -> RunSource:
    """Return the normalized run source from the client metadata header."""
    return resolve_source(run_source)


RunSourceDependency = Annotated[RunSource, Depends(get_run_source)]
