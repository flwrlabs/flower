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
"""Deprecated grid compatibility APIs."""


from flwr.server.grid import Grid


class Driver(Grid):
    """Deprecated abstract base class ``Driver``, use ``Grid`` instead.

    This class is provided solely for backward compatibility with legacy
    code that previously relied on the ``Driver`` class. It has been deprecated
    in favor of the updated abstract base class ``Grid``, which now encompasses
    all communication-related functionality and improvements between the
    ServerApp and the SuperLink.

    .. warning::
        ``Driver`` is deprecated and will be removed in a future release.
        Use ``Grid`` in the signature of your ServerApp.

    Examples
    --------
    Legacy (deprecated) usage::

        @app.main()
        def main(driver: Driver, context: Context) -> None:
            ...

    Updated usage::

        @app.main()
        def main(grid: Grid, context: Context) -> None:
            ...
    """
