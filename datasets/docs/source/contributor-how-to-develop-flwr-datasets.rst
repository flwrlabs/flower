How to develop flwr-datasets
============================

Flower Datasets uses ``uv`` for development and CI. The commands presented in this guide assume you have cloned the `Flower GitHub repository <https://github.com/flwrlabs/flower>`_.


Setup
-----

.. code-block:: bash

   cd datasets
   uv sync --all-extras

.. tip::

   Use ``uv sync --frozen --all-extras`` to ensure ``uv.lock`` is not modified.

Run checks (formatting + unit tests)
------------------------------------

.. code-block:: bash

   cd datasets
   uv run ./dev/test.sh

Format
------

.. code-block:: bash

   cd datasets
   uv run ./dev/format.sh

Build docs
----------

.. code-block:: bash

   cd datasets
   uv run ./dev/build-flwr-datasets-docs.sh

Run E2E tests
-------------

.. code-block:: bash

   cd datasets/e2e/pytorch
   uv sync --frozen
   uv run python -m unittest discover -p "*_test.py"

Repeat for ``datasets/e2e/scikit-learn`` and ``datasets/e2e/tensorflow``.

Dependency management (no ``uv pip``)
-------------------------------------

.. code-block:: bash

   cd datasets

   # Add a runtime dependency
   uv add <package>

   # Add a dev dependency
   uv add --dev <package>

   # Add a dependency to an extra (e.g. "vision")
   uv add --optional vision <package>

   # Update lockfile (commit the result)
   uv lock
