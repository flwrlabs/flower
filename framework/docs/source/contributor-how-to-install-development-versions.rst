##############################
 Install development versions
##############################

****************************************
 Install development versions of Flower
****************************************

Using Poetry (recommended)
==========================

Install a ``flwr`` pre-release from PyPI: update the ``flwr`` dependency in
``pyproject.toml`` and then reinstall (ensure to delete ``poetry.lock`` via ``rm
poetry.lock`` before running ``python -m poetry install``).

- ``flwr = { version = "1.0.0a0", allow-prereleases = true }`` (without extras)
- ``flwr = { version = "1.0.0a0", allow-prereleases = true, extras = ["simulation"] }``
  (with extras)

Install ``flwr`` from a local copy of the Flower source code via ``pyproject.toml``:

- ``flwr = { path = "../../", develop = true }`` (without extras)
- ``flwr = { path = "../../", develop = true, extras = ["simulation"] }`` (with extras)

Install ``flwr`` from a local wheel file via ``pyproject.toml``:

- ``flwr = { path = "../../dist/flwr-1.8.0-py3-none-any.whl" }`` (without extras)
- ``flwr = { path = "../../dist/flwr-1.8.0-py3-none-any.whl", extras = ["simulation"]
  }`` (with extras)

Please refer to the Poetry documentation for further details: `Poetry Dependency
Specification <https://python-poetry.org/docs/dependency-specification/>`_

Using pip (recommended on Colab)
================================

Install a ``flwr`` pre-release from PyPI:

- ``pip install -U --pre flwr`` (without extras)
- ``pip install -U --pre 'flwr[simulation]'`` (with extras)

Python packages can be installed from git repositories. Use one of the following
commands to install the Flower directly from GitHub.

Install ``flwr`` from the default GitHub branch (``main``):

- ``pip install flwr@git+https://github.com/flwrlabs/flower.git#subdirectory=framework``
  (without extras)
- ``pip install
  'flwr[simulation]@git+https://github.com/flwrlabs/flower.git#subdirectory=framework'``
  (with extras)

Install ``flwr`` from a specific GitHub branch (``branch-name``):

- ``pip install
  flwr@git+https://github.com/flwrlabs/flower.git@branch-name#subdirectory=framework``
  (without extras)
- ``pip install
  'flwr[simulation]@git+https://github.com/flwrlabs/flower.git@branch-name#subdirectory=framework'``
  (with extras)
