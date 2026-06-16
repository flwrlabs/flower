##############################
 Install development versions
##############################

****************************************
 Install development versions of Flower
****************************************

Using uv (recommended)
======================

Install a ``flwr`` pre-release from PyPI with ``uv add``, which updates
``pyproject.toml`` and syncs the environment:

- ``uv add --prerelease=allow "flwr==1.0.0a0"`` (without extras)
- ``uv add --prerelease=allow "flwr[simulation]==1.0.0a0"`` (with extras)

Install ``flwr`` from a local copy of the Flower source code:

- ``uv add --editable "../../"`` (without extras)
- ``uv add --editable "../../" --extra simulation`` (with extras)

Install ``flwr`` from a local wheel file:

- ``uv add "../../dist/flwr-1.8.0-py3-none-any.whl"`` (without extras)
- ``uv add "../../dist/flwr-1.8.0-py3-none-any.whl" --extra simulation`` (with extras)

Please refer to the uv documentation for further details: `Managing dependencies with uv
<https://docs.astral.sh/uv/concepts/projects/dependencies/>`_

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
