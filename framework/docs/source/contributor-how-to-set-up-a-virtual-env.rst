######################
 Set up a virtual env
######################

It is recommended to run your Python setup within a virtual environment. For framework
development, uv creates and manages the virtual environment from the project lockfile.
Tools such as pyenv or Anaconda can still provide the Python interpreter, but they do
not need to create a separate project environment.

****************
 Python Version
****************

Flower requires at least `Python 3.10 <https://docs.python.org/3.10/>`_.

.. note::

    Due to a known incompatibility with `ray <https://docs.ray.io/en/latest/>`_, we
    currently recommend utilizing at most `Python 3.11 <https://docs.python.org/3.11/>`_
    for running Flower simulations.

********************
 Virtualenv with uv
********************

From the repository root, run the bootstrap script:

.. code-block:: shell

    ./dev/bootstrap.sh

This creates ``framework/.venv`` with the repository's default Python version and
installs all framework dependencies from ``framework/uv.lock``. To use a specific Python
version, pass it as the first argument:

.. code-block:: shell

    ./dev/bootstrap.sh 3.10.19

Activate the created virtual environment with:

.. code-block:: shell

    source framework/.venv/bin/activate

You can also run uv directly from the ``framework`` directory:

.. code-block:: shell

    cd framework
    uv sync --python 3.10.19 --locked --all-extras --all-groups

****************************
 Python Versions with Pyenv
****************************

If you use `pyenv <https://github.com/pyenv/pyenv>`_, install the Python version you
want uv to use:

.. code-block:: shell

    pyenv install 3.10.19

Then pass that version to the bootstrap script:

.. code-block:: shell

    ./dev/bootstrap.sh 3.10.19

*******************************
 Python Versions with Anaconda
*******************************

If you prefer conda, create and activate a Python environment, then let uv create the
framework project environment. See the `conda installation guide
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ for
installation instructions.

.. code-block:: shell

    conda create -n flower-3.10.19 python=3.10.19
    conda activate flower-3.10.19
    ./dev/bootstrap.sh 3.10.19
