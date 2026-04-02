:og:description: Learn how to install Flower, the Python-based federated learning framework, using PyPI, conda, or Docker in this easy-to-follow guide.
.. meta::
    :description: Learn how to install Flower, the Python-based federated learning framework, using PyPI, conda, or Docker in this easy-to-follow guide.

################
 Install Flower
################

****************
 Python version
****************

Flower requires at least `Python 3.10 <https://docs.python.org/3.10/>`_.

************************
 Install stable release
************************

Using pip
=========

Stable releases are available on `PyPI <https://pypi.org/project/flwr/>`_:

::

    python -m pip install flwr

For simulations that use the Simulation Runtime, ``flwr`` should be installed with the
``simulation`` extra:

::

    python -m pip install "flwr[simulation]"

Using conda (or mamba)
======================

Flower can also be installed from the ``conda-forge`` channel.

If you have not added ``conda-forge`` to your channels, you will first need to run the
following:

::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Once the ``conda-forge`` channel has been enabled, ``flwr`` can be installed with
``conda``:

::

    conda install flwr

or with ``mamba``:

::

    mamba install flwr

*********************
 Verify installation
*********************

The following command can be used to verify if Flower was successfully installed. If
everything worked, it should print the version of Flower to the command line:

.. code-block:: console
    :substitutions:

    $ flwr --version
    Flower version: |stable_flwr_version|

.. note::

    If you're on Windows and see unexpected terminal output (e.g.: ``� □[32m□[1m``),
    check :ref:`this FAQ entry <faq-windows-unexpected-output>`.

*******************************
 Advanced installation options
*******************************

Install via Docker
==================

:doc:`Run Flower using Docker <docker/index>`

Install pre-release
===================

New (possibly unstable) versions of Flower are sometimes available as pre-release
versions (alpha, beta, release candidate) before the stable release happens:

::

    python -m pip install -U --pre flwr

For simulations that use the Simulation Runtime, ``flwr`` pre-releases should be
installed with the ``simulation`` extra:

::

    python -m pip install -U --pre 'flwr[simulation]'

Install nightly release
=======================

The latest (potentially unstable) changes in Flower are available as nightly releases:

::

    python -m pip install -U flwr-nightly

For simulations that use the Simulation Runtime, ``flwr-nightly`` should be installed
with the ``simulation`` extra:

::

    python -m pip install -U flwr-nightly[simulation]
