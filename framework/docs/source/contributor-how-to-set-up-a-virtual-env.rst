######################
 Set up a virtual env
######################

It is recommended to run your Python setup within a virtual environment. This guide
shows three different examples how to create a virtual environment with pyenv
virtualenv, poetry, or Anaconda. You can follow the instructions or choose your
preferred setup.

****************
 Python Version
****************

Flower requires at least `Python 3.10 <https://docs.python.org/3.10/>`_.

.. note::

    Due to a known incompatibility with `ray <https://docs.ray.io/en/latest/>`_, we
    currently recommend utilizing at most `Python 3.11 <https://docs.python.org/3.11/>`_
    for running Flower simulations.

**********************************
 Virtualenv with Pyenv/Virtualenv
**********************************

One of the recommended virtual environment is `pyenv
<https://github.com/pyenv/pyenv>`_/`virtualenv
<https://github.com/pyenv/pyenv-virtualenv>`_. Please see `Flower apps
<https://github.com/flwrlabs/flower/tree/main/hub/apps/>`_ for details.

Once Pyenv is set up, you can use it to install `Python Version 3.10
<https://docs.python.org/3.10/>`_ or above:

.. code-block:: shell

    pyenv install 3.10.19

Create the virtualenv with:

.. code-block:: shell

    pyenv virtualenv 3.10.19 flower-3.10.19

Activate the virtualenv by running the following command:

.. code-block:: shell

    echo flower-3.10.19 > .python-version

************************
 Virtualenv with Poetry
************************

The Flower examples are based on `Poetry <https://python-poetry.org/docs/>`_ to manage
dependencies. After installing Poetry you can create and activate a virtual environment
with:

.. code-block:: shell

    poetry shell

If you open a new terminal you can activate the previously created virtual environment
with the following command:

.. code-block:: shell

    source $(poetry env info --path)/bin/activate

**************************
 Virtualenv with Anaconda
**************************

If you prefer to use Anaconda for your virtual environment then install and setup the
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
package. After setting it up you can create a virtual environment with:

.. code-block:: shell

    conda create -n flower-3.10.19 python=3.10.19

and activate the virtual environment with:

.. code-block:: shell

    conda activate flower-3.10.19
