Use Baselines
=============

.. warning::
  We are changing the way we structure the Flower baselines. While we complete the transition to the new format, you can still find the existing baselines and use them: `baselines (old) <https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines>`_.
  Currently, you can make use of baselines for `FedAvg <https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist>`_, `FedOpt <https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization>`_,  and `LEAF-FEMNIST <https://github.com/flwrlabs/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist>`_.


Structure
---------

All baselines are available in the directory `baselines <https://github.com/flwrlabs/flower/blob/main/baselines>`_ with each baseline directory being fully self-contained in terms of source code. In addition, each baseline uses its very own Python environment as designed by the contributors of such baseline in order to replicate the experiments in the paper. Each baseline directory contains the following structure: 

.. code-block:: shell

    flower/baselines/<baseline_name>/
                          ├── LICENSE
                          ├── README.md
                          ├── pyproject.toml # defines dependencies
                          ├── _static # optionally a directory to save plots
                          └── <baseline_name>
                                      └── *.py # several .py files


Setting up your machine
-----------------------

.. tip::
  Flower baselines are designed to run on Ubuntu 22.04 and Python 3.10. While a GPU is not required to run the baselines, some of the more computationally demanding ones do benefit from GPU acceleration.
  All baselines are expected to make use of `pyenv <https://github.com/pyenv/pyenv>`_.

.. note::
  We are in the process of migrating all baselines to use `flwr run`. Those that haven't yet been migrated still make use of `Poetry <https://python-poetry.org/docs/>`_, a tool to manage Python dependencies.
  Identifying whether the baseline you want to run requires Poetry or not is easy: check if the `Environment Setup` section in the baseline readme mentions Poetry. 
  Follow the instructions later in this section if you need to setup Poetry in your system.

Let's begin by installing :code:`pyenv`. We'll be following the standard procedure. Please refer to the `pyenv docs <https://github.com/pyenv/pyenv#installation>`_ for alternative ways of installing it, including for platforms other than Ubuntu.

.. code-block:: bash

    # first install a few packages needed later for pyenv
    sudo apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev \
              libreadline-dev libbz2-dev libffi-dev liblzma-dev

    # now clone pyenv into your home directory (this is the default way of installing pyenv)
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv

    # Then add pyenv to your path by adding the below to your .bashrc/.zshrc
    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"

Verify your installation by opening a new terminal and

.. code-block:: bash

    # check python versions available
    pyenv versions
    # * system (...)  # <-- it should just show one

Then you can proceed and install any version of Python. Baselines use Python 3.10, so we'll be installing a recent version of it.

.. code-block:: bash

    pyenv install 3.10.14
    # this will take a little while
    # once done, you should see that that version is available
    pyenv versions
    # system
    # * 3.10.14  # <-- you just installed this

Next, let's install the :code:`virtualenv` plugin. Check `the documentation <https://github.com/pyenv/pyenv-virtualenv>`_ for alternative installation methods.

.. code-block:: bash

    # Clone `pyenv-virtualenv`
    git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

    # Restart your shell
    exec "$SHELL"


Using :code:`pyenv`
~~~~~~~~~~~~~~~~~~~

Creating a virtual environment can be done as follows:

.. code-block:: bash

    # Create an environment for Python 3.10.14 named test-env
    pyenv virtualenv 3.10.14 test-env

    # Then activate it
    pyenv activate test-env

    # Deactivate it as follows
    pyenv deactivate


(optional) Setup Poetry
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have :code:`pyenv` installed, we are ready to install :code:`poetry`. It can be done from a single command:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

    # add to path by putting this line at the end of your .zshrc/.bashrc
    export PATH="$HOME/.local/bin:$PATH"


To install Poetry from source, to customise your installation, or to further integrate Poetry with your shell after installation, please check `the Poetry documentation <https://python-poetry.org/docs/#installation>`_.


Using a Flower Baseline
-----------------------

To use Flower Baselines you need first to install :code:`pyenv` and, depending on the baselines, also :code:`Poetry`, then:

1. Clone the flower repository

.. code-block:: bash

    git clone https://github.com/flwrlabs/flower.git && cd flower

2. Navigate inside the directory of the baseline you'd like to run
3. Follow the :code:`[Environment Setup]` instructions in the :code:`README.md`. 
4. Run the baseline as indicated in the :code:`[Running the Experiments]` section in the :code:`README.md` or in the :code:`[Expected Results]` section to reproduce the experiments in the paper.
