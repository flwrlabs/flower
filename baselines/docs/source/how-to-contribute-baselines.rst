Contribute Baselines
====================

Do you have a new federated learning paper and want to add a new baseline to Flower? Or do you want to add an experiment to an existing baseline paper? Great, we really appreciate your contribution.

The goal of Flower Baselines is to reproduce experiments from popular papers to accelerate researchers by enabling faster comparisons to new strategies, datasets, models, and federated pipelines in general. 

Before you start to work on a new baseline or experiment, please check the `Flower Issues <https://github.com/flwrlabs/flower/issues>`_ or `Flower Pull Requests <https://github.com/flwrlabs/flower/pulls>`_ to see if someone else is already working on it. Please open a new issue if you are planning to work on a new baseline or experiment with a short description of the corresponding paper and the experiment you want to contribute.
If you are proposing a brand new baseline, please indicate what experiments from the paper are planning to include.

Requirements
------------

Contributing a new baseline is really easy. You only have to make sure that your federated learning experiments run with Flower, use `Flower Datasets <https://flower.ai/docs/datasets/>`_, and replicate the results of a paper.
Preferably, the baselines make use of PyTorch, but other ML frameworks are also welcome. The baselines are expected to run in a machine with Ubuntu 22.04, but if yours runs also on macOS even better!


Add a new Flower Baseline
-------------------------
.. note::
    The instructions below are a more verbose version of what's present in the `Baselines README on GitHub <https://github.com/flwrlabs/flower/tree/main/baselines>`_.

Let's say you want to contribute the code of your most recent Federated Learning publication, *FedAwesome*. There are only three steps necessary to create a new *FedAwesome* Flower Baseline:

#. **Get the Flower source code on your machine**
    #. Fork the Flower codebase: go to the `Flower GitHub repo <https://github.com/flwrlabs/flower>`_ and fork the code (click the *Fork* button in the top-right corner and follow the instructions)
    #. Clone the (forked) Flower source code: :code:`git clone git@github.com:[your_github_username]/flower.git`
#. **Create a new baseline using the template**
    #. Create a new Python environment with Python 3.12 (we recommend doing this with `pyenv <https://github.com/pyenv/pyenv>`_)
    #. Install flower with: :code:`pip install flwr`.
    #. Navigate to the baselines directory and run: :code:`flwr new @flwrlabs/baseline`. This will create a new baseline named `baseline`.
    #. Rename the newly created baseline directory to your baseline name, e.g., :code:`mv baseline fedawesome`.
    #. Follow the instructions in the :code:`README.md` in your baseline directory.
    
    .. tip::
        At this point, your baseline contains source code showing how a simple :code:`PyTorch+CIFAR10` project can be built with Flower.
        You can run it directly by executing :code:`flwr run .` from inside the directory of your baseline. Update the code with that
        needed to implement your baseline.

#. **Open a pull request**
    #. Stage your changes: :code:`git add .`
    #. Commit & push: :code:`git commit -m "Create new FedAwesome baseline" ; git push`
    #. Open a pull request: go to *your* fork of the Flower codebase and create a pull request that targets the Flower ``main`` branch
    #. Interact with the Flower maintainers during the merging process and make adjustments to your code as needed.

Further reading:

* `GitHub docs: About forks <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks>`_
* `GitHub docs: Creating a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_
* `GitHub docs: Creating a pull request from a fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_



Usability
---------

Flower is known and loved for its usability. Therefore, make sure that your baseline or experiment can be executed with a single command after installing the baseline project:

.. code-block:: bash

    # Install the baseline project
    pip install -e .

    # Run the baseline using default config
    flwr run .

    # Run the baseline overriding the config
    flwr run . --run-config "lr=0.01 num-server-rounds=200"


We look forward to your contribution! 