Installation
============

Python Version
--------------

Flower Datasets requires `Python 3.10 <https://docs.python.org/3.10/>`_ or above.


Install stable release (pip)
----------------------------

Stable releases are available on `PyPI <https://pypi.org/project/flwr_datasets/>`_

.. code-block:: bash

  python -m pip install flwr-datasets

For vision datasets (e.g. MNIST, CIFAR10) ``flwr-datasets`` should be installed with the ``vision`` extra

.. code-block:: bash

  python -m pip install "flwr-datasets[vision]"

For audio datasets (e.g. Speech Command) ``flwr-datasets`` should be installed with the ``audio`` extra

.. code-block:: bash

  python -m pip install "flwr-datasets[audio]"

Install directly from GitHub (pip)
----------------------------------

Installing Flower Datasets directly from GitHub ensures you have access to the most up-to-date version. 
If you encounter any issues or bugs, you may be directed to a specific branch containing a fix before 
it becomes part of an official release.

.. code-block:: bash

  python -m pip install "flwr-datasets@git+https://github.com/flwrlabs/flower.git"\
  "@TYPE-HERE-BRANCH-NAME#subdirectory=datasets"

Similarly to the situation before, you can specify the ``vision`` or ``audio`` extra after the name of the library.

.. code-block:: bash

  python -m pip install "flwr-datasets[vision]@git+https://github.com/flwrlabs/flower.git"\
  "@TYPE-HERE-BRANCH-NAME#subdirectory=datasets"

e.g. for the main branch:

.. code-block:: bash

  python -m pip install "flwr-datasets@git+https://github.com/flwrlabs/flower.git"\
  "@main#subdirectory=datasets"

Since `flwr-datasets` is a part of the Flower repository, the `subdirectory` parameter (at the end of the URL) is used to specify the package location in the GitHub repo.

Verify installation
-------------------

The following command can be used to verify if Flower Datasets was successfully installed:

.. code-block:: bash

  python -c "import flwr_datasets;print(flwr_datasets.__version__)"

If everything works, it should print the version of Flower Datasets to the command line:

.. parsed-literal::

  |release|

