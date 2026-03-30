Use an App from Flower Hub
==========================

Flower Hub is the home for federated applications. It provides a unified experience for discovering, running, and sharing federated apps across both simulation and deployment environments.

Create a New App from Hub
-------------------------

First, ensure that :code:`flwr` is installed on your machine:

.. code-block:: bash

    pip install flwr

Next, select the app you want to download. For example, to create an app from `@flwrlabs/quickstart-pytorch <https://flower.ai/apps/flwrlabs/quickstart-pytorch/>`_, run:

.. code-block:: bash

    flwr new @flwrlabs/quickstart-pytorch

This command downloads the application from Flower Hub into your local environment.


Run a Hub App
-------------

After creating the app, you can run it in either **Simulation** or **Deployment** runtime without modifying the source code.

If you are new to Flower, we recommend starting with **Simulation Runtime**, as it requires fewer components to be launched manually. By default, :code:`flwr run` uses the Simulation Runtime.


Run with the Simulation Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tip::
   See the `Simulation Runtime documentation <https://flower.ai/docs/framework/how-to-run-simulations.html>`_ to learn more about running simulations, scaling the number of virtual SuperNodes, and configuring CPU/GPU resources for your ClientApp.

First, install the dependencies defined in `pyproject.toml`:

.. code-block:: bash

    cd <app_name>
    pip install -e .

Then, run the app with the default settings:

.. code-block:: bash

    flwr run .

This starts the simulation locally using Flower's built-in Simulation Runtime.


Run with the Deployment Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the app using Flower's Deployment Runtime, point the ``SuperNode`` to the path where your data is. For prototyping and first-time users, we recommend generating demo data using `Flower Datasets <https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html>`_.

Next, launch the SuperNode passing as arguments the address of the SuperLink and the path to the data. For example:

.. code-block:: bash

    flower-supernode \
        --insecure \
        --superlink <SUPERLINK-FLEET-API> \
        --node-config="data-path='/path/to/demo_data/partition_0'"

.. tip::
   For a detailed walkthrough, see the `step-by-step guide <https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html>`_ on using the Deployment Runtime.

.. note::
   In this example, :code:`data-path` is an application-specific configuration key used by the ClientApp to locate its local dataset.
   The required :code:`--node-config` parameters depend on the specific app you are running. Always refer to the app's README for the exact configuration keys and expected values.

Make sure the environment of each SuperNode has all required dependencies for the app you envision it to run installed.

Finally, launch the run using :code:`flwr run`, pointing to the appropriate SuperLink connection:

.. code-block:: bash

    flwr run . <SUPERLINK-CONNECTION> --stream

.. tip::
   Afterward, consider enabling `secure TLS connections <https://flower.ai/docs/framework/how-to-enable-tls-connections.html>`_ and configuring `SuperNode authentication <https://flower.ai/docs/framework/how-to-authenticate-supernodes.html>`_ for production deployments.

.. tip::
   Learn more about SuperLink connections and how to configure them in the `Flower Configuration documentation <https://flower.ai/docs/framework/ref-flower-configuration.html>`_.

Run a Hub App Without Creating It Locally
-----------------------------------------

In the Deployment Runtime, you can run an app directly from Flower Hub without first creating it locally. Note this assumes SuperNodes are already connected to the SuperLink:

.. code-block:: bash

    flwr run @<account_name>/<app_name> --stream

In this case, the SuperLink downloads the app from Flower Hub and distributes the :code:`FAB` file to each SuperNode for execution.
