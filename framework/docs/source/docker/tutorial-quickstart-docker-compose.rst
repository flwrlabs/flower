:og:description: Learn how to quickly set up Flower using Docker Compose, enable TLS, and persist application state for federated learning with minimal configuration effort.
.. meta::
    :description: Learn how to quickly set up Flower using Docker Compose, enable TLS, and persist application state for federated learning with minimal configuration effort.

################################
 Quickstart with Docker Compose
################################

This quickstart shows you how to set up Flower using Docker Compose in a single command,
allowing you to focus on developing your application without worrying about the
underlying infrastructure.

You will also learn how to easily enable TLS encryption and persist application state
locally, giving you the freedom to choose the configuration that best suits your
project's needs.

***************
 Prerequisites
***************

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.
- Docker Compose V2 is `installed <https://docs.docker.com/compose/install/>`_.

****************
 Step 1: Set Up
****************

1. Clone the Docker Compose ``complete`` directory:

   .. code-block:: bash
       :substitutions:

       $ git clone --depth=1 --branch v|stable_flwr_version| https://github.com/flwrlabs/flower.git _tmp \
                   && mv _tmp/framework/docker/complete . \
                   && rm -rf _tmp && cd complete

2. Create a new Flower app (PyTorch):

   .. code-block:: bash

       $ flwr new @flwrlabs/quickstart-pytorch

3. Export the path of the newly created project. The path should be relative to the
   location of the Docker Compose files:

   .. code-block:: bash

       $ export PROJECT_DIR=quickstart-pytorch

   Setting the ``PROJECT_DIR`` helps Docker Compose locate the ``pyproject.toml`` file,
   allowing it to install dependencies in the ``ServerApp`` and ``ClientApp`` images
   correctly.

*************************************
 Step 2: Run Flower in Insecure Mode
*************************************

To begin, start Flower with the most basic configuration. In this setup, Flower will run
without TLS and without persisting the state.

.. note::

    Without TLS, the data sent between the services remains **unencrypted**. Use it only
    for development purposes.

    For production-oriented use cases, :ref:`enable TLS <TLS>` for secure data
    transmission.

Open your terminal and run:

.. code-block:: bash

    $ docker compose up --build -d

.. dropdown:: Understand the command

    * ``docker compose``: The Docker command to run the Docker Compose tool.
    * ``--build``: Rebuild the images for each service if they don't already exist.
    * ``-d``: Detach the containers from the terminal and run them in the background.

************************************
 Step 3: Run the Quickstart Project
************************************

Now that the Flower services have been started via Docker Compose, it is time to run the
quickstart example.

To ensure the ``flwr`` CLI connects to the SuperLink, you need to specify the SuperLink
connection in your Flower configuration file.

1. Find the Flower Configuration TOML file in your machine. This file is automatically
   create for your when you first use a Flower CLI command. Use ``flwr config list`` to
   see available SuperLink connections as well as the path to the configuration file.

   .. code-block:: console
       :emphasize-lines: 3

       $ flwr config list

       Flower Config file: /path/to/.flwr/config.toml
       SuperLink connections:
         supergrid
         local (default)

2. Add the following lines to the ``config.toml``:

   .. code-block:: toml
       :caption: config.toml

       [superlink.local-deployment]
       address = "127.0.0.1:9093"
       insecure = true

2. Run the quickstart example, monitor the ``ServerApp`` logs and wait for the summary
   to appear:

   .. code-block:: bash

       $ flwr run quickstart-pytorch local-deployment --stream

********************************
 Step 4: Update the Application
********************************

In the next step, change the application code.

1. For example, go to the ``task.py`` file in the ``quickstart-pytorch/pytorchexample/``
   directory and add a ``print`` call in the ``get_weights`` function:

   .. code-block:: python
       :caption: quickstart-pytorch/pytorchexample/task.py

       # ...
       def get_weights(net):
           print("Get weights")
           return [val.cpu().numpy() for _, val in net.state_dict().items()]


       # ...

2. Rebuild and restart the services.

   .. note::

       If you have modified the dependencies listed in your ``pyproject.toml`` file, it
       is essential to rebuild images.

       If you haven't made any changes, you can skip this step.

   Run the following command to rebuild and restart the services:

   .. code-block:: bash

       $ docker compose up --build -d

3. Run the updated quickstart example:

   .. code-block:: bash

       $ flwr run quickstart-pytorch local-deployment --stream

   In the ``ServerApp`` logs, you should find the ``Get weights`` line:

   .. code-block::
       :emphasize-lines: 5

       INFO :      Starting logstream for run_id `10386255862566726253`
       INFO :      Starting Flower ServerApp
       WARNING :   Option `--insecure` was set. Starting insecure HTTP channel to superlink:9091.
       🎊 Successfully installed quickstart-pytorch to /app/.flwr/apps/flower.quickstart-pytorch.1.0.0.35361a47.
       Get weights
       INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout

****************************************
 Step 5: Persisting the SuperLink State
****************************************

In this step, Flower services are configured to persist the state of the SuperLink
service, ensuring that it maintains its state even after a restart.

.. note::

    When working with Docker Compose on Linux, you may need to create the ``state``
    directory first and change its ownership to ensure proper access and permissions.

    For more information, consult the following page: :doc:`persist-superlink-state`.

1. Run the command:

   .. code-block:: bash

       $ docker compose -f compose.yml -f with-state.yml up --build -d

   .. dropdown:: Understand the command

       * ``docker compose``: The Docker command to run the Docker Compose tool.
       * ``-f compose.yml``: Specify the YAML file that contains the basic Flower service definitions.
       * | ``-f with-state.yml``: Specifies the path to an additional Docker Compose file that
         | contains the configuration for persisting the SuperLink state.
         |
         | Docker merges Compose files according to `merging rules <https://docs.docker.com/compose/multiple-compose-files/merge/#merging-rules>`_.
       * ``--build``: Rebuild the images for each service if they don't already exist.
       * ``-d``: Detach the containers from the terminal and run them in the background.

2. Rerun the ``quickstart-pytorch`` project:

   .. code-block:: bash

       $ flwr run quickstart-pytorch local-deployment --stream

3. Check the content of the ``state`` directory:

   .. code-block:: bash

       $ ls state/
       state.db

   You should see a ``state.db`` file in the ``state`` directory. If you restart the
   service, the state file will be used to restore the state from the previously saved
   data. This ensures that the data persists even if the containers are stopped and
   started again.

.. _tls:

*****************************
 Step 6: Run Flower with TLS
*****************************

1. To demonstrate how to enable TLS, generate self-signed certificates using the
   ``certs.yml`` Compose file.

   .. important::

       These certificates should be used only for development purposes.

       For production environments, use a service like `Let's Encrypt
       <https://letsencrypt.org/>`_ to obtain your certificates.

   Run the command:

   .. code-block:: bash

       $ docker compose -f certs.yml run --rm --build gen-certs

2. Add a new SuperLink connection to your Flower Configuration file:

   .. code-block:: toml
       :caption: config.toml

       [superlink.local-deployment-tls]
       address = "127.0.0.1:9093"
       root-certificates = "/absolute/path/to/superlink-certificates/ca.crt"

3. Restart the services with TLS enabled:

   .. code-block:: bash

       $ docker compose -f compose.yml -f with-tls.yml up --build -d

4. Rerun the ``quickstart-pytorch`` project:

   .. code-block:: bash

       $ flwr run quickstart-pytorch local-deployment-tls --stream

*********************************************
 Step 7: Add another SuperNode and ClientApp
*********************************************

You can add more SuperNodes and ClientApps by uncommenting their definitions in the
``compose.yml`` file:

.. code-block:: yaml
    :caption: compose.yml
    :substitutions:

      # other service definitions

      supernode-3:
        image: flwr/supernode:${FLWR_VERSION:-|stable_flwr_version|}
        command:
          - --insecure
          - --superlink
          - superlink:9092
          - --clientappio-api-address
          - 0.0.0.0:9096
          - --isolation
          - process
          - --node-config
          - "partition-id=1 num-partitions=2"
        depends_on:
          - superlink

      superexec-clientapp-3:
        build:
          context: ${PROJECT_DIR:-.}
          dockerfile_inline: |
            FROM flwr/superexec:${FLWR_VERSION:-|stable_flwr_version|}

            USER root
            RUN apt-get update \
                && apt-get -y --no-install-recommends install \
                build-essential \
                && rm -rf /var/lib/apt/lists/*
            USER app

            WORKDIR /app
            COPY --chown=app:app pyproject.toml .
            RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
              && python -m pip install -U --no-cache-dir .

            ENTRYPOINT ["flower-superexec"]
        command:
          - --insecure
          - --plugin-type
          - clientapp
          - --appio-api-address
          - supernode-3:9096
        deploy:
          resources:
            limits:
              cpus: "2"
        stop_signal: SIGINT
        depends_on:
          - supernode-3

If you also want to enable TLS for the new SuperNode, uncomment the definition in the
``with-tls.yml`` file:

.. code-block:: yaml
    :caption: with-tls.yml

      # other service definitions

      supernode-3:
        command:
          - --superlink
          - superlink:9092
          - --clientappio-api-address
          - 0.0.0.0:9096
          - --isolation
          - process
          - --node-config
          - "partition-id=1 num-partitions=2"
          - --root-certificates
          - certificates/superlink-ca.crt
        secrets:
          - source: superlink-ca-certfile
            target: /app/certificates/superlink-ca.crt

Restart the services with:

.. code-block:: bash

    $ docker compose up --build -d
    # or with TLS enabled
    $ docker compose -f compose.yml -f with-tls.yml up --build -d

*********************************************************
 Step 8: Persisting the SuperLink State and Enabling TLS
*********************************************************

To run Flower with persisted SuperLink state and enabled TLS, a slight change in the
``with-state.yml`` file is required:

1. Comment out the lines 2-6 and uncomment the lines 7-13:

   .. code-block:: yaml
       :caption: with-state.yml
       :linenos:
       :emphasize-lines: 2-13

         superlink:
           # command:
           # - --insecure
           # - --isolation
           # - process
           # - --database=state/state.db
           command:
             - --isolation
             - process
             - --ssl-ca-certfile=certificates/ca.crt
             - --ssl-certfile=certificates/server.pem
             - --ssl-keyfile=certificates/server.key
             - --database=state/state.db
           volumes:
             - ./state/:/app/state/:rw

2. Restart the services:

   .. code-block:: bash

       $ docker compose -f compose.yml -f with-tls.yml -f with-state.yml up --build -d

3. Rerun the ``quickstart-pytorch`` project:

   .. code-block:: bash

       $ flwr run quickstart-pytorch local-deployment-tls --stream

**************************************
 Step 9: Merge Multiple Compose Files
**************************************

You can merge multiple Compose files into a single file. For instance, if you wish to
combine the basic configuration with the TLS configuration, execute the following
command:

.. code-block:: bash

    $ docker compose -f compose.yml \
       -f with-tls.yml config --no-path-resolution > my_compose.yml

This will merge the contents of ``compose.yml`` and ``with-tls.yml`` into a new file
called ``my_compose.yml``.

*******************
 Step 10: Clean Up
*******************

Remove all services and volumes:

.. code-block:: bash

    $ docker compose down -v

******************
 Where to Go Next
******************

- :doc:`run-quickstart-examples-docker-compose`
