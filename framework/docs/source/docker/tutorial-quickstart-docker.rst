:og:description: Containerize a Flower federated learning project and run it end-to-end with this guide, including SuperLink, SuperNode, ServerApp, and ClientApp setup.
.. meta::
    :description: Containerize a Flower federated learning project and run it end-to-end with this guide, including SuperLink, SuperNode, ServerApp, and ClientApp setup.

########################
 Quickstart with Docker
########################

This quickstart aims to guide you through the process of containerizing a Flower project
and running it end to end using Docker on your local machine.

This tutorial does not use production-ready settings, so you can focus on understanding
the basic workflow that uses the minimum configurations.

***************
 Prerequisites
***************

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.

****************
 Step 1: Set Up
****************

1. Create a new Flower project (PyTorch):

   .. code-block:: console

       $ flwr new @flwrlabs/quickstart-pytorch

       🔗 Requesting download link for @flwrlabs/quickstart-pytorch...
       🔽 Downloading ZIP into memory...
       📦 Unpacking into /Users/alice/quickstart-pytorch...
       🎊 Flower App creation successful.

       To run your Flower App, first install its dependencies:

               cd quickstart-pytorch && pip install -e .

       then, run the app:

               flwr run .

       💡 Check the README in your app directory to learn how to
       customize it and how to run it using the Deployment Runtime.

       $ cd quickstart-pytorch

2. Create a new Docker bridge network called ``flwr-network``:

   .. code-block:: console

       $ docker network create --driver bridge flwr-network

   User-defined networks, such as ``flwr-network``, enable IP resolution of container
   names, a feature absent in the default bridge network. This simplifies quickstart
   example by avoiding the need to determine host IP first.

*****************************
 Step 2: Start the SuperLink
*****************************

Open your terminal and run:

.. code-block:: console
    :substitutions:

    $ docker run --rm \
          -p 9091:9091 -p 9092:9092 -p 9093:9093 \
          --network flwr-network \
          --name superlink \
          --detach \
          flwr/superlink:|stable_flwr_version| \
          --insecure \
          --isolation \
          process

.. dropdown:: Understand the command

    * ``docker run``: This tells Docker to run a container from an image.
    * ``--rm``: Remove the container once it is stopped or the command exits.
    * ``-p 9091:9091 -p 9092:9092 -p 9093:9093``: Map port ``9091``, ``9092`` and ``9093`` of the
      container to the same port of the host machine, allowing other services to access the
      ServerAppIO API on ``http://localhost:9091``, the Fleet API on ``http://localhost:9092`` and
      the Control API on ``http://localhost:9093``.
    * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
    * ``--name superlink``: Assign the name ``superlink`` to the container.
    * ``--detach``: Run the container in the background, freeing up the terminal.
    * :substitution-code:`flwr/superlink:|stable_flwr_version|`: The name of the image to be run and the specific
      tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a :doc:`specific version <pin-version>` of the image.
    * ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
      unencrypted communication.
    * ``--isolation process``: Tells the SuperLink that the ServerApp is executed by separate
      independent process. The SuperLink does not attempt to execute it. You can learn more about
      the different process modes here: :doc:`run-as-subprocess`.

******************************
 Step 3: Start the SuperNodes
******************************

Start two SuperNode containers.

1. Start the first container:

   .. code-block:: console
       :substitutions:

       $ docker run --rm \
           -p 9094:9094 \
           --network flwr-network \
           --name supernode-1 \
           --detach \
           flwr/supernode:|stable_flwr_version|  \
           --insecure \
           --superlink superlink:9092 \
           --node-config "partition-id=0 num-partitions=2" \
           --clientappio-api-address 0.0.0.0:9094 \
           --isolation process

   .. dropdown:: Understand the command

       * ``docker run``: This tells Docker to run a container from an image.
       * ``--rm``: Remove the container once it is stopped or the command exits.
       * ``-p 9094:9094``: Map port ``9094`` of the container to the same port of
         the host machine, allowing other services to access the SuperNode API on
         ``http://localhost:9094``.
       * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
       * ``--name supernode-1``: Assign the name ``supernode-1`` to the container.
       * ``--detach``: Run the container in the background, freeing up the terminal.
       * :substitution-code:`flwr/supernode:|stable_flwr_version|`: This is the name of the
         image to be run and the specific tag of the image.
       * ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
         unencrypted communication.
       * ``--superlink superlink:9092``: Connect to the SuperLink's Fleet API at the address
         ``superlink:9092``.
       * ``--node-config "partition-id=0 num-partitions=2"``: Set the partition ID to ``0`` and the
         number of partitions to ``2`` for the SuperNode configuration.
       * ``--clientappio-api-address 0.0.0.0:9094``: Set the address and port number that the
         SuperNode is listening on to communicate with the ClientApp process. If
         multiple SuperNodes are started on the same machine, use a different port number for each SuperNode.
         (E.g. In the next step, we set the second SuperNode container to listen on port 9095)
       * ``--isolation process``: Tells the SuperNode that the ClientApp is executed by a separate
         independent process. The SuperNode does not attempt to execute it.

2. Start the second container:

   .. code-block:: console
       :substitutions:

       $ docker run --rm \
           -p 9095:9095 \
           --network flwr-network \
           --name supernode-2 \
           --detach \
           flwr/supernode:|stable_flwr_version|  \
           --insecure \
           --superlink superlink:9092 \
           --node-config "partition-id=1 num-partitions=2" \
           --clientappio-api-address 0.0.0.0:9095 \
           --isolation process

***************************************************
 Step 4: Start the SuperExec to execute ServerApps
***************************************************

The **SuperExec** Docker image comes with a pre-installed version of Flower and serves
as a base for building your own image. Use a **single** image and select the desired
behavior (ServerApps vs ClientApps) at runtime via the ``--plugin-type`` flag. SuperExec
spawns the corresponding processes on demand.

When using SuperExec with the *serverapp* plugin, pass ``--appio-api-address`` pointing
to the SuperLink's **ServerAppIO API** endpoint.

1. Create a Dockerfile called ``superexec.Dockerfile`` and paste the following:

   .. code-block:: dockerfile
       :caption: superexec.Dockerfile
       :substitutions:

       FROM flwr/superexec:|stable_flwr_version|

       WORKDIR /app

       COPY pyproject.toml .
       RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
          && python -m pip install -U --no-cache-dir .

       ENTRYPOINT ["flower-superexec"]

   .. dropdown:: Understand the Dockerfile

       * :substitution-code:`FROM flwr/superexec:|stable_flwr_version|`: This line specifies that the Docker image
         to be built from is the ``flwr/superexec`` image, version :substitution-code:`|stable_flwr_version|`.
       * ``WORKDIR /app``: Set the working directory for the container to ``/app``.
         Any subsequent commands that reference a directory will be relative to this directory.
       * ``COPY pyproject.toml .``: Copy the ``pyproject.toml`` file.
         from the current working directory into the container's ``/app`` directory.
       * ``RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml``: Remove the ``flwr`` dependency
         from the ``pyproject.toml``.
       * ``python -m pip install -U --no-cache-dir .``: Run the ``pip`` install command to
         install the dependencies defined in the ``pyproject.toml`` file.

         The ``-U`` flag indicates that any existing packages should be upgraded, and
         ``--no-cache-dir`` prevents pip from using the cache to speed up the installation.
       * ``ENTRYPOINT ["flower-superexec"]``: Set the command ``flower-superexec`` to be
         the default command run when the container is started.

   .. important::

       Note that `flwr <https://pypi.org/project/flwr/>`__ is already installed in the
       ``flwr/superexec`` base image, so only other package dependencies such as
       ``flwr-datasets``, ``torch``, etc., need to be installed. As a result, the
       ``flwr`` dependency is removed from the ``pyproject.toml`` after it has been
       copied into the Docker image (see line 5).

2. Afterward, in the directory that holds the Dockerfile, execute this Docker command to
   build the SuperExec image:

   .. code-block:: console

       $ docker build -f superexec.Dockerfile -t flwr_superexec:0.0.1 .

3. Start the **SuperExec for ServerApps** container:

   .. code-block:: console

       $ docker run --rm \
           --network flwr-network \
           --name superexec-serverapp \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type serverapp \
           --appio-api-address superlink:9091

   .. dropdown:: Understand the command

       * ``docker run``: This tells Docker to run a container from an image.
       * ``--rm``: Remove the container once it is stopped or the command exits.
       * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
       * ``--name superexec-serverapp``: Give the container a descriptive name.
       * ``--detach``: Run the container in the background, freeing up the terminal.
       * ``flwr_superexec:0.0.1``: This is the name of the image to be run and the specific tag
         of the image.
       * ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
         unencrypted communication. Secure connections will be added in future releases.
       * ``--plugin-type serverapp``: Load the *serverapp* plugin. SuperExec will spawn
         ServerApp processes as needed.
       * ``--appio-api-address superlink:9091``: Connect to the SuperLink's ServerAppIO API
         at the address ``superlink:9091``.

***************************************************
 Step 5: Start the SuperExec to execute ClientApps
***************************************************

For ClientApps, reuse the **same** image and change the plugin and API address. When
using the *clientapp* plugin, pass ``--appio-api-address`` pointing to the SuperNode's
**ClientAppIO API** endpoint.

1. (No new Dockerfile is needed; reuse ``flwr_superexec:0.0.1``.)
2. Start the first **SuperExec for ClientApps** container:

   .. code-block:: console

       $ docker run --rm \
           --network flwr-network \
           --name superexec-clientapp-1 \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type clientapp \
           --appio-api-address supernode-1:9094

   .. dropdown:: Understand the command

       * ``docker run``: This tells Docker to run a container from an image.
       * ``--rm``: Remove the container once it is stopped or the command exits.
       * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
       * ``--name superexec-clientapp-1``: Give the container a descriptive name.
       * ``--detach``: Run the container in the background, freeing up the terminal.
       * ``flwr_superexec:0.0.1``: This is the name of the image to be run and the specific tag
         of the image.
       * ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
         unencrypted communication. Secure connections will be added in future releases.
       * ``--plugin-type clientapp``: Load the *clientapp* plugin. SuperExec will spawn
         ClientApp processes as needed.
       * ``--appio-api-address supernode-1:9094``: Connect to the SuperNode's ClientAppIO API
         at the address ``supernode-1:9094``.

3. Start the second **SuperExec for ClientApps** container:

   .. code-block:: console

       $ docker run --rm \
           --network flwr-network \
           --name superexec-clientapp-2 \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type clientapp \
           --appio-api-address supernode-2:9095

************************************
 Step 6: Run the Quickstart Project
************************************

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

2. Add the following lines to the ``config.toml`` to add a new SuperLink connection:

   .. code-block:: toml
       :caption: config.toml

       [superlink.local-deployment]
       address = "127.0.0.1:9093"
       insecure = true

2. Run the ``quickstart-pytorch`` project and follow the ServerApp logs to track the
   execution of the run:

   .. code-block:: console

       $ flwr run . local-deployment --stream

********************************
 Step 7: Update the Application
********************************

1. Change the application code. For example, change the ``seed`` in
   ``pytorchexample/task.py`` to ``43`` and save it:

   .. code-block:: python
       :caption: pytorchexample/task.py

       # ...
       partition_train_test = partition.train_test_split(test_size=0.2, seed=43)
       # ...

2. Stop the running containers:

   .. note::

       If you have modified the dependencies listed in your ``pyproject.toml`` file, it
       is essential to rebuild images.

       If you haven't made any changes, you can skip steps 2 through 4.

   .. code-block:: console

       $ docker stop $(docker ps -a -q --filter ancestor=flwr_superexec:0.0.1)

3. Rebuild the SuperExec image:

   .. code-block:: console

       $ docker build -f superexec.Dockerfile -t flwr_superexec:0.0.1 .

4. Launch **one SuperExec container** for the new ServerApp and **two SuperExec
   containers** for the new ClientApps:

   .. code-block:: console

       $ docker run --rm \
           --network flwr-network \
           --name superexec-serverapp \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type serverapp \
           --appio-api-address superlink:9091
       $ docker run --rm \
           --network flwr-network \
            --name superexec-clientapp-1 \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type clientapp \
           --appio-api-address supernode-1:9094
       $ docker run --rm \
           --network flwr-network \
           --name superexec-clientapp-2 \
           --detach \
           flwr_superexec:0.0.1 \
           --insecure \
           --plugin-type clientapp \
           --appio-api-address supernode-2:9095

5. Run the updated project:

   .. code-block:: console

       $ flwr run . local-deployment --stream

******************
 Step 8: Clean Up
******************

Remove the containers and the bridge network:

.. code-block:: console

    $ docker stop $(docker ps -a -q --filter ancestor=flwr_superexec:0.0.1) \
       supernode-1 \
       supernode-2 \
       superlink
    $ docker network rm flwr-network

******************
 Where to Go Next
******************

- :doc:`enable-tls`
- :doc:`persist-superlink-state`
- :doc:`tutorial-quickstart-docker-compose`
