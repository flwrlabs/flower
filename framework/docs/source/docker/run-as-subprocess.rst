:og:description: Execute Flower ServerApp and ClientApp as subprocesses within SuperLink and SuperNode Docker containers to optimize resource usage, isolation, and security.
.. meta::
    :description: Execute Flower ServerApp and ClientApp as subprocesses within SuperLink and SuperNode Docker containers to optimize resource usage, isolation, and security.

############################################
 Run ServerApp or ClientApp as a Subprocess
############################################

The SuperLink and SuperNode components support two distinct isolation modes, allowing
for flexible deployment and control:

1. Subprocess Mode: In this configuration (default), the SuperLink and SuperNode take
   responsibility for launching the ServerApp and ClientApp processes internally. This
   differs from the ``process`` isolation-mode which uses separate containers, as
   demonstrated in the :doc:`tutorial-quickstart-docker` guide.

   Using the ``subprocess`` approach reduces the number of running containers, which can
   be beneficial for environments with limited resources. However, it also means that
   the applications are not isolated from their parent containers, which may introduce
   additional security concerns.

2. Process Mode: In this mode, the ServerApp and ClientApps run in completely separate
   processes. Unlike the alternative Subprocess mode, the SuperLink or SuperNode does
   not attempt to create or manage these processes. Instead, they must be started
   externally.

Both modes can be mixed for added flexibility. For instance, you can run the SuperLink
in ``subprocess`` mode while keeping the SuperNode in ``process`` mode, or vice versa.

To run the SuperLink and SuperNode in isolation mode ``process``, refer to the
:doc:`tutorial-quickstart-docker` guide. To run them in ``subprocess`` mode, follow the
instructions below.

.. tab-set::

    .. tab-item:: ServerApp

        **Prerequisites**

        1. Before running the ServerApp as a subprocess, ensure that the FAB dependencies have
        been installed in the SuperLink images. This can be done by extending the SuperLink image:

        .. code-block:: dockerfile
            :caption: superlink.Dockerfile
            :linenos:
            :substitutions:

            FROM flwr/superlink:|stable_flwr_version|

            WORKDIR /app
            COPY pyproject.toml .
            RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
                && python -m pip install -U --no-cache-dir .

            ENTRYPOINT ["flower-superlink"]

        2. Next, build the SuperLink Docker image by running the following command in the
        directory where Dockerfile is located:

        .. code-block:: shell

            $ docker build -f superlink.Dockerfile -t flwr_superlink:0.0.1 .

        **Run the ServerApp as a Subprocess**

        Start the SuperLink and run the ServerApp as a subprocess (note that
        the subprocess mode is the default, so you do not have to explicitly set the ``--isolation`` flag):

        .. code-block:: shell

            $ docker run --rm \
                -p 8000:8000 -p 9092:9092 \
                --detach \
                flwr_superlink:0.0.1 \
                --insecure \
                --host 0.0.0.0

    .. tab-item:: ClientApp

        **Prerequisites**

        1. Before running the ClientApp as a subprocess, ensure that the FAB dependencies have
        been installed in the SuperNode images. This can be done by extending the SuperNode
        image:

        .. code-block:: dockerfile
            :caption: supernode.Dockerfile
            :linenos:
            :substitutions:

            FROM flwr/supernode:|stable_flwr_version|

            WORKDIR /app
            COPY pyproject.toml .
            RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
                && python -m pip install -U --no-cache-dir .

            ENTRYPOINT ["flower-supernode"]

        2. Next, build the SuperNode Docker image by running the following command in the
        directory where Dockerfile is located:

        .. code-block:: shell

            $ docker build -f supernode.Dockerfile -t flwr_supernode:0.0.1 .

        **Run the ClientApp as a Subprocess**

        Start the SuperNode and run the ClientApp as a subprocess (note that
        the subprocess mode is the default, so you do not have to explicitly set the ``--isolation`` flag):

        .. code-block:: shell

            $ docker run --rm \
                --detach \
                flwr_supernode:0.0.1 \
                --insecure \
                --superlink <superlink-address>:9092
