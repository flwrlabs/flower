:og:description: Comparison of Flower Runtimes for simulation and deployment of federated AI apps.
.. meta::
    :description: Comparison of Flower Runtimes for simulation and deployment of federated AI apps.

###########################
 Flower Runtime Comparison
###########################

Flower Apps can be executed with both `simulation <how-to-run-simulations.html>`_ or
`deployment <how-to-run-flower-with-deployment-engine.html>`_ runtimes. Switching
between runtimes simply requires specifying a different type of `federation` when
executing the `flwr run` command via the `Flower CLI <ref-api-cli.html>`_.

While Flower Apps can run in both simulation and deployment without making changes to
their code, there are some differences on how they get executed depending on the
runtime. The following table outlines the key characteristics that differ when executing
a Flower App in a Flower federation with the simulation runtime from another using the
deployment runtime.

.. list-table::
    :widths: 15 25 25
    :header-rows: 1

    - - Dimension
      - Simulation Runtime
      - Deployment Runtime
    - - **Lifecycle Stage**
      - Ideal for rapid prototyping, algorithm validation, research, debugging, and
        experimentation.
      - Deploy validated use cases in production, real-world privacy-preserving
        applications.
    - - **Environment**
      - Local or remote, single-node or multi-node, controlled.
      - Distributed, remote.
    - - **Data**
      - Simulated data partitions, public or private datasets or artificially generated
        - a natural fit for `Flower Datasets <https://flower.ai/docs/datasets/>`_.
      - Real client-side data, residing on local databases or filesystems.
    - - **Backend**
      - Multiple Python processes/workers coordinated using `Ray
        <https://docs.ray.io/>`_.
      - Multiple independent processes or subprocesses running in coordination with the
        SuperLink and SuperNodes.
    - - **Execution Mode**
      - Multiprocessing execution where each process simulates a distinct client.
      - Parallel execution mode across a network of physical machines/devices or
        computing environment.
    - - **Communication**
      - In-memory communication.
      - TLS-enabled gRPC.
    - - **Server-side Infrastructure**
      - In the standard local CLI workflow, ``flwr run`` submits the run to a managed
        local SuperLink, which then coordinates the simulation runtime and the workers
        acting as `simulated` SuperNodes. The simulation runtime itself can still be
        started with or without the `SuperLink <ref-api-cli.html#flower-superlink>`_.
      - The SuperLink awaits for SuperNodes to connect. User interface with the
        SuperLink using the `Flower CLI <ref-api-cli.html>`_.
    - - **Server-side App execution**
      - A ``ServerApp`` process is initialized inside a controlled environment and
        communicates in-memory with workers.
      - ``ServerApp`` `process or subprocess <ref-flower-network-communication.html>`_
        runs independently from the SuperLink and communicates with it over gRPC via the
        ServerAppIO API.
    - - **Client-side Infrastructure**
      - No user-managed client-side infrastructure is required. For local CLI workflows,
        the managed local SuperLink and simulation runtime remain self-contained.
      - SuperNodes connect to the SuperLink via TLS-enabled gRPC using the Fleet API.
        Node authentication can be enabled.
    - - **Client-side App execution**
      - Each process executes a ``ClientApp`` on demand. They might execute multiple
        instances of the same ``ClientApp`` to simulate large amounts of clients.
        ``ClientApps`` are stateless.
      - Initialized as a ``ClientApp`` `process or subprocess
        <ref-flower-network-communication.html>`_, it runs independently from the
        SuperNode and communicates with it over gRPC via the ClientAppIo API.
        ``ClientApps`` are stateless.
