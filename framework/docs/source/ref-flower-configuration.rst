:og:description: Learn how to setup the Flower Configuration and how to use it to run your Flower App and interact with SuperLinks.
.. meta::
    :description: Learn how to setup the Flower Configuration and how to use it to run your Flower App and interact with SuperLinks.

######################
 Flower Configuration
######################

.. note::

    The Flower Configuration is a new feature introduced in Flower 1.26.0. While its
    functionality will evolve over time, as a start it serves as a direct replacement of
    the `federations` section in the `pyproject.toml` file of Flower Apps in older
    versions.

.. note::

    From Flower 1.28.0 onwards, it isn't recommended to define Simulation Runtime
    settings (i.e. fields starting with ``options.``) in the Flower Configuration. Check
    the :doc:`Simulation Runtime documentation <how-to-run-simulations>` for the
    recommended way to configure simulations.

**What is it?**

The Flower Configuration is a system for managing SuperLink connections, stored in a
TOML file (which we refer to as the Flower Configuration file) that lives in the
directory specified by the ``FLWR_HOME`` environment variable (which defaults to
``$HOME/.flwr`` when unset). It is designed to facilitate the usage of `Flower CLI
<ref-api-cli.html>`_ commands.

**Why use it?**

The Flower Configuration allows you to define reusable connection configurations that
you can reference by name when running ``flwr`` commands. For example, you can set up
configurations for local testing, staging servers, and production deployments, then
easily switch between them.

Most ``flwr`` commands (like ``flwr log``, ``flwr list``, and ``flwr stop``) can use the
Flower Configuration from anywhere on your system. The exception is ``flwr run``, which
must be executed from within a Flower App directory to access the app code.

.. tip::

    **First-time setup:** If the Flower Configuration file doesn't exist in your system,
    it will be automatically created for you the first time you run a Flower CLI
    command.

    **Upgrading from older versions:** If you are upgrading from a previous version of
    Flower, the ``federations`` section in your ``pyproject.toml`` file will be
    automatically migrated to the Flower Configuration. The syntax remains the same with
    the exception of the name of the section and the ``superlink`` keyword instead of
    ``tool.flwr.federations``.

*************************************
 Flower Configuration File Structure
*************************************

**Understanding the terminology**

The Flower Configuration file uses the term **"superlink"** to refer to SuperLink
connection configurations. Each connection configuration describes how to connect to a
Flower **SuperLink** (the central server component that coordinates federated learning).
You can define multiple SuperLink connection configurations for different scenarios,
such as local simulations or remote deployments.

The configuration structure is similar to the older ``federations`` section in
``pyproject.toml`` before Flower 1.26.0, but now lives in a central location and uses
clearer naming.

**Basic example**

.. code-block:: toml

    [superlink]
    default = "local"

    [superlink.local]
    address = ":local:"

    [superlink.local-poc]
    address = "127.0.0.1:9093"
    insecure = true

**Explanation**

- ``[superlink]`` section defines which connection configuration to use by default
- ``default = "local"`` means the ``superlink.local`` configuration will be used when
  you don't specify a connection explicitly in your ``flwr`` commands
- ``[superlink.local]`` defines a managed local SuperLink profile. The special address
  value ``":local:"`` tells Flower to launch or reuse the background local SuperLink on
  demand with its default on-disk state. The alternative value ``":local-in-memory:"``
  starts the managed local SuperLink with an in-memory database instead. The SuperLink
  will run simulations using the default configuration of the :doc:`Simulation Runtime
  <how-to-run-simulations>`.
- ``[superlink.local-poc]`` defines a configuration for connecting to a locally running
  SuperLink at address ``127.0.0.1:9093``

Connection configuration names must be unique and use the ``superlink.`` prefix.

**************************
 Listing your connections
**************************

You can list all your connection configurations using the ``flwr config list`` command.
Assuming the default configuration file shown earlier, the expected output will be:

.. code-block:: console

    $ flwr config list

    Flower Config file: /path/to/your/.flwr/config.toml
    SuperLink connections:
      local (default)
      local-poc

This shows you have two connection configurations available, with ``local`` set as the
default.

**************************
 Local Simulation Example
**************************

Local simulations allow you to test your Flower App on your own machine using virtual
SuperNodes instead of real distributed SuperNodes. This is useful for development and
testing before deploying to real distributed environments.

**Basic simulation configuration**

.. code-block:: toml

    [superlink.local]
    address = ":local:"

This creates a managed local SuperLink profile using the default simulation
configuration, which involves running 10 simulated SuperNodes through the simulation
runtime. Check the :doc:`Simulation Runtime docs <how-to-run-simulations>` to learn how
to customize the number of SuperNodes and resources (CPU/GPU) allocated to your
simulations.

.. _flower-config-local-in-memory:

**Simulation with in-memory local state**

.. code-block:: toml

    [superlink.local-in-memory]
    address = ":local-in-memory:"

This creates a managed local SuperLink profile that keeps its database in memory instead
of on disk. This can be useful on networked filesystems where SQLite locking or
performance issues can occur. The tradeoff is that local run history and logs are lost
when the managed local SuperLink is shut down.

**When to use each**

- Use the basic configuration for most local development and testing scenarios,
  especially when you want to keep a history of your runs and logs on disk.
- Use ``address = ":local-in-memory:"`` when the managed local SuperLink runs on a
  filesystem where SQLite performs poorly, such as some network drives or HPC storage

Learn more in the `How to Run Simulations
<https://flower.ai/docs/framework/how-to-run-simulations.html>`_ guide about other
optional parameters you can use to configure your local simulation.

When you use a local simulation profile with ``address = ":local:"`` or ``address =
":local-in-memory:"``, Flower CLI commands that communicate with SuperLink use the
Control API. Flower starts a managed local SuperLink automatically when needed and
reuses it across ``flwr run``, ``flwr list``, ``flwr log``, and ``flwr stop``. See
:doc:`how-to-run-flower-locally` for the full local workflow, background process
behavior, and runtime file locations.

***************************
 Remote Deployment Example
***************************

When you're ready to deploy your federated learning app to real distributed nodes, you
configure connections that point to a remote SuperLink.

**Example configuration**

.. code-block:: toml

    [superlink.remote-deployment]
    address = "<SUPERLINK-ADDRESS>:<PORT>"
    root-certificates = "/absolute/path/to/root/cert.crt"  # Optional, for TLS
    # insecure = true  # Disable TLS (not recommended for production)

**Explanation**

- ``address`` (required): The address of the SuperLink Control API to connect to (e.g.,
  ``my-server.example.com:9093``).
- ``root-certificates``: Path to the root certificate file for TLS encryption. This
  secures the communication between your CLI and the SuperLink. If omitted, Flower uses
  the default gRPC root certificate. This field is ignored if ``insecure`` is set to
  ``true``.
- ``insecure``: Set to ``true`` to disable TLS encryption (only use this for local
  testing, never in production). Defaults to ``false`` if omitted, meaning TLS is
  enabled by default.

**TLS (Transport Layer Security) explained**

TLS encrypts the communication between your local machine and the remote SuperLink to
prevent eavesdropping and tampering. In production, you should always use TLS by either:

- Providing a ``root-certificates`` file (recommended for custom certificates)
- Omitting both ``root-certificates`` and ``insecure`` to use default certificates

Only set ``insecure = true`` for local testing environments.

Refer to the `deployment documentation <https://flower.ai/docs/framework/deploy.html>`_
for TLS setup and advanced configurations.

********************************************
 Upgrading from previous versions of Flower
********************************************

**If you're new to Flower 1.26.0+, you can skip this section.**

For users upgrading from versions before 1.26.0: The ``federations`` section in your
``pyproject.toml`` file will be automatically migrated to the Flower Configuration the
first time you run a ``flwr`` command.

**What happens during migration**

1. The Flower Configuration file will be created at ``$HOME/.flwr/config.toml`` if not
   already existing
2. Your federation definitions are copied and renamed (``federations`` → ``superlink``)
3. The old ``[tool.flwr.federations]`` section in ``pyproject.toml`` is commented out
   for your reference

**After migration**

You can safely delete the commented-out ``federations`` section from your
``pyproject.toml`` file. All connection configurations now live in the Flower
Configuration and work across all your Flower projects.
