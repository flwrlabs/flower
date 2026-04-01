:og:description: Upgrade seamlessly to Flower 1.28 with this guide for transitioning your setup to the latest features and enhancements.
.. meta::
    :description: Upgrade seamlessly to Flower 1.28 with this guide for transitioning your setup to the latest features and enhancements.

########################
 Upgrade to Flower 1.28
########################

Welcome to the migration guide for updating your Flower Configuration to the latest
version, 1.28. This guide will walk you through the necessary steps to ensure a smooth
transition, including any changes in configuration syntax, new features, and best
practices for leveraging the latest capabilities of Flower. This guide is relevant for
users upgrading from versions 1.26 and 1.27.

Let's dive in!

********************
 Summary of changes
********************

With Flower 1.26, we introduced the Flower Configuration, a new system for managing
SuperLink connections that replaces the older ``federations`` section in the
``pyproject.toml`` file. This change allows for more flexible and reusable connection
configurations across all your Flower apps. You can read more about it in the
:doc:`Flower Configuration reference <ref-flower-configuration>` page.

Now with Flower 1.28, we have made further improvements to the Flower Configuration,
closing the syntax gap between SuperLink connections used for Flower simulations and
those used for running Flower with the Deployment Runtime. The result? A more unified
and consistent experience when defining SuperLink connections, regardless of the runtime
you are using. This means that the configuration of simulations (i.e. number of
`virtual` SuperNodes, client resources, etc.) is now defined at the ``SuperLink`` that
executes the simulation. Still, you can override these settings on a per-run basis if
desired.

This guide will help you:

- Understand how to migrate your existing connections for simulation in your Flower
  Configuration to the new syntax.
- Learn how to view and adjust the simulation configuration in your ``SuperLink``. You
  can do this either permanently or on a per-run basis.

Before we begin, ensure you have updated to Flower 1.28.

.. code-block:: shell

    $ pip install -U "flwr[simulation]"
    $ flwr --version # should show 1.28.x

**********************************
 Update your Flower Configuration
**********************************

The main update in your Flower Configuration is to remove or comment simulation-specific
settings (i.e. those starting with ``options.``). These settings are now defined at the
``SuperLink`` level and will apply to all simulation runs using that SuperLink.
Additionally, a new field ``address`` needs to be added. It will be used for launching a
managed ``SuperLink`` on demand. Let's see how to do this with an example. Use the
following steps to update your Flower Configuration:

1. Locate your Flower Configuration file. You can find it by running:

   .. code-block:: shell

       $ flwr config list

   This will show you the path to your Flower Configuration file and the SuperLink
   connections defined in it. Example output:

.. code-block:: shell

    Flower Config file: /path/to/your/.flwr/config.toml
    SuperLink connections:
      local (default)
      supergrid

2. Open the Flower Configuration file in a text editor. You should see something like
   this:

.. code-block:: toml
    :caption: config.toml

    [superlink]
    default = "local"

    [superlink.supergrid]
    address = "supergrid.flower.ai"

    [superlink.local]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 2
    options.backend.client-resources.num-gpus = 0.0

3. Remove or comment out the simulation-specific settings under the
   ``[superlink.local]`` section (and any other connections you may have added that also
   include ``options.`` fields). Then, add a new field containing ``address =
   ":local:"``. The updated configuration should look like this:

.. code-block:: toml
    :caption: config.toml
    :emphasize-lines: 8,9,10,11

    [superlink]
    default = "local"

    [superlink.supergrid]
    address = "supergrid.flower.ai"

    [superlink.local]
    address = ":local:"
    # options.num-supernodes = 10 # or remove this line
    # options.backend.client-resources.num-cpus = 2 # or remove this line
    # options.backend.client-resources.num-gpus = 0.0 # or remove this line

Introducing the ``address`` field with the value ``:local:`` allows Flower to start a
managed ``SuperLink`` on demand when you submit a run using this connection. Read more
about this in the :doc:`Run Flower Locally with a Managed SuperLink
<how-to-run-flower-locally>` guide.

***************************************
 Using custom settings for simulations
***************************************

.. note::

    This guide assumes you don't have a local SuperLink running. If you do, please stop
    it. If you aren't sure, please refer to the :ref:`section on stopping a local
    SuperLink <stop-background-local-superlink>`.

In the section above we removed the simulation-specific settings from the Flower
Configuration. Now, you may be wondering how to configure the number of virtual
SuperNodes or client resources for your simulations. First, you can inspect whether your
SuperLink connection is pointing to a managed local SuperLink ready for simulations by
running:

.. code-block:: shell

    $ flwr federation list

    📄 Listing federations...
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
    ┃  Federation   ┃                    Description                     ┃  Runtime   ┃ Status ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
    │ @none/default │ A federation for testing and development purposes. │ simulation │ active │
    └───────────────┴────────────────────────────────────────────────────┴────────────┴────────┘

Note the ``Runtime`` type is ``simulation``. Then, you can view the current simulation
configuration of your managed local SuperLink with the following command:

.. code-block:: shell

    $ flwr federation list --federation="@none/default"

                            Simulation Configuration
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
    ┃            Setting             ┃            Key            ┃ Value ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
    │ Number of Simulated SuperNodes │ num-supernodes            │    10 │
    ├────────────────────────────────┼───────────────────────────┼───────┤
    │ ClientApp Resources (CPUs)     │ client-resources-num-cpus │     2 │
    ├────────────────────────────────┼───────────────────────────┼───────┤
    │ ClientApp Resources (GPUs)     │ client-resources-num-gpus │   0.0 │
    ├────────────────────────────────┼───────────────────────────┼───────┤
    │ Backend Name                   │ backend                   │   ray │
    └────────────────────────────────┴───────────────────────────┴───────┘

As you can see, the current configuration is the same as the one we had before in the
Flower Configuration file ``config.toml``. To change any of these settings, refer to the
``Customize the Simulation Runtime`` section in the :doc:`Flower Simulation Runtime
reference <how-to-run-simulations>` guide.
