Publish an App on Flower Hub
============================

Have you built an exciting federated app? Now it's time to share it with the world! Contributing your federated learning app to Flower Hub allows you to share your work with the community, receive feedback, and enable others to build on top of your contributions.

There are no restrictions on the type of federated apps you can publish. Whether you've developed:

- A new federated learning algorithm  
- A novel real-world application  
- A federated agent  
- An implementation accompanying a research paper  
- A benchmark  
- A tool that enhances the federated learning ecosystem  

We welcome your contribution!


Build Your App
--------------

If you're unsure what to contribute, start by exploring the `available apps on Flower Hub <https://flower.ai/apps>`_, and see if you can add a new app that is not yet present or if you can improve an existing one. A good starting point is `@flwrlabs/quickstart-pytorch <https://flower.ai/apps/flwrlabs/quickstart-pytorch/>`_, which provides a simple example of a Flower app structure.

Simulation vs. Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

Your app should run in both **Simulation** and **Deployment** runtimes **without requiring code changes**.

To achieve this, implement separate data-loading logic for each mode while keeping the training logic identical.

One way to distinguish between Simulation and Deployment is using :code:`context.node_config` inside your ClientApp. If the keys :code:`"partition-id"` and :code:`"num-partitions"` are present, the app is running in Simulation mode.

Example:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        batch_size = context.run_config["batch-size"]

        if (
            "partition-id" in context.node_config
            and "num-partitions" in context.node_config
        ):
            # Simulation Engine: partition data on the fly
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]
            trainloader, _ = load_sim_data(
                partition_id, num_partitions, batch_size
            )
        else:
            # Deployment Engine: load demo or real user data
            data_path = context.node_config["data-path"]
            trainloader, _ = load_local_data(data_path, batch_size)

        # Training logic continues identically for both modes


.. tip::
   Recommendations:

   - For Simulation, use `Flower Datasets <https://flower.ai/docs/datasets/index.html>`_ to partition data on the fly.
   - For Deployment, generate demo data using the CLI command :code:`flwr-datasets create` (see the `deployment data guide <https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html>`_ for details).

.. note::
   The app's README should clearly document the following:

   - For Simulation, specify how many virtual SuperNodes are expected and whether modifications to the `Flower Configuration <https://flower.ai/docs/framework/ref-flower-configuration.html>`_ are required.

   - For Deployment, explain how to generate synthetic data using :code:`flwr-datasets create`.  
     If the app relies on externally provided data, describe how users can obtain it and how it should be formatted before running the app.


Update App Metadata
~~~~~~~~~~~~~~~~~~~

Before publishing your app, ensure that its metadata is complete and accurate. This information is defined in :code:`pyproject.toml` and is used by Flower Hub to display and identify your application.

Under :code:`[project]`, provide:

- :code:`name` — the app's unique name. A valid app name must start with a letter and only contain letters, digits, and hyphens.
- :code:`version` — the current release version.
- :code:`description` — a short summary of the app.
- :code:`license` — the applicable software license.

Under :code:`[tool.flwr.app]`, specify:

- :code:`publisher` — your Flower account username

Example:

.. code-block:: toml

    [project]
    name = "my-federated-app"
    version = "0.1.0"
    description = "Federated training for medical image classification."
    license = "Apache-2.0"

    [tool.flwr.app]
    publisher = "your-username"  # Must match your Flower account username

License Field Formats
^^^^^^^^^^^^^^^^^^^^^

Flower Hub accepts :code:`[project].license` in the following forms:

.. code-block:: toml

    # SPDX identifier (string)
    license = "Apache-2.0"

    # Inline text
    license = { text = "Apache-2.0" }

    # File reference (must be at the project root)
    license = { file = "LICENSE" }
    # or
    license = { file = "LICENSE.md" }

Rules:

- :code:`file` and :code:`text` cannot be set together.
- If you use :code:`license.file`, it must be exactly :code:`LICENSE` or :code:`LICENSE.md`.
- The referenced license file must exist and be included in the files uploaded by :code:`flwr app publish`.


.. warning::
   The :code:`name` and :code:`description` are publicly visible on Flower Hub.
   Choose them carefully to ensure your app is clear, descriptive, and easy to discover.
   The :code:`name` **cannot be changed** after the first publication, so make sure it is final before releasing your app.

.. note::
   Flower Hub currently supports the following *content* file formats: :code:`.py`, :code:`.toml`, :code:`.md`, :code:`.yaml`, :code:`.yml`, :code:`.json`, and :code:`.jsonl`.
   In addition, certain non-content files are supported, including root-level license files :code:`LICENSE` and :code:`LICENSE.md`, as well as common dotfiles such as :code:`.gitignore` and :code:`.editorconfig`.
   Before publishing, ensure that all required files for your app (e.g., source code, metadata, README) are included in the app directory.

   Support for additional file formats is planned for future releases.


Understand Which Files Are Uploaded
------------------------------------

Flower Hub stores your app as a *project* — the full source that others can browse,
clone, and build on. This is intentionally broader than a FAB: project files like
:code:`.gitignore` and :code:`.editorconfig` are included so that anyone who pulls your
app can reproduce your development environment, even though those files are never
packaged into a FAB for a federation run. Think of what you upload as the source of
truth, and the FAB as the runtime-optimised subset derived from it.

When you run :code:`flwr app publish`, Flower collects files from your app directory,
filters them, and validates the result before sending anything to Flower Hub.

**Collecting files**

Flower recursively walks your app directory without following symlinks. Files more than
**10 directory levels** deep are not collected.

**Filtering**

**Allowed file types** (``APP_PUBLISH_INCLUDE_PATTERNS``):

These are the file types that make up a typical Flower app, plus a small set of
non-content files that are useful on Flower Hub (dotfiles and license files):

.. code-block:: text

    **/*.py           Python source files
    **/*.toml         TOML configuration files
    **/*.md           Markdown documentation
    **/*.yaml         YAML configuration files
    **/*.yml          YAML configuration files (alternate extension)
    **/*.json         JSON data files
    **/*.jsonl        JSON Lines data files
    /.gitignore       Root-level gitignore file
    **/.editorconfig  Editor configuration files
    /LICENSE          Root-level license file
    /LICENSE.md       Root-level license file (Markdown)

**Always excluded** (``APP_PUBLISH_EXCLUDE_PATTERNS``):

These paths are never uploaded — they are Flower internals and regenerated cache
directories that have no place on Flower Hub:

.. code-block:: text

    .flwr/**           Flower internal directory
    **/__pycache__/**  Python bytecode cache

After the type filter, your :code:`.gitignore` patterns are applied. Any file dropped at
this stage is printed as a warning so you can see exactly what was left out.

**Validation**

Once the file set is ready, Flower checks each file is valid UTF-8, then verifies:

- at most **1,000 files**
- no single file larger than **1 MB**
- total size no more than **10 MB**

If everything passes, you will see a confirmation before the upload begins. If any
check fails, Flower raises an error before sending anything to Flower Hub.

.. note::

    :code:`flwr app publish` uploads your source files directly — Flower Hub builds the
    FAB server-side. Publish rules and FAB packaging rules (:code:`flwr build`) are
    related but not identical: publish includes a few extra types like
    :code:`.gitignore` and :code:`.editorconfig` that are not bundled into a FAB.


Create a Flower Account
-----------------------

If you don't already have one, create a Flower account at: `https://flower.ai/ <https://flower.ai/>`_.

Click **Sign Up** in the top-right corner and follow the instructions. Make sure the username is the same as the publisher name defined in your app's :code:`pyproject.toml`.

Publishing on behalf of an organization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since organization accounts are not yet officially supported, please:

- Create a standard user account  
- Use your organization's name as the username (e.g., ``flwrlabs``)  
- Update your profile with the appropriate logo and description  

.. note::
   Organization accounts will be migrated once official organization support becomes available.


Publish Your App on Flower Hub
------------------------------

Apps are published using the Flower CLI.

First, ensure :code:`flwr` is installed:

.. code-block:: bash

    pip install flwr

Next, log in to your SuperLink connection that points to SuperGrid:

.. code-block:: bash

    flwr login supergrid

This will open a browser window where you can authenticate using your Flower account.

.. note::
   To login to SuperGrid it is expected that your Flower Configuration has a SuperLink connection
   named :code:`superlink.supergrid` with the address :code:`supergrid.flower.ai`. This connection should
   be present by default after intalling Flower. Read more about SuperLink connections and how to configure them in the
   `Flower Configuration documentation <https://flower.ai/docs/framework/ref-flower-configuration.html>`_.

After logging in, publish your app:

.. code-block:: bash

    flwr app publish <your-app-path>

.. note::
   :code:`flwr app publish` uploads your project files (source + metadata), not a prebuilt :code:`.fab` file.
   Flower Hub builds the FAB server-side from the uploaded project contents.
   This means publish upload rules and FAB packaging rules are related but not identical.
   For details on FAB packaging, see the Flower Framework CLI reference for :code:`flwr build`:
   `https://flower.ai/docs/framework/ref-api-cli.html <https://flower.ai/docs/framework/ref-api-cli.html>`_.

🎉 That's it! Your app is now live on Flower Hub.

You can view it at:

.. code-block:: text

    https://flower.ai/apps/<account_name>/<app_name>/


Publish a New Version of Your App
---------------------------------

Published apps cannot be removed from Flower Hub. However, you can release updated versions of your app to introduce improvements, new features, or bug fixes.

To publish a new version:

1. Make the necessary changes to your app's source code.
2. Update the :code:`version` field under :code:`[project]` in :code:`pyproject.toml`.

For example, if the current version is ``0.1.0``, you might update it to:

- ``0.1.1`` for a bug fix (patch release)
- ``0.2.0`` for new features (minor release)
- ``1.0.0`` for major changes

After updating the version number, publish the new release using the same command:

.. code-block:: bash

    flwr app publish <your-app-path>

The new version will be published alongside previous versions, allowing users to reference and run specific releases as needed.

We encourage you to share your app with colleagues and on social media to help grow the Flower Hub ecosystem and make federated AI more accessible to everyone.
