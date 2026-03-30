:og:description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components and runtime settings.
.. meta::
    :description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components and runtime settings.

##############################
 Configure ``pyproject.toml``
##############################

All Flower Apps need a ``pyproject.toml``. When you create a new Flower App using ``flwr
new``, a ``pyproject.toml`` file is generated. This file defines your app's
dependencies, and configuration setup.

A complete ``pyproject.toml`` file, for example, looks like this:

.. dropdown:: Example ``pyproject.toml``

    .. code-block:: toml

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "flower-app"
        version = "1.0.0"
        description = "A Flower app example"
        license = "Apache-2.0"
        dependencies = [
            "flwr[simulation]>=1.20.0",
            "numpy>=2.0.2",
        ]

        [tool.hatch.build.targets.wheel]
        packages = ["."]

        [tool.flwr.app]
        publisher = "your-name-or-organization"

        [tool.flwr.app.components]
        serverapp = "your_module.server_app:app"
        clientapp = "your_module.client_app:app"

        [tool.flwr.app.config]
        num-server-rounds = 3
        any-name-you-like = "any value supported by TOML"

Here are a few key sections to look out for:

*******************************
 App Metadata and Dependencies
*******************************

.. code-block:: toml

    [project]
    name = "your-flower-app-name"
    version = "1.0.0"
    description = ""
    license = "Apache-2.0"
    dependencies = [
        "flwr[simulation]>=1.20.0",
        "numpy>=2.0.2",
    ]

    [tool.flwr.app]
    publisher = "your-name-or-organization"

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

        These fields follow the standard ``pyproject.toml`` metadata format, commonly used by tools like ``uv``, ``poetry``, and others. Flower reuses these for configuration and packaging.

    - ``name``\*: The name of your Flower app.
    - ``version``\*: The current version of your app, used for packaging and distribution. Must follow Semantic Versioning (e.g., "1.0.0").
    - ``description``: A short summary of what your app does.
    - ``license``: The license your app is distributed under (e.g., Apache-2.0).
    - ``dependencies``\*: A list of Python packages required to run your app.
    - ``publisher``\*: The name of the person or organization publishing the app.

Specify the metadata, including the app name, version, etc., in these sections. Add any
Python packages your app needs under ``dependencies``. These will be installed when you
run:

.. code-block:: shell

    pip install -e .

****************
 App Components
****************

.. code-block:: toml

    [tool.flwr.app.components]
    serverapp = "your_module.server_app:app"
    clientapp = "your_module.client_app:app"

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

    - ``serverapp``\*: The import path to your ``ServerApp`` object.
    - ``clientapp``\*: The import path to your ``ClientApp`` object.

These entries point to your ``ServerApp`` and ``ClientApp`` definitions, using the
format ``<module>:<object>``. Only update these import paths if you rename your modules
or the variables that reference your ``ServerApp`` or ``ClientApp``.

*******************
 App Configuration
*******************

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3
    any-name-you-like = "any value supported by TOML"

Define configuration values that should be available to your app at runtime. You can
specify any number of key-value pairs in this section. All the configuration values in
this section are optional.

Access these values in your code using ``context.run_config``. For example:

.. code-block:: python

    server_rounds = context.run_config["num-server-rounds"]

.. tip::

    You can also override the ``run_config`` values by passing the ``--run-config`` flag
    followed by key-value pairs when executing ``flwr run``. See the
    |flwr_run_cli_link|_ for more details.

**************************
 Federation Configuration
**************************

.. note::

    What was previously called "federation config" for SuperLink connections in
    ``pyproject.toml`` has been renamed and moved. These settings are now **SuperLink
    connection configuration** and are defined in the Flower configuration file. Refer
    to the `Flower Configuration <ref-flower-configuration.html>`_ for more information.

.. |flwr_run_cli_link| replace:: ``flwr run`` CLI documentation

.. _flwr_run_cli_link: ref-api-cli.html#flwr-run
