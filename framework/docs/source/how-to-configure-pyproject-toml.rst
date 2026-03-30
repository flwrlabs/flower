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
        :substitutions:

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "flower-app"
        version = "1.0.0"
        description = "A Flower app example"
        license = "Apache-2.0"
        dependencies = [
            "flwr[simulation]>=|stable_flwr_version|",
            "numpy>=2.0.2",
        ]

        [tool.hatch.build.targets.wheel]
        packages = ["."]

        [tool.flwr.app]
        publisher = "your-name-or-organization"
        fab-include = ["path/to/include_file.py"]  # Optional
        fab-exclude = ["path/to/exclude_file.py"]  # Optional

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
    :substitutions:

    [project]
    name = "your-flower-app-name"
    version = "1.0.0"
    description = ""
    license = "Apache-2.0"
    dependencies = [
        "flwr[simulation]>=|stable_flwr_version|",
        "numpy>=2.0.2",
    ]

    [tool.flwr.app]
    publisher = "your-name-or-organization"
    fab-include = ["path/to/include_file.py"]  # Optional
    fab-exclude = ["path/to/exclude_file.py"]  # Optional

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
    - ``fab-include``: A list of gitignore-style patterns to include in the Flower App Bundle. See `Defining Included/Excluded Files`_ for details.
    - ``fab-exclude``: A list of gitignore-style patterns to exclude from the Flower App Bundle. See `Defining Included/Excluded Files`_ for details.

Specify the metadata, including the app name, version, etc., in these sections. Add any
Python packages your app needs under ``dependencies``. These will be installed when you
run:

.. code-block:: shell

    pip install -e .

**********************************
 Defining Included/Excluded Files
**********************************

The ``fab-include`` and ``fab-exclude`` fields let you control which files end up in
your Flower App Bundle (FAB) — the package that carries your application code to the
SuperLink and SuperNodes.

Both fields are optional. When omitted, Flower uses sensible `built-in defaults
<built-in defaults for fab included/excluded files_>`_ that include common source files
and the top-level ``LICENSE`` file (``/LICENSE``) while excluding virtual environments,
build artifacts, ``__pycache__`` directories, and test files.

.. code-block:: toml

    [tool.flwr.app]
    publisher = "your-name-or-organization"
    fab-include = ["src/**/*.py", "conf/*.yaml"]    # Optional
    fab-exclude = ["src/scratch.py"]                # Optional

.. dropdown:: Understanding each field

    - ``fab-include``: A list of gitignore-style patterns that narrows the FAB to only
      the files you want. When set, Flower starts with an empty candidate set and adds
      only files that match at least one of your patterns. The built-in constraints are
      then applied on top — if any of your patterns would pull in an unsupported file
      type (e.g. ``.txt`` or a binary), Flower raises an error listing the offending
      files so you can fix or remove the pattern.
    - ``fab-exclude``: A list of gitignore-style patterns that removes specific files
      from the FAB. Files that match at least one of your patterns are dropped before
      the built-in constraints run, so you can safely exclude anything the defaults
      would otherwise keep.

    .. note ::

        Both fields are optional. Omit a field entirely to rely on the built-in
        defaults — setting it to an empty list is an error. Every pattern you provide
        must match at least one file; Flower raises an error at build time for
        unresolved patterns, keeping typos from silently changing your bundle.

Flower applies filtering in two stages:

1. **Publish filter** — Files are first narrowed to supported types, and any patterns in
   your ``.gitignore`` are applied to remove ignored files. Refer to the `Flower Hub
   documentation <https://flower.ai/docs/hub/how-to-publish-app-on-hub.html>`_ for more
   details on how this works when you publish an app to Flower Hub.
2. **FAB filter** — Your ``fab-include`` and ``fab-exclude`` patterns are applied next,
   followed by non-overridable built-in constraints that enforce supported file types
   and exclude directories like ``.venv/`` or ``__pycache__/``.

In short, ``fab-include`` and ``fab-exclude`` give you fine-grained control *within* the
boundaries of what Flower supports. You cannot use them to include unsupported file
types (e.g., ``.txt``) — Flower will flag any such conflicts with a clear error message.

.. dropdown:: Example: Bundling only your source package and a config file

    Suppose your project looks like this::

        my-flower-app/
        ├── pyproject.toml
        ├── README.md
        ├── conf/
        │   └── config.yaml
        └── your_module/
            ├── client_app.py
            ├── server_app.py
            └── scratch.py      ← you want to exclude this in the FAB

    Add the following to your ``pyproject.toml``:

    .. code-block:: toml

        [tool.flwr.app]
        publisher = "your-name-or-organization"
        fab-include = ["your_module/**/*.py", "conf/*.yaml"]
        fab-exclude = ["your_module/scratch.py"]

    When you execute |flwr_run_cli_link|_ or |flwr_build_cli_link|_, the resulting FAB
    will contain ``pyproject.toml``, ``your_module/client_app.py``,
    ``your_module/server_app.py``, and ``conf/config.yaml`` — but not
    ``your_module/scratch.py`` or ``README.md``.

***************************************************
 Built-in Defaults for FAB Included/Excluded Files
***************************************************

A FAB is a structured package consumed by the SuperLink and SuperNodes, and they only
know how to work with certain file types. Allowing arbitrary file types into a FAB would
not only break compatibility across the federation and bloat bundle sizes, but also
create a security risk — sensitive files like credentials, private keys, or environment
configs could accidentally be packaged and distributed to every SuperNode in the
federation. That is why Flower enforces a set of built-in patterns on top of whatever
you put in ``fab-include`` or ``fab-exclude`` — and these cannot be overridden. They are
defined in ``flwr.common.constant`` as ``FAB_INCLUDE_PATTERNS`` and
``FAB_EXCLUDE_PATTERNS``.

**Allowed file types** (``FAB_INCLUDE_PATTERNS``):

These are the file types that make up a typical Flower app — source code, configuration,
documentation, and data descriptors. Anything outside this set (for example, ``.txt`` or
binary files) is not a recognised FAB file type and cannot be included:

.. code-block:: text

    **/*.py        Python source files
    **/*.toml      TOML configuration files
    **/*.md        Markdown documentation
    **/*.yaml      YAML configuration files
    **/*.yml       YAML configuration files (alternate extension)
    **/*.json      JSON data files
    **/*.jsonl     JSON Lines data files
    /LICENSE       Top-level license file

**Always excluded** (``FAB_EXCLUDE_PATTERNS``):

These are paths that should never travel across the network: generated artefacts that
can be reproduced locally (caches, build outputs, packaging directories), virtual
environments that are machine-specific, test files that are not needed at runtime, and
Flower's own internal directory:

.. code-block:: text

    .flwr/**           Flower internal directory
    **/__pycache__/**  Python bytecode cache
    pyproject.toml     Re-serialized separately; the original is never bundled as-is
    **/*_test.py       Test files
    **/test_*.py       Test files
    build/**           Build output
    eggs/**            Egg build artifacts
    .eggs/**
    lib/**
    lib64/**
    parts/**
    *.egg
    .venv/**           Virtual environments
    env/**
    venv/**
    ENV/**
    env.bak/**
    venv.bak/**

.. note::

    If you use ``fab-include`` to add a file that does not match any of the built-in
    include patterns (for example, a ``.txt`` file), Flower will raise an error and list
    the conflicting files. The fix is to remove those patterns from ``fab-include``.
    Files that *do* match the built-in includes but are also matched by the built-in
    excludes (for example, a ``*.py`` file inside ``.venv/``) are silently dropped —
    this is expected behaviour.

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
    |flwr_run_cli_link|_ CLI documentation for more details.

**************************
 Federation Configuration
**************************

.. note::

    What was previously called "federation config" for SuperLink connections in
    ``pyproject.toml`` has been renamed and moved. These settings are now **SuperLink
    connection configuration** and are defined in the Flower configuration file. Refer
    to the `Flower Configuration <ref-flower-configuration.html>`_ for more information.

.. |flwr_run_cli_link| replace:: ``flwr run``

.. |flwr_build_cli_link| replace:: ``flwr build``

.. _flwr_build_cli_link: ref-api-cli.html#flwr-build

.. _flwr_run_cli_link: ref-api-cli.html#flwr-run
