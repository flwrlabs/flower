FAB Format Version
==================

Flower apps can declare a Flower App Bundle (`FAB <https://flower.ai/docs/framework/ref-api-cli.html#flwr-build>`_) format version in :code:`pyproject.toml` under
:code:`[tool.flwr.app]`.

The FAB format version defines which build-time rules apply to the app.
This lets Flower evolve the FAB format without breaking legacy apps, while
allowing apps to opt into newer capabilities by adopting a newer FAB format
version.

The FAB build contract can evolve over time. For example, some rules can apply
to metadata in :code:`pyproject.toml`, while others can apply to the files that
must be included in the built FAB.

The FAB format version makes those rules explicit.


Current Versions
----------------

Flower currently recognizes the following FAB format versions:

- Missing :code:`fab-format-version` or :code:`fab-format-version = 0`: 
  Legacy behavior. Flower does not require newer FAB format fields and may
  derive limited compatibility metadata when possible.
- :code:`fab-format-version = 1`: 
  Strict validation for Flower version compatibility and license file handling.


FAB Format Version 0
--------------------

:code:`fab-format-version = 0` is the legacy behavior.

For version 0, declaring a minimum Flower version is not mandatory. However, if
the app declares a usable inclusive lower bound in the :code:`flwr` dependency,
Flower can derive a minimum Flower version from it and Flower Hub can use that
metadata during publish.


FAB Format Version 1
--------------------

For :code:`fab-format-version = 1`, Flower validates both app metadata and the
final FAB contents.

The app must declare:

- a :code:`flwr` dependency in :code:`[project].dependencies`
- an inclusive lower bound using :code:`>=`, such as :code:`flwr>=1.28.0`
- a :code:`flwr-version-target` in :code:`[tool.flwr.app]`
- a root-level license file reference in :code:`[project].license.file`

The app must also include the declared license file in the final FAB.

Example:

.. code-block:: toml

    [project]
    name = "my-federated-app"
    version = "0.1.0"
    description = "Federated training for medical image classification."
    license = { file = "LICENSE" }
    dependencies = ["flwr>=1.28.0"]

    [tool.flwr.app]
    publisher = "your-username"
    fab-format-version = 1
    flwr-version-target = "1.28.0"


Minimum Flower Version
~~~~~~~~~~~~~~~~~~~~~~

For :code:`fab-format-version = 1`, Flower derives a minimum supported version
from the inclusive lower bound declared in the :code:`flwr` dependency.

Example:

.. code-block:: toml

    dependencies = ["flwr>=1.28.0"]

This derives a minimum Flower version of :code:`1.28.0`.


Target Flower Version
~~~~~~~~~~~~~~~~~~~~~

:code:`flwr-version-target` identifies the Flower version the app is intended to
target.

For :code:`fab-format-version = 1`, it is required and must be greater than or
equal to the derived minimum Flower version.

Example:

.. code-block:: toml

    [tool.flwr.app]
    fab-format-version = 1
    flwr-version-target = "1.28.1"


License File Requirement
~~~~~~~~~~~~~~~~~~~~~~~~

For :code:`fab-format-version = 1`, the app must declare:

.. code-block:: toml

    [project]
    license = { file = "LICENSE" }

or:

.. code-block:: toml

    [project]
    license = { file = "LICENSE.md" }

The declared file must exist at the project root and must be included in the
final FAB.


How Flower Hub Uses This
------------------------

Flower Hub builds the FAB server-side during publish. During that build, Flower
validates the declared FAB format version and derives compatibility metadata from
the app definition.

This affects two user-facing flows:

- :code:`flwr app publish`:
  Invalid FAB format declarations fail during publish.
- :code:`flwr new @<publisher>/<app>` and :code:`flwr run @<publisher>/<app>`:
  Hub can use the app's compatibility metadata to reject incompatible Flower
  versions before the app is downloaded and run.
