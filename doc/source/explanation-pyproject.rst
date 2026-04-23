#############################
 The ``pyproject.toml`` file
#############################

The ``tool.flwr`` section of the ``pyproject.toml`` file is composed of
multiple subsections

*********************
 The ``app`` section
*********************

``publisher``
=============

The username of the author of the Flower App. Required

.. code:: toml

   [tool.flwr.app]
   publisher = "flwrlabs"

``components``
==============

Requires both a ``serverapp`` and a ``clientapp`` reference. Required

Those references should be of the form ``<python_module>:<object_name>``

.. code:: toml

   [tool.flwr.app.components]
   serverapp = "mlxexample.server_app:app"
   clientapp = "mlxexample.client_app:app"

In the above example, the ``app`` object for the ``serverapp`` is
declared inside the the ``server_app`` module of the ``mlxexample``
package.

``config``
==========

The default config of a Flower App as described :doc:`here
<how-to-configure-flower-app>`. Optional

.. code:: toml

   [tool.flwr.app.config]
   num-server-rounds = 3
   local-epochs = 1
   lr = 0.1
   verbose = true
   run-name = "Small Run"

*****************************
 The ``federations`` section
*****************************

``default``
===========

The default federation that will be used to run the Flower App (if no
arguments are provided). Required

.. code:: toml

   [tool.flwr.federations]
   default = "local-simulation"

``<federation_name>``
=====================

Any federation name can be provided in order to declare its definition.
At least one federation name corresponding to the default one must be
declared, i.e, in the above case where ``default`` is set to
``"local-simulation"``, we need to at least declare
``tool.flwr.federations.local-simulation``.

.. code:: toml

   [tool.flwr.federations.local-simulation]
   # Federation definition

``<federation_name>.address``
-----------------------------

This is the address of the federation, i.e, the address of the
corresponding ``SuperExec``. Optional, if no address is provided, the
simulation engine will be used.

.. code:: toml

   [tool.flwr.federations.local-deployed]
   address = "127.0.0.1:9093"

``<federation_name>.options``
-----------------------------

This field is a table accepting arbitrary keys. Note that in simulation
mode, the ``num-supernodes`` key is required. This means that either
``tool.flwr.federations.<federation_name>.address`` is defined or
``tool.flwr.federations.<federation_name>.options.num-supernodes`` is.

Those options will be passed to the ``SuperExec`` plugin.

.. code:: toml

   [tool.flwr.federations.local-simulation]
   options.num-supernodes = 10
   options.foo = "bar"
