:og:description: Guide to manage Flower federations using the Deployment Runtime.
.. meta::
    :description: Guide to manage Flower federations using the Deployment Runtime.

.. |flower_cli_federation_link| replace:: ``Flower CLI``

.. _flower_cli_federation_link: ref-api-cli.html#flwr-federation

.. note::

    Flower Federation management is a new feature introduced in Flower 1.24.0. It gains
    new functionality in subsequent releases. Changes to the functionality described in
    this guide are also expected as the features provided via the ``flwr federation``
    commands mature.

###########################
 Manage Flower Federations
###########################

A Flower federation is comprised of a set of users and some or all the SuperNodes they
own that are registered with the same SuperLink. Members of a Flower federation can
execute runs (e.g. to federate the training of an AI model) across all SuperNodes that
are part of it.

Federations can be addressed by their federation ID, which has the form
``@<account-name>/<federation-name>``. The ``<account-name>`` is the name of the account
that created the federation, and ``<federation-name>`` is the name of the federation.

In this how-to guide, you will:

- Learn how to see the federations you are part of.
- Learn how to display information about a specific federation.

.. note::

    This guide covers inspecting federations through the Flower CLI when using a
    self-hosted SuperLink. To create and manage federations in the SuperGrid dashboard,
    see :doc:`how-to-create-and-manage-federations`.

******************
 List Federations
******************

With the |flower_cli_federation_link|_, you can easily inspect the federations your
Flower account is part of:

.. code-block:: shell

    $ flwr federation list

The above command will display a table with a row for each federation you are part of.
In this case there is one federation: ``@none/default``. Its federation name is
``default``, and it belongs to the ``none`` account:

.. code-block:: shell

    📄 Listing federations...
    ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┓
    ┃  Federation   ┃                    Description                     ┃  Runtime   ┃ Status ┃
    ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━┩
    │ @none/default │ A federation for testing and development purposes. │ deployment │ active │
    └───────────────┴────────────────────────────────────────────────────┴────────────┴────────┘

**********************
 Inspect a Federation
**********************

You can inspect a specific federation by passing its federation ID to ``flwr federation
list``. With this command, you will be able to see the following information about a
federation:

- The members of the federation.
- The SuperNodes registered with the federation and their status.
- The runs executed via the federation.

The ``flwr federation list --federation <federation-id>`` command expects the federation
ID as an argument:

.. code-block:: shell

    $ flwr federation list local-deployment --federation="@none/default"

Then, assuming that there are two ``SuperNodes`` connected and that three runs have been
submitted through the federation, a representative output would be similar to:

.. code-block:: shell

    📄 Showing ' @none/default' federation ...
        Federation Members
    ┏━━━━━━━━━━━━━━┳━━━━━━━━┓
    ┃ Account Name ┃  Role  ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━┩
    │ none         │ Member │
    └──────────────┴────────┘
           SuperNodes in the Federation
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
    ┃       Node ID        ┃ Owner ┃ Status ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
    │ 1277309880252492806  │ none  │ online │
    ├──────────────────────┼───────┼────────┤
    │ 13280365719060659445 │ none  │ online │
    └──────────────────────┴───────┴────────┘
                                Runs in the Federation
    ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃       Run ID        ┃             App            ┃       Status       ┃ Elapsed  ┃
    ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
    │ 6665860355925098787 │ @flwrlabs/vision==1.0.0    │ finished:completed │ 00:00:24 │
    ├─────────────────────┼────────────────────────────┼────────────────────┼──────────┤
    │ 6896250833792831197 │ @flwrlabs/analytics==2.0.0 │ finished:stopped   │ 00:00:08 │
    ├─────────────────────┼────────────────────────────┼────────────────────┼──────────┤
    │ 3918106370412458251 │ @flwrlabs/llm==1.5.0       │ running            │ 00:00:02 │
    └─────────────────────┴────────────────────────────┴────────────────────┴──────────┘

Note how the ``SuperNodes`` table shows a subset of the information available via the
command ``flwr supernode list`` (Learn more about this command in the
:doc:`how-to-authenticate-supernodes` guide). Similarly, the ``Runs`` table shows a
subset of the information available via the ``flwr list`` command.
