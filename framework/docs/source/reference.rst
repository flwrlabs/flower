###########
 Reference
###########

************
 References
************

Information-oriented API reference and other reference material.

.. autosummary::
    :toctree: ref-api
    :template: autosummary/module.rst
    :caption: API reference
    :recursive:

    flwr

.. toctree::
    :maxdepth: 2

    ref-api-cli

.. toctree::
    :maxdepth: 1
    :caption: Reference docs

    ref-flower-configuration
    ref-example-projects
    ref-telemetry
    ref-changelog
    ref-flower-runtime-comparison
    ref-flower-network-communication
    ref-exit-codes-dir
    ref-faq

AgentApp Space messaging
=======================

Space AgentApps access :class:`flwr.agentapp.AgentSpace` through ``agent.space``.
Use ``get_messages(unread_only=True)`` to poll unread messages and ``send_message(content)``
to publish text. ``tools()`` and ``call()`` expose publishing to the model as
``send_space_message``. Runtime derives the Space and author from the authenticated Task.
All reads advance the agent's read cursor; legacy sessions return ``None`` for
``agent.space``.
