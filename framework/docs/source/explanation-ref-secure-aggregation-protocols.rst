##############################
 Secure Aggregation Protocols
##############################

.. note::

    While this term might be used in other places, here it refers to a series of
    protocols, including ``SecAgg``, ``SecAgg+``, ``LightSecAgg``, ``FastSecAgg``, etc.
    This concept was first proposed by Bonawitz et al. in `Practical Secure Aggregation
    for Federated Learning on User-Held Data <https://arxiv.org/abs/1611.04482>`_.

Secure Aggregation protocols are used to securely aggregate model updates from multiple
clients while keeping the updates private. This is done by encrypting the model updates
before sending them to the server. The server can decrypt only the aggregated model
update without being able to inspect individual updates.

Flower now provides the ``SecAgg`` and ``SecAgg+`` protocols. While we plan to implement
more protocols in the future, one may also implement their own custom secure aggregation
protocol via low-level APIs.

************************************
 The ``SecAgg+`` protocol in Flower
************************************

The ``SecAgg+`` protocol is implemented using the ``SecAggPlusWorkflow`` in the
``ServerApp`` and the ``secaggplus_mod`` in the ``ClientApp``. The ``SecAgg`` protocol
is a special case of the ``SecAgg+`` protocol, and one may use ``SecAggWorkflow`` and
``secagg_mod`` for that.

You may find a detailed example in the `Secure Aggregation Example
<https://flower.ai/docs/examples/flower-secure-aggregation.html>`_. The documentation
for the ``SecAgg+`` protocol configuration is available at `SecAggPlusWorkflow
<https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html>`_.

The logic of the ``SecAgg+`` protocol is illustrated in the following sequence diagram:
the dashed lines represent communication over the network, and the solid lines represent
communication within the same process. The ``ServerApp`` is connected to ``SuperLink``,
and the ``ClientApp`` is connected to the ``SuperNode``; thus, the communication between
the ``ServerApp`` and the ``ClientApp`` is done via the ``SuperLink`` and the
``SuperNode``.

.. mermaid::

    sequenceDiagram
        participant ServerApp as ServerApp (in SuperLink)
        participant SecAggPlusWorkflow
        participant Mod as secaggplus_mod
        participant ClientApp as ClientApp (in SuperNode)

        ServerApp->>SecAggPlusWorkflow: Invoke

        note over SecAggPlusWorkflow,Mod: Stage 0: Setup
        SecAggPlusWorkflow-->>Mod: Send SecAgg+ configuration
        Mod-->>SecAggPlusWorkflow: Send public keys

        note over SecAggPlusWorkflow,Mod: Stage 1: Share Keys
        SecAggPlusWorkflow-->>Mod: Broadcast public keys
        Mod-->>SecAggPlusWorkflow: Send encrypted private key shares

        note over SecAggPlusWorkflow,ClientApp: Stage 2: Collect Masked Vectors
        SecAggPlusWorkflow-->>Mod: Forward the received shares
        Mod->>ClientApp: Fit instructions
        activate ClientApp
        ClientApp->>Mod: Updated model
        deactivate ClientApp
        Mod-->>SecAggPlusWorkflow: Send masked model parameters

        note over SecAggPlusWorkflow,Mod: Stage 3: Unmask
        SecAggPlusWorkflow-->>Mod: Request private key shares
        Mod-->>SecAggPlusWorkflow: Send private key shares

        SecAggPlusWorkflow->>SecAggPlusWorkflow: Unmask aggregated model
        SecAggPlusWorkflow->>ServerApp: Aggregated model
