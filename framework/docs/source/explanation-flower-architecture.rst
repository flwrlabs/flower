:og:description: Learn about the architecture of a deployed Flower federated learning system, including SuperLink, SuperNodes, multi-tenancy, and the role of SuperExec for managing concurrent training runs.
.. meta::
    :description: Learn about the architecture of a deployed Flower federated learning system, including SuperLink, SuperNodes, multi-tenancy, and the role of SuperExec for managing concurrent training runs.

#####################
 Flower Architecture
#####################

This page explains the architecture of deployed Flower federated learning system.

In federated learning (FL), there is typically one server and a number of clients that
are connected to the server. This is often called a federation.

The role of the server is to coordinate the training process. The role of each client is
to receive tasks from the server, execute those tasks and return the results back to the
server.

This is sometimes called a hub-and-spoke topology:

.. image:: ./_static/flower-architecture-hub-and-spoke-light.png
    :width: 400
    :alt: Hub-and-spoke topology in federated learning
    :class: no-scaled-link only-light

.. figure:: ./_static/flower-architecture-hub-and-spoke-dark.png
    :width: 400
    :alt: Hub-and-spoke topology in federated learning (dark theme)
    :class: no-scaled-link only-dark

    Hub-and-spoke topology in federated learning (one server, multiple clients).

In a real-world deployment, we typically want to run different projects on such a
federation. Each project could use different hyperparameters, different model
architectures, different aggregation strategies, or even different machine learning
frameworks like PyTorch and TensorFlow.

This is why, in Flower, both the server side and the client side are split into two
parts. One part is long-lived and responsible for communicating across the network, the
other part is short-lived and executes task-specific code.

A Flower *server* is composed of **SuperLink**, **SuperExec**, and ``ServerApp``:

- **SuperLink**: A long-running process that forwards task instructions to clients
  (SuperNodes) and receives task results back.
- **SuperExec**: A long-running process that schedules, launches, and manages multiple
  app processes (e.g., ``ServerApp``) on demand by communicating with the SuperLink. By
  default, it is started automatically by the SuperLink.
- ``ServerApp``: A short-lived app process containing project-specific code that
  customizes all server-side aspects of federated learning systems (client selection,
  client configuration, result aggregation). This is the component AI researchers and
  engineers implement when building Flower apps.

A Flower *client* is composed of **SuperNode**, **SuperExec**, and ``ClientApp``:

- **SuperNode**: A long-running process that connects to the SuperLink, requests tasks,
  executes them (for example, "train this model on your local data"), and returns the
  results to the SuperLink.
- **SuperExec**: A long-running process that schedules, launches, and manages multiple
  application processes (e.g., ``ClientApp``) on demand by communicating with the
  SuperNode. By default, it is started automatically by the SuperNode.
- ``ClientApp``: A short-lived app process containing project-specific code that
  customizes all client-side aspects of federated learning systems (local model
  training, evaluation, pre- and post-processing). This is the component AI researchers
  and engineers implement when building Flower apps.

Why SuperNode and SuperLink? Well, in federated learning, the clients are the actual
stars of the show. They hold the training data and they run the actual training. This is
why Flower decided to name them **SuperNode**. The **SuperLink** is then responsible for
acting as the *missing link* among all those SuperNodes.

.. image:: ./_static/flower-architecture-basic-architecture-light.png
    :width: 600
    :alt: Basic Flower architecture
    :class: no-scaled-link only-light

.. figure:: ./_static/flower-architecture-basic-architecture-dark.png
    :width: 600
    :alt: Basic Flower architecture (dark theme)
    :class: no-scaled-link only-dark

    The basic Flower architecture for federated learning.

In a Flower app project, users will typically develop the ``ServerApp`` and the
``ClientApp``. All the network communication between *server* and *clients* is taken
care of by the SuperLink and SuperNodes.

.. tip::

    For more details, please refer to the |serverapp_link|_ and |clientapp_link|_
    documentation.

With *multi-run*, multiple ``ServerApp``\s and ``ClientApp``\s can run within the same
federation, which consists of a single long-running SuperLink and multiple long-running
SuperNodes. This capability is sometimes referred to as *multi-tenancy* or *multi-job*.

As shown in the figure below, two Flower App projects, each consisting of a
``ServerApp`` and a ``ClientApp``, could share the same SuperLink and SuperNodes.

.. image:: ./_static/flower-architecture-multi-run-light.png
    :width: 600
    :alt: Multi-run federated learning architecture
    :class: no-scaled-link only-light

.. figure:: ./_static/flower-architecture-multi-run-dark.png
    :width: 600
    :alt: Multi-run federated learning architecture (dark theme)
    :class: no-scaled-link only-dark

    Multi-run federated learning architecture with Flower

To illustrate how multi-run works, consider one federated learning training run where a
``ServerApp`` and a ``ClientApp`` are participating in ``[run 1]``. Note that a
SuperNode will only run a ``ClientApp`` if it is selected to participate in the training
run.

In ``[run 1]`` below, all the SuperNodes are selected and therefore run their
corresponding ``ClientApp``\s:

.. image:: ./_static/flower-architecture-multi-run-1-light.png
    :width: 600
    :alt: Multi-tenancy federated learning architecture - Run 1
    :class: no-scaled-link only-light

.. figure:: ./_static/flower-architecture-multi-run-1-dark.png
    :width: 600
    :alt: Multi-tenancy federated learning architecture - Run 1 (dark theme)
    :class: no-scaled-link only-dark

    Run 1 in a multi-run federated learning architecture with Flower. All SuperNodes
    participate in the training round.

However, in ``[run 2]``, only the first and second SuperNodes are selected to
participate in the training:

.. image:: ./_static/flower-architecture-multi-run-2-light.png
    :width: 600
    :alt: Multi-tenancy federated learning architecture - Run 2
    :class: no-scaled-link only-light

.. figure:: ./_static/flower-architecture-multi-run-2-dark.png
    :width: 600
    :alt: Multi-tenancy federated learning architecture - Run 2 (dark theme)
    :class: no-scaled-link only-dark

    Run 2 in a multi-run federated learning architecture with Flower. Only the first and
    second SuperNodes are selected to participate in the training round.

Therefore, with Flower multi-run, different Flower App projects can run on different
sets of clients.

.. note::

    This explanation covers the Flower Deployment Runtime. An explanation covering the
    Flower Simulation Runtime will follow.

.. important::

    As we continue to enhance Flower at a rapid pace, we'll periodically update this
    explainer document. Feel free to share any feedback with us.

.. |clientapp_link| replace:: ``ClientApp``

.. |serverapp_link| replace:: ``ServerApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _serverapp_link: ref-api/flwr.server.ServerApp.html
