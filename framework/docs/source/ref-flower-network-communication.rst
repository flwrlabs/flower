:og:description: The Flower Network Communication reference describes all mandatory and optional network connections in Flower federated AI systems.
.. meta::
    :description: The Flower Network Communication reference describes all mandatory and optional network connections in Flower federated AI systems.

##############################
 Flower Network Communication
##############################

This reference complements the `Flower Architecture
<explanation-flower-architecture.html>`_ explanation by detailing the network
connections used in a deployed Flower federated AI system.

.. note::

    Optionally, a third connection to a third-party service can be established to
    provide user-level authentication via `OIDC
    <https://openid.net/developers/how-connect-works/>`_. This means that only users
    authenticated via ``flwr login`` are able to interface with the SuperLink.

.. raw:: html

    <div id="diagram1" style="display:block;">
        <img class="themed-image"
             data-light="./_static/flower-network-diagram-subprocess-light.svg"
             data-dark="./_static/flower-network-diagram-subprocess-dark.svg"
             alt="Flower Network Diagram (subprocess)">
    </div>
    <div id="diagram2" style="display:none;">
        <img class="themed-image"
             data-light="./_static/flower-network-diagram-process-light.svg"
             data-dark="./_static/flower-network-diagram-process-dark.svg"
             alt="Flower Network Diagram (process)">
    </div>
    <div style="text-align: center; margin-bottom: 1em;">
        <button onclick="document.getElementById('diagram1').style.display='block'; document.getElementById('diagram2').style.display='none';">Subprocess Mode</button>
        <button onclick="document.getElementById('diagram1').style.display='none'; document.getElementById('diagram2').style.display='block';">Process Mode</button>
    </div>

    <script>
      function currentTheme() {
        const t = document.body.dataset.theme || "auto";
        if (t === "dark") return "dark";
        if (t === "light") return "light";
        // auto → follow system
        return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      }

      function updateThemedImages() {
        const theme = currentTheme();
        document.querySelectorAll("img.themed-image").forEach((img) => {
          img.src = img.dataset[theme];
        });
      }

      document.addEventListener("DOMContentLoaded", () => {
        updateThemedImages();

        // Update whenever the theme toggle is clicked
        document.querySelectorAll(".theme-toggle").forEach((btn) => {
          btn.addEventListener("click", () => {
            requestAnimationFrame(updateThemedImages);
          });
        });
      });
    </script>

.. tip::

    Click the buttons above to toggle between the network diagrams for isolation modes
    **subprocess** and **process**.

*******************************
 Mandatory Network Connections
*******************************

Deployed Flower systems have at least two types of network connections:

- **CLI to SuperLink (Control API)**: The ``flwr`` `CLI command <ref-api-cli.html>`_,
  typically run on the users workstation, is used to interface with a deployed Flower
  federation consisting of SuperLink and SuperNodes. From a networking perspective, the
  ``flwr`` CLI acts as a gRPC client and the SuperLink acts as a gRPC server. The
  ``flwr`` CLI is the only way for a user (AI researchers, data scientist) to interface
  with a deployed Flower federation. They cannot, for example, interface directly with
  SuperNodes connected to the SuperLink. The ``flwr`` CLI to SuperLink connection should
  always use TLS, but ``insecure`` mode is supported for local testing.
- **SuperNode to SuperLink (Fleet API)**: In Flower terminology, a Flower federation is
  a set of SuperNodes connected to the same SuperLink. From a networking perspective,
  each SuperNode acts as a gRPC client and the SuperLink acts as a gRPC server. This
  means that, when deploying a SuperNode, only outgoing connections are necessary to
  connect to the SuperLink. Only the SuperNodes can initiate such requests and they do
  not respond to incoming requests. The SuperNode to SuperLink connection should always
  use TLS (see :doc:`how-to-enable-tls-connections` to learn more), but ``insecure``
  mode is supported for local testing.

******************************
 Optional Network Connections
******************************

Depending on the SuperLink and SuperNode configuration, Flower systems can have/use a
number of additional network connections.

Flower Components APIs
======================

All Flower components — SuperLink, SuperNode, SuperExec, ``ServerApp`` process, and
``ClientApp`` process — expose APIs to interact with other Flower components. The
SuperLink component includes three such APIs: the ServerAppIo API, Fleet API, and the
Control API. Similarly, the SuperNode component includes the ClientAppIo API. Each of
these APIs serves a distinct purpose when running a Flower app using the deployment
runtime, as summarized in the table below.

.. list-table::
    :widths: 25 25 35 65
    :header-rows: 1

    - - Component
      - Default Port
      - API
      - Purpose
    - - SuperLink
      - 9091
      - ServerAppIo API
      - Used by the SuperExec and the ``ServerApp`` processes
    - -
      - 9092
      - Fleet API
      - Used by the SuperNodes
    - -
      - 9093
      - Control API
      - Users interface with the SuperLink via this API using the `FlowerCLI
        <ref-api-cli.html>`_
    - - SuperNode
      - 9094
      - ClientAppIo API
      - Used by the SuperExec and the ``ClientApp`` processes

.. note::

    AppIo APIs enforce runtime version compatibility between the API server and its
    callers. By default, requests from callers using an incompatible major.minor runtime
    version are rejected before AppIo communication proceeds. Older callers without
    runtime metadata are currently accepted by this compatibility check for backward
    compatibility. Keep the SuperLink, SuperNode, SuperExec, ``ServerApp``, and
    ``ClientApp`` runtime components on compatible Flower versions.

Isolation Mode
==============

Both SuperLink and SuperNode can operate in different isolation modes. The **SuperExec**
is responsible for scheduling, launching, and managing app processes, such as the
``ServerApp`` process and the ``ClientApp`` process.

The ``subprocess`` isolation mode configures the SuperLink/SuperNode to automatically
run the SuperExec as a subprocess upon start. The ``process`` isolation mode, by
contrast, expects the SuperExec to run in a separately managed external process, so the
SuperLink/SuperNode will not launch one automatically. This enables, for example,
running the SuperLink/SuperNode and SuperExec in separate Docker containers with
different dependency sets, or running them on different servers within the same network.
See the :doc:`docker/index` guide for a deeper understanding of how to use both modes.

When using the ``process`` isolation mode, additional network connections are necessary
to allow the external process running the SuperExec, ``ServerApp``, or ``ClientApp`` to
communicate with the SuperLink or SuperNode:

- **SuperExec/ServerApp process to SuperLink (ServerAppIO API)**: Both the SuperExec for
  ``ServerApp``\s and the ``ServerApp`` processes act as gRPC clients and connect to the
  SuperLink's ServerAppIO API. This connection enables the SuperExec to discover runs to
  launch and the ``ServerApp`` process to pull the necessary inputs to execute the
  ``ServerApp``. It also allows the ``ServerApp``, once running, to do typical things
  like sending/receiving messages to/from available SuperNodes (via the SuperLink).
- **SuperExec/ClientApp process to SuperNode (ClientAppIO API)**: Both the SuperExec for
  ``ClientApp``\s and the ``ClientApp`` processes act as gRPC clients and connect to the
  SuperNode's ClientAppIO API. This connection enables the SuperExec to discover runs to
  launch and the ``ClientApp`` process to pull the necessary details (e.g., FAB file) to
  execute the ``ClientApp``, execute the ``ClientApp`` (e.g., local model training), and
  return the execution results (e.g., locally update model parameters) to the SuperNode.

.. note::

    The AppIo links above can run with plaintext communication or with
    server-authenticated TLS. Without the ``--appio-ssl-*`` options, these links remain
    unencrypted and should stay inside a trusted network:

    - SuperLink + SuperExec + ``ServerApp`` process
    - SuperNode + SuperExec + ``ClientApp`` process

    To secure these links, configure the ServerAppIo and ClientAppIo API servers with
    ``--appio-ssl-certfile``, ``--appio-ssl-keyfile``, and ``--appio-ssl-ca-certfile``.
    AppIo clients verify server certificates with ``--root-certificates``. In
    ``subprocess`` isolation mode, the SuperLink and SuperNode pass the CA path to the
    SuperExec processes they launch. This is not mTLS. See
    :doc:`how-to-enable-tls-connections` for concrete commands.

.. warning::

    When running without TLS, each group must remain inside a single trusted network.
    They should never communicate with each other over untrusted networks (e.g., the
    public internet).

Account Authentication
======================

When account authentication is enabled, Flower uses an OIDC-compatible server to
authenticate requests:

- **SuperLink to OIDC server**: A SuperLink can optionally be configured to only allow
  authenticated users to interact with it. In this setting, the Flower SuperLink acts as
  a REST client to the OIDC-compatible server.

Application-specific Connections
================================

Users who write Flower Apps (``ServerApp`` and ``ClientApp``) can also make additional
network requests. This is, strictly speaking, not part of Flower as a Federated AI
Platform. It is a decision of (a) the user about what kinds of third-party systems their
Flower App should connect to and (b) the system administrator about what kinds of
connections they want to allow.

Typical examples include:

- **ClientApp to Database**: ``ClientApp`` instances typically need to be able to access
  the data to perform the action they have been designed for (e.g. train locally a
  model, run a DB query). How this connection is established depends on what storage
  technology is used at the client side. Note that in the diagram above, we show two
  representative connections to DBs in Client-A and Client-B. Your DB connection(s) may
  likely be different to the illustration above.
- **ServerApp to Database**: ``ServerApp`` instances might want to access the data to
  perform the action they have been designed for (e.g. evaluate a model on some data
  after aggregation). How this connection is established depends on what storage
  technology used at the client side. Note that in the diagram above we have omitted
  showing a DB connected to the ``ServerApp`` components.
- **ServerApp to metric logging service**: Metric logging services like TensorBoard,
  MLFlow and Weights & Biases are often used to track the progress of training runs. In
  this setting, the ``ServerApp`` typically acts as a client to the metric logging
  service.

Communication Model
===================

During real-world deployment, the push/pull communication model adopted by each
component can influence decisions related to resource provisioning, scaling, monitoring,
and reliability. To support such decisions, the list below outlines the communication
model used between the Flower components:

- **SuperLink ↔ SuperNode (Fleet API)**: The SuperNode pulls/pushes Messages from/to the
  SuperLink via the Fleet API. The SuperNode also pulls the FAB if a new run is being
  executed.
- **SuperLink ↔ ServerApp (ServerAppIo API)**: The ``ServerApp`` process pulls/pushes
  Messages from/to the SuperLink via the ServerAppIo API. The ``ServerApp`` also pulls
  the FAB as part of the first interaction with the SuperLink, and at the end of the
  execution it pushes the Context back to the SuperLink.
- **SuperNode ↔ ClientApp (ClientAppIo API)**: The ``ClientApp`` process pulls/pushes
  Messages from/to the SuperNode via the ClientAppIo API. The ``ClientApp`` also pulls
  the FAB as part of the first interaction with the SuperNode, and at the end of the
  execution it pushes the Context back to the SuperNode.
