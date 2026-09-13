#####
 FAQ
#####

This page collects answers to commonly asked questions about Federated Learning with
Flower.

.. dropdown:: :fa:`eye,mr-1` How can I run Federated Learning on a Raspberry Pi?

    Find the `blog post about federated learning on embedded device here <https://flower.ai/blog/2020-12-16-running_federated_learning_applications_on_embedded_devices_with_flower>`_ and the corresponding `GitHub code example <https://github.com/flwrlabs/flower/tree/main/examples/embedded-devices>`_.

.. dropdown:: :fa:`eye,mr-1` Does Flower support federated learning on Android devices?

    Yes, it does. Please take a look at our `blog post <https://flower.ai/blog/2021-12-15-federated-learning-on-android-devices-with-flower>`_ or check out the code examples:

    * `Android Kotlin example <https://flower.ai/docs/examples/android-kotlin.html>`_
    * `Android Java example <https://flower.ai/docs/examples/android.html>`_

.. dropdown:: :fa:`eye,mr-1` Can I combine federated learning with blockchain?

    Yes, of course. A list of available examples using Flower within a blockchain environment is available here:

    * `FLock: A Decentralised AI Training Platform <https://www.flock.io/#/>`_.
        * Contribute to on-chain training the model and earn rewards.
        * Local blockchain with federated learning simulation.
    * `Flower meets Nevermined GitHub Repository <https://github.com/nevermined-io/fl-demo/tree/master/image-classification-flower>`_.
    * `Flower meets Nevermined YouTube video <https://www.youtube.com/watch?v=A0A9hSlPhKI>`_.
    * `Flower meets KOSMoS <https://www.isw-sites.de/kosmos/wp-content/uploads/sites/13/2021/05/Talk-Flower-Summit-2021.pdf>`_.
    * `Flower meets Talan blog post <https://www.linkedin.com/pulse/federated-learning-same-mask-different-faces-imen-ayari/?trackingId=971oIlxLQ9%2BA9RB0IQ73XQ%3D%3D>`_ .
    * `Flower meets Talan GitHub Repository <https://gitlab.com/Talan_Innovation_Factory/food-waste-prevention>`_ .

.. dropdown:: :fa:`eye,mr-1` I see unexpected terminal output (e.g.: ``� □[32m□[1m``) on Windows. How do I fix this?

    .. _faq-windows-unexpected-output:

    If you see output (ANSI escape sequences or broken emojis) like this:

    - ``� □[32m□[1m``
    - ``□[0m□[96m□[1m``
    - ``�``

    this is usually a terminal host issue. Make sure you have installed the latest `Windows Terminal <https://aka.ms/terminal>`_ **application** (Microsoft's terminal app), and then run Flower commands there.

    To quickly check whether your current PowerShell session is running in Windows Terminal:

    .. code-block:: powershell

        echo $env:WT_SESSION

    If this prints a value (for example, ``b4c5f2c8-...``), you are in Windows Terminal.
    If it prints nothing, you are likely running in a non-Windows-Terminal host (for example, conhost), which can show raw ANSI escape codes or incorrect emoji rendering.

.. dropdown:: :fa:`eye,mr-1` I got SQL database errors (like ``Exception calling application: database is locked``) when running local simulations. What should I do?

    .. _faq-local-superlink-db-error:

    Local simulations run through a managed local SuperLink. By default, that local
    SuperLink stores its state in a SQLite database under ``$FLWR_HOME``. SQLite is reliable
    on a local file system, but it can perform poorly on networked filesystems such
    as NFS-mounted home directories or HPC cluster storage. In those environments, you
    might see errors such as ``database is locked`` or other SQLite-related failures.

    To avoid these issues, stop the background local SuperLink and switch the local
    connection to the in-memory mode in your Flower configuration:

    .. code-block:: toml

        [superlink.local]
        address = ":local-in-memory:"

    After that, start your local simulation again with ``flwr run``. Flower will launch
    the managed local SuperLink with an in-memory database instead of an on-disk SQLite
    database, which avoids filesystem locking issues. See
    :ref:`Flower Configuration <flower-config-local-in-memory>` for details.

    The tradeoff is that this mode is not persistent. When the managed local SuperLink
    stops, it loses its state, including run history and stored logs for previous runs.
    If you need persistence, prefer keeping ``$FLWR_HOME`` on a local disk instead of a
    network drive.
