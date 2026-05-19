#############################
 Write your first Flower App
#############################

Welcome to the second part of the Flower collaborative AI tutorial!

In the previous tutorial, you created your first federation on SuperGrid using the
simulation runtime. This allowed you to experiment with simulated nodes, run an existing
Flower App on SuperGrid, and explore the dashboard to follow its progress and view its
logs.

In this tutorial, you'll learn about the core components in a Flower App and how to
write your own. You will use the `@flwrlabs/demo
<https://flower.ai/apps/flwrlabs/demo/>`__ app as a starting point and modify it to
create your own Flower App. By the end, you'll have your very own Flower App that you
can run on SuperGrid!

.. tip::

    `Star Flower on GitHub <https://github.com/flwrlabs/flower>`__ ⭐️ and join the
    Flower community on Flower Discuss and the Flower Slack to connect, ask questions,
    and get help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's get started! 🌼

*************
 Preparation
*************

In this tutorial, you'll edit the code of an existing Flower App that you'll pull from
the Flower Hub. In order to do that, you'll need to install ``flwr``, the Flower Python
package. What follows is a brief guide on how this can be done:

Installing dependencies
=======================

First, we install the Flower package ``flwr`` in a new Python environment.

.. code-block:: shell

    $ pip install -U "flwr[simulation]"

Then, use ``flwr new`` to fetch an existing Flower App from the Flower Hub. In this
case, you'll fetch the `@flwrlabs/demo <https://flower.ai/apps/flwrlabs/demo/>`__ app.

.. code-block:: shell

    $ flwr new @flwrlabs/demo

After running it you'll notice a new directory named ``demo`` has been created. It
should have the following structure:

.. code-block:: shell

    demo
    ├── quickstart_numpy
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

***********************
 Flower App Components
***********************

***************************
 Run your App on SuperGrid
***************************

************
 Next steps
************

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 2
<tutorial-series-use-a-federated-learning-strategy-pytorch>` goes into more depth about
strategies and all the advanced things you can build with them.
