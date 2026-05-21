:og:description: Create and manage Flower federations in SuperGrid.
.. meta::
    :description: Create and manage Flower federations in SuperGrid.

###############################
 Create and Manage Federations
###############################

This guide shows how to create and manage federations in SuperGrid. A federation defines
the group of users and SuperNodes that can take part in Flower runs.

SuperGrid supports two kinds of federations:

- **Simulation federations** run Flower Apps with simulated SuperNodes. Use this when
  you want to test or iterate on an app before connecting real SuperNodes.
- **Deployment federations** run Flower Apps on connected SuperNodes. 

.. note::

    Deployment federations require SuperGrid access. Contact hello@flower.ai to request access.

In a federation, all members can see the runs launched by any other member. Members can
also launch new runs.

*********************
 Create a Federation
*********************

Open SuperGrid and go to the federations view. Choose whether you want to create a
simulation federation or a deployment federation, then provide the federation name and
the required configuration.

For a simulation federation, you'll be prompted to specify the number of simulated SuperNodes. You can change this later if needed. To begin with, we recommend starting with a small number of SuperNodes (e.g., 5 or 10) to keep the runs fast.


For a deployment federation, create the federation first, then connect the SuperNodes
that should participate in runs. Only federation members can add their own SuperNodes to
the federation. For the SuperNode connection steps, see
:doc:`how-to-connect-supernodes-to-supergrid`.

*********************
 Manage a Federation
*********************

After creating a federation, the owner can invite other users to collaborate. Invited
users become federation members after accepting the invitation. Members can inspect the
runs launched in the federation and submit their own runs.

Federation ownership controls administrative actions:

- The owner can invite users.
- The owner can remove users from the federation.
- The owner can archive the federation when the collaboration is complete.
- Members can leave the federation themselves.

Archiving a federation is useful when a project or collaboration has ended. After a
federation is archived, it should no longer be used for new runs.
