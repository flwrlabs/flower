###################
 Build from source
###################

This concise guide explains how to build the Flower Intelligence TypeScript SDK from
source. It also explains how to install and use ``pnpm`` as your package manager.

***************
 Prerequisites
***************

- ``git`` and ``Node.js`` must be installed.
- A working Internet connection for installing dependencies.

*********************
 Installing ``pnpm``
*********************

You can install ``pnpm`` globally using npm. Open your terminal and run:

.. code-block:: bash

    npm install -g pnpm

Alternatively, if you are using Corepack (bundled with recent Node.js versions), enable
pnpm with:

.. code-block:: bash

    corepack enable pnpm

*********************
 Setting up the repo
*********************

If you already have the Flower repository cloned somewhere, you can skip this step and
navigate to it and run ``git pull``.

**Clone the Flower repo and navigate into it.**

.. code-block:: bash

    git clone https://github.com/flwrlabs/flower.git
    cd flower/

**********************
 Building from source
**********************

1. **Navigate to the TypeScript SDK:**

.. code-block:: bash

    cd intelligence/ts

2. **Install dependencies:**

.. code-block:: bash

    pnpm install

3. **Compile the project:**

.. code-block:: bash

    pnpm build
