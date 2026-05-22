##############################
 Get started as a contributor
##############################

***************
 Prerequisites
***************

- `Python 3.11 <https://docs.python.org/3.11/>`_ or above
- `uv 0.10.7 <https://docs.astral.sh/uv/>`_ or above
- (Optional) `pyenv <https://github.com/pyenv/pyenv>`_

Flower uses ``pyproject.toml`` to manage dependencies and configure development tools
(the ones which support it). ``uv`` is used for dependency management, build, and
publishing workflows.

*************************
 Developer Machine Setup
*************************

Preliminaries
=============

Some system-wide dependencies are needed.

For macOS
---------

- Install `homebrew <https://brew.sh/>`_. Don't forget the post-installation actions to
  add `brew` to your PATH.
- Install `xz` (to install different Python versions) and `pandoc` to build the docs:

  ::

      $ brew install xz pandoc

For Ubuntu
----------

Ensure you system (Ubuntu 22.04+) is up-to-date, and you have all necessary packages:

::

    $ apt update
    $ apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev \
                  libreadline-dev libbz2-dev libffi-dev liblzma-dev pandoc

Create Flower Dev Environment
=============================

1. Clone the `Flower repository <https://github.com/flwrlabs/flower>`_ from GitHub:

       ::

           $ git clone git@github.com:flwrlabs/flower.git
           $ cd flower

2. Install uv by following the `uv installation instructions
   <https://docs.astral.sh/uv/getting-started/installation/>`_.
3. Bootstrap the framework development environment:

       ::

           $ ./dev/bootstrap.sh

   The bootstrap script creates ``framework/.venv`` using the Python version pinned by
   the repository. Pass a Python version as the first argument to use a different one:

       ::

           $ ./dev/bootstrap.sh 3.11.14

*********************
 Convenience Scripts
*********************

The Flower repository contains a number of convenience scripts to make recurring
development tasks easier and less error-prone. See the ``/dev`` subdirectory for a full
list. The following scripts are amongst the most important ones:

Compile ProtoBuf Definitions
============================

::

    $ ./framework/dev/protoc.sh

Auto-Format Code
================

::

    $ ./framework/dev/format.sh

Run Linters and Tests
=====================

::

    $ ./framework/dev/test.sh

Add a pre-commit hook
=====================

Developers may integrate a pre-commit hook into their workflow utilizing the `pre-commit
<https://pre-commit.com/#install>`_ library. The pre-commit hook is configured to
execute two primary operations: ``./framework/dev/format.sh`` and
``./framework/dev/test.sh`` scripts.

There are multiple ways developers can use this:

1. Install the pre-commit hook to your local git directory by simply running:

   ::

       $ pre-commit install

   - Each ``git commit`` will trigger the execution of formatting and linting/test
     scripts.
   - If in a hurry, bypass the hook using ``--no-verify`` with the ``git commit``
     command.

     ::

         $ git commit --no-verify -m "Add new feature"

2. For developers who prefer not to install the hook permanently, it is possible to
   execute a one-time check prior to committing changes by using the following command:

   ::

       $ pre-commit run --all-files

   This executes the formatting and linting checks/tests on all the files without
   modifying the default behavior of ``git commit``.

Run Github Actions (CI) locally
===============================

Developers could run the full set of Github Actions workflows under their local
environment by using `Act <https://github.com/nektos/act>`_. Please refer to the
installation instructions under the linked repository and run the next command under
Flower main cloned repository folder:

::

    $ act

The Flower default workflow would run by setting up the required Docker machines
underneath.

***************
 Build Release
***************

Flower uses ``uv`` to build releases. The necessary command is wrapped in a simple
script:

::

    $ ./framework/dev/build.sh

The resulting ``.whl`` and ``.tar.gz`` releases will be stored in the
``./framework/dist`` subdirectory.

*********************
 Build Documentation
*********************

Flower's documentation uses `Sphinx <https://www.sphinx-doc.org/>`_. To build the
documentation locally, run the following script:

::

    $ ./framework/dev/build-docs.sh

This will generate HTML documentation in ``./framework/doc/build/html``.

Note that, in order to build the documentation locally, `Pandoc
<https://pandoc.org/installing.html>`_ needs to be installed on the system.
