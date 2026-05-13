##############################
 Get started as a contributor
##############################

***************
 Prerequisites
***************

- `Python 3.10 <https://docs.python.org/3.10/>`_ or above
- `Poetry 2.3 <https://python-poetry.org/>`_ or above
- (Optional) `pyenv <https://github.com/pyenv/pyenv>`_
- (Optional) `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_

Flower uses ``pyproject.toml`` to manage dependencies and configure development tools
(the ones which support it). Poetry is a build tool which supports `PEP 517
<https://peps.python.org/pep-0517/>`_.

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

2. Create and activate a Python virtual environment for development. See `Set up a
   virtual env <contributor-how-to-set-up-a-virtual-env.rst>`_ for detailed
   instructions.

   One way to do this is by using `pyenv <https://github.com/pyenv/pyenv>`_ and
   `pyenv-virtualenv <https://github.com/pyenv/pyenv-virtualenv>`_. You can also
   optionally install a specific Python version using pyenv if you haven't already
   installed your desired version:

       ::

           $ pyenv install <your-python-version>
           $ pyenv virtualenv <your-python-version> <your-env-name>
           $ pyenv activate <your-env-name>

3. Install Poetry, which is used to manage dependencies and development workflows:

       ::

           (your-env-name) $ pip install poetry==2.3.2

4. Navigate to the ``framework`` directory and install the Flower project in development
   mode, including all optional dependencies:

       ::

           (your-env-name) $ cd framework
           (your-env-name) $ python -m poetry install --all-extras

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

Flower uses Poetry to build releases. The necessary command is wrapped in a simple
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
