#####################
 Write documentation
#####################

****************
 Project layout
****************

The Flower documentation lives in the ``doc`` directory. The Sphinx-based documentation
system supports both reStructuredText (``.rst`` files) and Markdown (``.md`` files).

Note that, in order to build the documentation locally (with ``python -m poetry run make
html``, as described below), `Pandoc <https://pandoc.org/installing.html>`_ needs to be
installed on the system.

***********************
 Edit an existing page
***********************

1. Edit an existing ``.rst`` (or ``.md``) file under ``framework/docs/source/``
2. Compile the docs: ``cd framework/docs``, then ``python -m poetry run make html``
3. Open ``framework/docs/build/html/index.html`` in the browser to check the result

*******************
 Create a new page
*******************

1. Add new ``.rst`` file under ``framework/docs/source/``
2. Add content to the new ``.rst`` file
3. Link to the new rst from ``index.rst``
4. Compile the docs: ``cd framework/docs``, then ``python -m poetry run make html``
5. Open ``framework/docs/build/html/index.html`` in the browser to check the result
