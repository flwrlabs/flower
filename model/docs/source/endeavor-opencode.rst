.. meta::
    :description: Set up Flower Endeavor 1.0 with OpenCode CLI and OpenCode Desktop on macOS using a Flower API key and the Flower Responses API.
    :property=og:description: Set up Flower Endeavor 1.0 with OpenCode CLI and OpenCode Desktop on macOS using a Flower API key and the Flower Responses API.

Use Endeavor with OpenCode
==========================

.. note::

    You need a Flower API key with Endeavor access. Request access by filling out
    the `Endeavor 1.0 access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

This guide connects OpenCode CLI and OpenCode Desktop to
``flwrlabs/endeavor-1.0`` at ``https://api.flower.ai/v1``. The commands below
use macOS and its default shell, zsh. Install your preferred client using the
`OpenCode setup guide <https://opencode.ai/download>`_ before continuing.

For model details, see :doc:`endeavor`. To use ChatGPT/Codex instead, see
:doc:`endeavor-chatgpt-codex`.

1. Configure OpenCode
---------------------

The CLI and desktop app share the global OpenCode configuration. Create its
directory if needed:

.. code-block:: zsh

   mkdir -p "$HOME/.config/opencode"

Save the following as ``~/.config/opencode/opencode.json`` in a text editor.
If you already have an ``opencode.json`` or ``opencode.jsonc`` file in that
directory, back it up and edit that file instead. Merge the top-level ``model``
and ``provider.flower-labs`` settings, preserving unrelated configuration.

.. code-block:: json

   {
     "$schema": "https://opencode.ai/config.json",
     "model": "flower-labs/flwrlabs/endeavor-1.0",
     "provider": {
       "flower-labs": {
         "npm": "@ai-sdk/openai",
         "name": "Flower Labs",
         "options": {
           "baseURL": "https://api.flower.ai/v1",
           "apiKey": "{env:FLOWER_API_KEY}"
         },
         "models": {
           "flwrlabs/endeavor-1.0": {
             "name": "Endeavor"
           }
         }
       }
     }
   }

Keep ``@ai-sdk/openai`` as shown: it uses the Responses API required by this
setup. See the `OpenCode provider documentation
<https://opencode.ai/docs/providers/#custom-provider>`_ for details.

For context window and API limit details, see :ref:`Context window and limits
<endeavor-context-limits>`.

2. Set your Flower API key
--------------------------

Follow :doc:`endeavor-api-key` to set ``FLOWER_API_KEY`` in your terminal.
For desktop use, also complete that guide's desktop setup before reopening
OpenCode. Keep the terminal open for the next step.

3. Start using Endeavor
-----------------------

OpenCode CLI
~~~~~~~~~~~~

In the same terminal, change to your project directory and start OpenCode:

.. code-block:: zsh

   opencode --model flower-labs/flwrlabs/endeavor-1.0

Send a prompt such as ``Explain the structure of this project.`` Repeat the
instructions in :doc:`endeavor-api-key` when starting from a new terminal
session.

OpenCode Desktop on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~

After completing the desktop key setup in :doc:`endeavor-api-key`, reopen
OpenCode, open a local project, start a new session, and select **Endeavor**
from **Flower Labs** in the model selector. Send a prompt to begin.

Get help
--------

If you encounter any issues, feel free to post on
`Flower Discuss <https://discuss.flower.ai>`_ or
`Flower Slack <https://flower.ai/join-slack>`_. The Flower team will get back
to you soon.
