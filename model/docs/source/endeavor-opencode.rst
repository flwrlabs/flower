Use Endeavor with OpenCode
==========================

You need a Flower API key with Endeavor access. Request access by filling out
the `Endeavor 1.0 access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

This guide connects OpenCode CLI and OpenCode Desktop to ``flower-endeavor``
at ``https://api.flower.ai/v1``. The commands below use macOS and its default
shell, zsh. Install your preferred client using the `OpenCode setup guide
<https://opencode.ai/docs/>`_ before continuing.

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
     "model": "flower-labs/flower-endeavor",
     "provider": {
       "flower-labs": {
         "npm": "@ai-sdk/openai",
         "name": "Flower Labs",
         "options": {
           "baseURL": "https://api.flower.ai/v1",
           "apiKey": "{env:FLOWER_API_KEY}"
         },
         "models": {
           "flower-endeavor": {
             "name": "Endeavor"
           }
         }
       }
     }
   }

Keep ``@ai-sdk/openai`` as shown: it uses the Responses API required by this
setup. See the `OpenCode provider documentation
<https://opencode.ai/docs/providers/#custom-provider>`_ for details.

2. Set your Flower API key
--------------------------

Run this in your terminal, paste your key at the prompt, and press Enter:

.. code-block:: console

   read -rs 'FLOWER_API_KEY?Flower API key: '
   echo
   export FLOWER_API_KEY

The input is hidden and is not saved in shell history. Keep this terminal
open for the next step.

3. Start using Endeavor
-----------------------

OpenCode CLI
~~~~~~~~~~~~

In the same terminal, change to your project directory and start OpenCode:

.. code-block:: zsh

   opencode --model flower-labs/flower-endeavor

Send a prompt such as ``Explain the structure of this project.`` Repeat
step 2 when starting from a new terminal session.

OpenCode Desktop on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~

Fully quit OpenCode. In the same terminal where you set the key, run:

.. code-block:: zsh

   launchctl setenv FLOWER_API_KEY "$FLOWER_API_KEY"

This makes the key available to apps opened from the Dock. Reopen OpenCode,
open a local project, start a new session, and select **Endeavor** from
**Flower Labs** in the model selector. Send a prompt to begin.

Repeat step 2 and the ``launchctl`` command after signing out of or restarting
macOS. To remove the key from the desktop environment, quit OpenCode and run
``launchctl unsetenv FLOWER_API_KEY``.

Get help
--------

If you encounter any issues, feel free to post on
`Flower Discuss <https://discuss.flower.ai>`_. The Flower team will get back to
you soon.
