.. meta::
    :description: Set up Flower Endeavor 1.0 with Codex CLI and the ChatGPT/Codex desktop app on macOS using a Flower API key and model catalog.
    :property=og:description: Set up Flower Endeavor 1.0 with Codex CLI and the ChatGPT/Codex desktop app on macOS using a Flower API key and model catalog.

Use Endeavor with ChatGPT/Codex
===============================

.. note::

    You need a Flower API key with Endeavor access. Request access by filling out
    the `Endeavor 1.0 access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

This guide connects Codex CLI and local Codex tasks in the ChatGPT/Codex
desktop app to ``flower-endeavor`` at ``https://api.flower.ai/v1``. The commands
below use macOS and its default shell, zsh. Before continuing, install your
preferred client using the `Codex CLI installation guide
<https://learn.chatgpt.com/docs/codex/cli>`_ or the `desktop app setup guide
<https://learn.chatgpt.com/docs/quickstart>`_.

For model details, see :doc:`endeavor`. To use OpenCode instead, see
:doc:`endeavor-opencode`.

1. Save the model catalog
-------------------------

Download :download:`flower-models.json <_static/flower-models.json>` to your
``Downloads`` folder, keeping that filename. This catalog makes **Endeavor**
available in the model selector and defines its capabilities.

In your terminal, copy the file to the Codex configuration directory:

.. code-block:: zsh

   mkdir -p "$HOME/.codex"
   cp "$HOME/Downloads/flower-models.json" "$HOME/.codex/flower-models.json"
   printf 'model_catalog_json = "%s/.codex/flower-models.json"\n' "$HOME"

Copy the printed ``model_catalog_json`` line for the next step.

2. Configure Codex
------------------

The CLI and desktop app share ``~/.codex/config.toml``. Create the file if
needed, back it up, and open it:

.. code-block:: zsh

   touch "$HOME/.codex/config.toml"
   cp -p "$HOME/.codex/config.toml" \
     "$HOME/.codex/config.toml.backup.$(date +%Y%m%d-%H%M%S)"
   open -a TextEdit "$HOME/.codex/config.toml"

Add or update the following settings, preserving unrelated configuration.
Keep the first four settings at the top level, before any ``[section]``.
Replace the ``model_catalog_json`` line with the one printed in step 1, then
save the file. Update existing keys and sections rather than adding duplicates.

.. code-block:: toml

   model = "flower-endeavor"
   model_provider = "flower"
   model_reasoning_effort = "low"
   model_catalog_json = "/Users/YOUR_USERNAME/.codex/flower-models.json"

   [model_providers.flower]
   name = "Flower Labs"
   base_url = "https://api.flower.ai/v1"
   wire_api = "responses"
   env_key = "FLOWER_API_KEY"

For more options, see the `Codex configuration reference
<https://developers.openai.com/codex/config-reference/>`_.

3. Set your Flower API key
--------------------------

Run this in your terminal, paste your key at the prompt, and press Enter:

.. code-block:: zsh

   read -rs 'FLOWER_API_KEY?Flower API key: '
   echo
   export FLOWER_API_KEY

The input is hidden and is not saved in shell history. Keep this terminal
open for the next step.

4. Start using Endeavor
-----------------------

Codex CLI
~~~~~~~~~

In the same terminal, change to your project directory and start Codex:

.. code-block:: zsh

   codex

Endeavor is selected by default. Send a prompt such as
``Explain the structure of this project.`` Repeat step 3 when starting from a
new terminal session.

ChatGPT/Codex desktop app on macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fully quit the app. In the same terminal where you set the key, run:

.. code-block:: zsh

   launchctl setenv FLOWER_API_KEY "$FLOWER_API_KEY"

This makes the key available to apps opened from the Dock. Reopen ChatGPT or
Codex, start a new local Codex task, and select **Endeavor** in the model
selector. Send a prompt to begin.

Repeat step 3 and the ``launchctl`` command after signing out of or restarting
macOS. To remove the key from the desktop environment, quit the app and run
``launchctl unsetenv FLOWER_API_KEY``.

Get help
--------

If you encounter any issues, feel free to post on
`Flower Discuss <https://discuss.flower.ai>`_. The Flower team will get back to
you soon.
