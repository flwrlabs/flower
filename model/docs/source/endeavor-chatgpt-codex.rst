.. meta::
    :description: Set up Flower Endeavor with Codex CLI and the ChatGPT/Codex Desktop app on macOS using a Flower API key and model catalog.
    :property=og:description: Set up Flower Endeavor with Codex CLI and the ChatGPT/Codex Desktop app on macOS using a Flower API key and model catalog.

#################################
 Use Endeavor with ChatGPT/Codex
#################################

.. note::

    You need a Flower API key with Endeavor access. Request access by filling out the
    `Endeavor access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

This guide connects Codex CLI and local Codex tasks in the ChatGPT/Codex desktop app to
:substitution-code:`|endeavor_model|` at ``https://api.flower.ai/v1``. The commands
below use macOS and its default shell, zsh. Before continuing, install your preferred
client using the `Codex CLI installation guide
<https://learn.chatgpt.com/docs/codex/cli>`_ or the `desktop app setup guide
<https://learn.chatgpt.com/docs/quickstart>`_.

The example uses Endeavor |endeavor_version|. For release status, see :ref:`Available
versions <endeavor-versions>`. For model details, see :doc:`endeavor`. To use OpenCode
instead, see :doc:`endeavor-opencode`.

***************************
 1. Save the model catalog
***************************

Save the `flower-models.json <_static/flower-models.json>`_ catalog to your
``Downloads`` folder, keeping that filename. This catalog makes **Endeavor** available
in the model selector and defines its capabilities.

.. note::

    The supplied catalog configures a 128,000-token context window and a 96,000-token
    automatic compaction threshold for this client. API request limits depend on the
    deployment. See :ref:`Context window and limits <endeavor-context-limits>` for
    details.

In your terminal, copy the file to the Codex configuration directory:

.. code-block:: zsh

    mkdir -p "$HOME/.codex"
    cp "$HOME/Downloads/flower-models.json" "$HOME/.codex/flower-models.json"
    printf 'model_catalog_json = "%s/.codex/flower-models.json"\n' "$HOME"

Copy the printed ``model_catalog_json`` line for the next step.

********************
 2. Configure Codex
********************

The CLI and desktop app share ``~/.codex/config.toml``. Create the file if needed, back
it up, and open it:

.. code-block:: zsh

    touch "$HOME/.codex/config.toml"
    cp -p "$HOME/.codex/config.toml" \
      "$HOME/.codex/config.toml.backup.$(date +%Y%m%d-%H%M%S)"
    open -a TextEdit "$HOME/.codex/config.toml"

Add or update the following settings, preserving unrelated configuration. Keep the first
four settings at the top level, before any ``[section]``. Replace the
``model_catalog_json`` line with the one printed in `1. Save the model catalog`_, then
save the file. Update existing keys and sections rather than adding duplicates.

.. code-block:: toml
    :substitutions:

    model = "|endeavor_model|"
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

****************************
 3. Set your Flower API key
****************************

Follow :doc:`endeavor-api-key` to set ``FLOWER_API_KEY`` in your terminal. Keep the
terminal open for the next step.

*************************
 4. Start using Endeavor
*************************

Codex CLI
=========

In the same terminal, change to your project directory and start Codex:

.. code-block:: zsh

    codex

Endeavor is selected by default. Send a prompt such as ``Explain the structure of this
project.`` Repeat the instructions in :doc:`endeavor-api-key` when starting from a new
terminal session.

ChatGPT/Codex desktop app on macOS
==================================

Fully quit the desktop app. In the same terminal where you set ``FLOWER_API_KEY``,
launch it with the key:

.. code-block:: zsh

    open -a ChatGPT --env FLOWER_API_KEY="$FLOWER_API_KEY"

Start a new local Codex task and select **Endeavor** in the model selector. Send a
prompt to begin. Repeat this launch command each time you start the desktop app. Opening
it from the Dock may not supply the key.

**********
 Get help
**********

If you encounter any issues, feel free to post on `Flower Discuss
<https://discuss.flower.ai>`_ or `Flower Slack <https://flower.ai/join-slack>`_. The
Flower team will get back to you soon.
