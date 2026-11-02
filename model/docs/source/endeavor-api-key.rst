.. meta::
    :description: Set your Flower API key for Endeavor in Codex and OpenCode CLI and Desktop apps on macOS.
    :property=og:description: Set your Flower API key for Endeavor in Codex and OpenCode CLI and Desktop apps on macOS.

#########################
 Set your Flower API key
#########################

.. note::

    You need a Flower API key with Endeavor access. Request access by filling out the
    `Endeavor access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

Use these instructions to make your Flower API key available to Codex or OpenCode.

*************************
 Set the key for CLI use
*************************

These commands work in Bash and zsh. Run ``printf`` and ``read``, paste your key at the
prompt, and press Enter. Then run ``echo`` and ``export``.

.. code-block:: zsh

    printf 'Flower API key: '
    read -rs FLOWER_API_KEY
    echo
    export FLOWER_API_KEY

The input is hidden and is not saved in shell history. Keep this terminal open and start
your client from it. Repeat these commands when starting from a new terminal session.

************************
 Test the key with curl
************************

In the same terminal, send a test request to the Flower Responses API:

.. code-block:: zsh
    :substitutions:

    curl --fail-with-body -sS 'https://api.flower.ai/v1/responses' \
      -H "Authorization: Bearer ${FLOWER_API_KEY}" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "|endeavor_model|",
        "input": "Say hello in one sentence.",
        "max_output_tokens": 64,
        "stream": false
      }'

The response should contain JSON with a short greeting in ``output``. If the request
fails, check the error message and confirm that your key has Endeavor access.

*****************************
 Set the key for Desktop use
*****************************

Desktop apps may not inherit environment variables set in a terminal. Fully quit your
client, then launch it from the terminal where you set ``FLOWER_API_KEY``. Use the
command in the desktop section of your client guide below.

The launch command passes the key to the new app process. Fully quit the client and
repeat the command each time you start it.

*********************************
 Continue with your client guide
*********************************

- :doc:`endeavor-chatgpt-codex`
- :doc:`endeavor-opencode`
