Set your Flower API key
=======================

Use these instructions to make your Flower API key available to Codex or
OpenCode. You need a key with Endeavor access. Request access using
the `Endeavor 1.0 access form <https://flowerlabs.typeform.com/to/jlniHsuy>`_.

Set the key for CLI use
-----------------------

In a zsh terminal, run the following, paste your key at the prompt, and press
Enter:

.. code-block:: zsh

   read -rs 'FLOWER_API_KEY?Flower API key: '
   echo
   export FLOWER_API_KEY

The input is hidden and is not saved in shell history. Keep this terminal
open and start your client from it. Repeat these commands when starting from
a new terminal session.

Set the key for desktop use
---------------------------

Fully quit your desktop client. On macOS, run the following in the same terminal
where you set the key:

.. code-block:: zsh

   launchctl setenv FLOWER_API_KEY "$FLOWER_API_KEY"

This makes the key available to apps opened from the Dock. Reopen your client
after running the command. Repeat the CLI key setup and this command after
signing out of or restarting macOS.

To remove the key from the desktop environment, quit your client and run:

.. code-block:: zsh

   launchctl unsetenv FLOWER_API_KEY

Continue with your client guide
-------------------------------

- :doc:`endeavor-chatgpt-codex`
- :doc:`endeavor-opencode`
