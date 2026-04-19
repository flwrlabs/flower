Sign Hub Apps
=============

This guide explains how app signing works in Flower Hub and how to use it in practice.

Before an app can be reviewed and signed, it must already be published to Flower Hub.
The app does not need to be published by the same account that reviews and signs it.

.. note::

   App signing and app-page verification metadata are preview features. They may
   change over time.


How App Signing Works
---------------------

At a high level:

1. An app exists on Flower Hub.
2. The reviewer has an Ed25519 key pair and has registered the public key in
   their Flower account profile.
3. The reviewer downloads the FAB and signs it with the matching Ed25519
   private key via :code:`flwr app review`.
4. The signature is attached to app verification metadata in Flower Hub.
5. Users can inspect the app's verification metadata on the app page and decide
   which signers they trust.


Prerequisites
-------------

- Ensure :code:`flwr` is installed
- A Flower account and active login
- An Ed25519 OpenSSH key pair for signing
- The corresponding public key added to your Flower account profile
- An app on Flower Hub to review and sign

Log in to your Flower account on SuperGrid:

.. code-block:: bash

    flwr login supergrid


Create a Signing Key
--------------------

Generate an Ed25519 key pair in OpenSSH format:

.. code-block:: bash

    ssh-keygen -t ed25519 -f hub_signing_key -C "hub-review-key"

This creates:

- :code:`hub_signing_key` (private key, keep secret)
- :code:`hub_signing_key.pub` (public key)

.. warning::

   Keep private keys secure. Anyone with this private key can produce signatures
   attributed to that signer.


Register Your Public Key in Your Flower Account
-----------------------------------------------

Add the public key to your Flower account profile:

- Open :code:`https://flower.ai/profile/<account_username>/`
- Add the content of :code:`hub_signing_key.pub` to your profile keys

When signing, you must use the private key corresponding to a public key
registered in the reviewer account.


Choose an App to Sign
---------------------

You can sign:

- your own app, or
- an app published by someone else.

Supported app specs:

- :code:`@account/app` (latest version)
- :code:`@account/app==x.y.z` (specific version)

Examples:

.. code-block:: bash

    # Sign latest version
    flwr app review @flwrlabs/quickstart-pytorch

    # Sign a specific version
    flwr app review @flwrlabs/quickstart-pytorch==1.1.0


Review and Sign the App
-----------------------

Sign an app version:

.. code-block:: bash

    flwr app review @account/app==x.y.z

The CLI will:

1. Download the FAB.
2. Unpack it for manual inspection.
3. Ask you to type :code:`SIGN`.
4. Ask for the path to your Ed25519 OpenSSH private key.
5. Submit the signature to Flower Hub.

.. note::

   :code:`flwr app review` signs the FAB digest plus timestamp. The resulting
   signature is submitted with the app ID and version.


Check Verifications on the App Page
-----------------------------------

If you want to run an app and evaluate trust, open the app page on Flower Hub
and check the :code:`Verifications` section.
Use this section to see who signed the app and decide whether you trust those
signers.


Troubleshooting
---------------

- **Private key errors during review**

  Ensure that your private key is an **Ed25519 OpenSSH** key. Other key types
  or formats are not supported.

- **Signature not displayed on the app page**

  Verify that you:

  1. Successfully completed :code:`flwr app review`
  2. Signed the app using a private key that corresponds to a public key
     registered in your Flower account profile

  If the public key is not registered, or does not match the private key used
  for signing, the signature will not be displayed.


See Also
--------

- :doc:`how-to-publish-app-on-hub`
- :doc:`how-to-use-app-from-hub`
