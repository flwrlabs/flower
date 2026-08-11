:og:description: Configure SuperLink account authentication with OpenID Connect.
.. meta::
    :description: Configure SuperLink account authentication with OpenID Connect.

##########################################
 Authenticate Accounts via OpenID Connect
##########################################

.. note::

    OpenID Connect Authentication is a Flower Enterprise feature. See `Flower Enterprise
    <https://flower.ai/enterprise>`_ for details.

In this guide, you'll learn how to configure SuperLink with account authentication and
how to log in using the ``flwr`` CLI. Once logged in, Flower accounts can run CLI
commands that interact with the SuperLink.

.. important::

    Account authentication does not replace resource-level access checks. Run ownership,
    federation membership and roles, and entitlements continue to constrain what an
    authenticated account can access.

***************
 Prerequisites
***************

To enable account authentication, the SuperLink must be deployed with an `OpenID Connect
(OIDC) <https://openid.net/developers/how-connect-works/>`_ provider. The OIDC provider
verifies account identity and supplies the account information used by SuperLink's
ownership, federation, and entitlement checks.

Enable Account Authentication on the SuperLink
==============================================

Set the following environment variables on the SuperLink process:

.. code-block:: bash

    FLWR_OIDC_ENABLED=1
    FLWR_OIDC_ISSUER=https://<domain>/realms/<realm>
    FLWR_OIDC_CLIENT_ID=<client_id>
    FLWR_OIDC_CLIENT_SECRET=<client_secret>
    FLWR_OIDC_VERIFY_TLS=1

``FLWR_OIDC_ENABLED`` defaults to ``0``. ``FLWR_OIDC_VERIFY_TLS`` defaults to
``1``. As with other Flower binary environment variables, use ``1`` for true and
``0`` for false. Control authentication remains NoOp unless
``FLWR_OIDC_ENABLED=1``, even when credentials are present.
``FLWR_OIDC_VERIFY_TLS`` affects only requests to the OIDC provider.

Start the SuperLink with these variables in its environment:

.. code-block:: bash

    $ FLWR_OIDC_ENABLED=1 \
        FLWR_OIDC_ISSUER=https://<domain>/realms/<realm> \
        FLWR_OIDC_CLIENT_ID=<client_id> \
        FLWR_OIDC_CLIENT_SECRET=<client_secret> \
        flower-superlink <other flags>

************************
 Login to the SuperLink
************************

Once a SuperLink with account authentication is up and running, an account can interface
with it after installing the ``flwr`` PyPI package via the Flower CLI. Configure the
SuperLink connection in your Flower Configuration file (typically located at
``$HOME/.flwr/config.toml``):

.. code-block:: toml
    :caption: config.toml

    [superlink]
    default = "my-prod-superlink"  # Set the default connection configuration

    [superlink.my-prod-superlink]
    address = "<SUPERLINK-ADDRESS>:<CONTROL-API-PORT>"   # Address of the SuperLink Control API
    root-certificate = "<PATH/TO/ca.crt>" # TLS certificate set for the SuperLink. Required for self-signed certificates.

.. note::

    - Account authentication is only supported with TLS connections.
    - Setting the default connection is optional. If you don't set your SuperLink as
      default, you can specify the connection name explicitly in each command, for
      example: ``flwr login my-prod-superlink``.

Learn more about the Flower Configuration file in the `Flower Configuration
<ref-flower-configuration.html>`_ reference.

You need to login first before other CLI commands can be executed. Upon executing ``flwr
login``, a URL will be returned by the authentication plugin in the SuperLink. Click on
it and authenticate directly against the OIDC provider.

.. code-block:: console

    $ flwr login
    A browser window has been opened for you to log into your Flower account.
    If it did not open automatically, use this URL:
    https://account.flower.blue/realms/flower/device?user_code=...
    # [... follows URL and logs in ... in the meantime the CLI will wait ...]
    ✅ Login successful.

Once the login is successful, the credentials returned by the OIDC provider via the
SuperLink will be stored locally. The tokens will be sent transparently with each
subsequent ``flwr`` CLI request to the SuperLink, and it will relay them to the OIDC
provider to perform the authentication checks.

*****************************************
 Run authenticated ``flwr`` CLI commands
*****************************************

With the above steps completed, you can now run ``flwr`` CLI commands against a
SuperLink setup with account authentication. For example, you can run the ``flwr run``
command to start a Flower app:

.. code-block:: console

    $ flwr run
    🎊 Successfully started run 1859953118041441032

SuperLink uses the authenticated account identity to enforce access to specific runs and
federations through ownership, federation membership and roles, and entitlement checks.
