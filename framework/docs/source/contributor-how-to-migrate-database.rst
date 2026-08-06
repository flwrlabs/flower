################################
 Migrate Flower Database Schema
################################

When making changes to the database schema used in Flower, it is essential to create a
migration script to ensure that existing on-disk databases can be updated to the new
schema without data loss. From Flower version ``1.26.0`` onwards, the framework uses
`Alembic <https://alembic.sqlalchemy.org/en/latest/>`_ as our database migration tool.

This guide describes the steps required to create a new migration script after modifying
the database schema.

****************
 Pre-requisites
****************

Install development versions of Flower according to the instructions in
:doc:`contributor-how-to-install-development-versions` with the ``dev`` dependencies.

***********************
 Generating Migrations
***********************

The Flower SQL database schema is defined under ``supercore/state/schema/``. After
making changes to the schema (e.g., adding a new column to a table), generate the
migration revision:

.. code-block:: shell

    python -m dev.generate_migration "Descriptive message about the schema change"

This command:

1. Creates a temporary SQLite database
2. Upgrades it to all current ``heads``
3. Runs autogenerate to detect your schema changes, targeting ``flwr@head`` by default
4. Generates a new migration file that extends the current ``flwr`` branch head in
   ``py/flwr/supercore/state/alembic/versions/``
5. Automatically cleans up the temporary database

The generator does not loop over migration files itself. The ``alembic upgrade heads``
command asks Alembic to traverse the revision graph. Each revision identifies its
predecessor through ``down_revision``, so Alembic applies every pending revision in
dependency order until all configured branch heads have been reached. Once the
temporary database is current, the generator runs ``alembic revision --autogenerate``
with ``--head flwr@head``. This makes the new revision extend the Flower branch instead
of leaving the parent revision ambiguous when multiple heads exist. To target another
configured branch, pass its branch head explicitly with ``--head <branch>@head``.

*****************************
 Review Generated Migrations
*****************************

Always review the generated migration file before committing:

- Check that the detected changes match your intent
- Verify data migration logic if renaming/removing columns
- Test the upgrade and downgrade paths

*******************************
 Manual Workflow (Alternative)
*******************************

If you prefer using the Alembic CLI directly:

.. code-block:: shell

    cd framework
    alembic upgrade heads
    alembic revision --autogenerate --head flwr@head \
      -m "Descriptive message about the schema change"
    rm state.db  # Clean up the generated database file

.. admonition:: Important

    Use ``heads`` when upgrading so every configured migration branch is current, and
    explicitly select ``flwr@head`` when creating a Flower revision. The manual workflow
    creates a ``state.db`` file that should not be committed to git.
