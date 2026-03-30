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
2. Upgrades it to the current ``head`` revision
3. Runs autogenerate to detect your schema changes
4. Generates a new migration file in ``py/flwr/supercore/state/alembic/versions/``
5. Automatically cleans up the temporary database

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
    alembic upgrade head
    alembic revision --autogenerate -m "Descriptive message about the schema change"
    rm state.db  # Clean up the generated database file

.. admonition:: Important

    The manual workflow creates a ``state.db`` file that should not be committed to git.
