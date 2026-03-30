# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Alembic environment configuration for State migrations."""


from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from flwr.supercore.state.alembic.utils import get_combined_metadata

# Alembic Config object
config = context.config  # pylint: disable=no-member

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


# Target metadata for autogenerate
target_metadata = get_combined_metadata()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an Engine is
    acceptable here as well. By skipping the Engine creation we don't even need a DBAPI
    to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(  # pylint: disable=no-member
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():  # pylint: disable=no-member
        context.run_migrations()  # pylint: disable=no-member


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an engine and associates a connection with the context. Supports two modes:

    1. Standard: Creates a new connection from the configured URL.
    2. Pre-connected: Uses an existing connection from config.attributes["connection"].

    Pre-connected mode is necessary for in-memory SQLite databases, where each new
    connection creates a separate database instance. This allows
    _get_baseline_metadata() to run migrations and reflect schema from the same
    in-memory database without requiring filesystem write access.
    """
    # Check if a connection was provided (e.g., for in-memory databases).
    # This allows callers to pass an active connection that should be reused
    # instead of creating a new one from the URL.
    connection = config.attributes.get("connection", None)

    if connection is None:
        # Standard path: create engine from config
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            # pylint: disable-next=no-member
            context.configure(connection=connection, target_metadata=target_metadata)

            with context.begin_transaction():  # pylint: disable=no-member
                context.run_migrations()  # pylint: disable=no-member
    else:
        # Use the provided connection directly (for in-memory databases)
        context.configure(  # pylint: disable=no-member
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():  # pylint: disable=no-member
            context.run_migrations()  # pylint: disable=no-member


if context.is_offline_mode():  # pylint: disable=no-member
    run_migrations_offline()
else:
    run_migrations_online()
