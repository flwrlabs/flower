# Flower 1.31 SuperLink HA Test Report

Date: 2026-06-04
Branch: `codex/ha-release-tests-clean`
Base: `origin/main` at `95dccb71f`

## Scope

This report covers the recent high-availability and shared-state fixes on `main`:

- Atomic message instruction/reply claiming across multiple `SqlLinkState` replicas
- Duplicate message reply rejection
- Node heartbeat handling after node deletion
- ObjectStore concurrent preregistration, cleanup, and put/delete races
- `ServerAppIo` shared upload-hint preservation
- `ServerAppIo` task claim behavior across two gRPC replicas
- PostgreSQL migration advisory locking and migration compatibility

## Tests Added

- `dev/test-superlink-ha-release.sh`
  - Shareable runner for the full focused HA release gate.
  - Default mode requires only `uv`.
  - `--with-postgres` starts and stops a disposable PostgreSQL container and runs the real PostgreSQL migration smoke test.

- `py/flwr/server/superlink/linkstate/linkstate_test.py::SqlFileBasedTest::test_store_message_res_rejects_concurrent_duplicate_across_replicas`
  - Starts two `SqlLinkState` replicas over the same SQL database.
  - Races two replies for the same instruction.
  - Verifies only one reply is stored and later returned.

- `py/flwr/supercore/state/alembic/utils_test.py::TestAlembicRun::test_run_migrations_unlocks_postgresql_when_migration_leaves_no_transaction`
  - Exercises PostgreSQL advisory lock lifecycle when the migration workflow leaves no open transaction.
  - Verifies no unnecessary commit is issued before advisory unlock.

- `py/flwr/supercore/state/alembic/utils_test.py::TestAlembicRun::test_run_migrations_on_real_disposable_postgresql`
  - Skipped by default.
  - Runs only when `FLWR_TEST_POSTGRES_DISPOSABLE_URL` is set.
  - Resets the disposable `public` schema, applies all migrations on real PostgreSQL, verifies Alembic heads, and checks key SuperLink/CoreState/ObjectStore tables exist.

## How To Verify

Default local gate:

```bash
cd framework
dev/test-superlink-ha-release.sh
```

Disposable PostgreSQL gate:

```bash
cd framework
dev/test-superlink-ha-release.sh --with-postgres
```

Requirements:

- `uv`
- Docker only for `--with-postgres`
- Local port `55432` available, or set `POSTGRES_PORT=<port>`

## Areas Covered

| Area | Coverage |
| --- | --- |
| Message instruction claim | Concurrent shared-SQL replica test ensures only one replica claims one instruction. |
| Message reply claim | Concurrent shared-SQL replica test ensures only one replica claims one reply. |
| Duplicate replies | New concurrent shared-SQL replica test ensures only one reply can be stored for an instruction. |
| Work distribution | Existing contention test ensures two replicas can claim two available messages without collapsing onto one message. |
| Task STARTING -> RUNNING claim | Existing process-level and gRPC-level HA tests ensure only one replica can activate a task. |
| Node heartbeat deletion race | Existing SQL test ensures heartbeat cannot revive an unregistered node. |
| ObjectStore cleanup/preregister race | Existing SQL ObjectStore test ensures another run's shared object survives concurrent cleanup. |
| ObjectStore put/delete race | Existing SQL ObjectStore test ensures `put` does not report success after concurrent delete. |
| PushMessages upload hints | Existing `ServerAppIo` test ensures accepted message upload hints are preserved when a later message is rejected. |
| PostgreSQL migration lock lifecycle | Existing and new mocked-Postgres tests cover lock, commit, rollback, unlock, and no-open-transaction paths. |
| Real PostgreSQL migrations | New opt-in test applies all migrations against a temporary PostgreSQL 15 container. |

## Remaining Gaps

- Runtime `SqlLinkState` and `SqlObjectStore` still reject PostgreSQL through `SQL_ALLOWED_DIALECTS = {"sqlite"}` on current `main`. The real PostgreSQL smoke test can validate migrations, but not normal message/object runtime behavior.
- There is no full two-SuperLink plus load-balancer plus PostgreSQL deployment e2e in this branch.
- The PR #7244 lease/ack message-delivery model is not on `main`; if it lands before Flower 1.31, it needs a separate release gate covering large object transfers, duplicate active pullers, generated replies, and confirm/cleanup semantics.

## Release Recommendation

For the HA fixes currently on `main`, this branch adds a clean, shareable regression gate for the main multi-replica and migration failure modes.

This should be treated as regression coverage for the merged HA fixes, not proof that SuperLink is fully production-HA-ready with PostgreSQL. Full production readiness still requires enabling PostgreSQL in the runtime path and adding a true multi-SuperLink deployment e2e.
