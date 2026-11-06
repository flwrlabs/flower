# AGENTS.md

## PR review philosophy

When reviewing or generating code, apply this checklist:

1. Necessity
   - Ask whether each added block is required for the requested behavior.
   - Flag speculative abstractions, premature generalization, and dead paths.

2. Simplicity
   - Prefer less code when readability and correctness are preserved.
   - Suggest idiomatic code for the language in that module only if it makes the result easier to understand.
   - Avoid "clever" one-liners that reduce maintainability.

3. Readability
   - Prefer explicit names, small functions, shallow nesting, and linear control flow.
   - Flag dense logic that would be easier to read if split or renamed.

4. Local consistency
   - Compare with nearby modules and existing patterns before proposing structure/style changes.
   - Follow existing naming, error-handling, typing, and test conventions.

5. PR sizing
   - Flag PRs that combine unrelated concerns.
   - Suggest a split when refactoring, behavior changes, and cleanup are mixed together.

## Comments for PR review

For every PR, add or update concise source comments explaining each logical block
introduced or materially changed in non-test code. Write for a reviewer unfamiliar
with the change: explain the block's purpose and any important decisions,
assumptions, invariants, or side effects. Avoid merely paraphrasing individual
statements.

Keep explanations next to the code they describe. Reuse clear existing comments
or docstrings instead of duplicating them, and update explanations when behavior
changes. This requirement excludes test files. Before opening or updating a PR,
review the full diff for missing, misleading, or stale explanations.

## Review output format

When reviewing a PR, output:
- Critical issues
- Simplicity/readability suggestions
- Consistency concerns
- Whether the PR should be split
- A brief overall verdict

## Development Patterns

### Database Migrations

Python services with databases use Alembic:

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration "Description"
```

For Alembic-backed services, do not write a new migration file from scratch when
the intended change is a schema diff. Help the user use
`python -m dev.generate_migration` from the `framework/` directory instead, then
review the generated revision and make only the minimal adjustments needed. This
helps avoid schema drift between SQLAlchemy models and committed migrations.

The migration generator is Alembic-branch aware. It upgrades the temporary
database to all heads, then autogenerates a revision against the selected branch
head. The default branch head is `flwr@head`; these commands are equivalent:

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration "Widen integer columns"
```

```bash
cd framework
uv run --no-sync --python=3.11.14 python -m dev.generate_migration --head flwr@head "Widen integer columns"
```

After autogeneration:

- Confirm the new revision has the intended `down_revision`.
- Confirm the generated operations match the SQLAlchemy metadata change.
- Review generated `batch_alter_table` blocks for SQLite compatibility.
- Update generated schema documentation if table metadata changed.
