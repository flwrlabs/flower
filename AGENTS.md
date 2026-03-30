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

## Review output format

When reviewing a PR, output:
- Critical issues
- Simplicity/readability suggestions
- Consistency concerns
- Whether the PR should be split
- A brief overall verdict
