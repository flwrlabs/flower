---
name: flower-release-changelog
description: Polish a generated Flower framework release changelog in framework/docs/source/changelog/vX.Y.Z.md using pull-request metadata and diffs cached by framework/dev/update_changelog.py. Use when Codex must classify generated PR entries into release-level topics, write or refine topic summaries, separate incompatible changes, or incrementally place newly appended PRs into an already polished release changelog.
---

# Flower Release Changelog

Polish one generated Framework release file using only the local changelog cache.

## Source files

- Release: `framework/docs/source/changelog/v<VERSION>.md`
- Cache state: `.cache/update_changelog/state.json`
- PR metadata: `.cache/update_changelog/prs/<PR>.json`
- PR diff: `.cache/update_changelog/prs/<PR>.diff`

Treat the cache as the complete PR evidence source. Do not use `gh`, GitHub APIs, web browsing, or PR pages. If the release file, state file, or required cache entries are missing, ask the user to run `framework/dev/update_changelog.py` and stop.

## Workflow

1. Read all applicable repository instructions.
2. Resolve the requested version. When the user does not specify one, use `version` from `state.json`. Require the requested version and cached version to match after removing an optional leading `v`.
3. Read the complete target release and the three most recent completed release files for local style only.
4. Before editing, record every PR number in the target file and the PR numbers currently under `### Incompatible changes`. This is the expected final inventory.
5. Identify the generated, unfinished PR entries. Use `new_pr_numbers` as a hint, not as the sole test:
   - Every PR entry under a heading other than `Thanks to our contributors`, `What's new?`, or `Incompatible changes` is unfinished.
   - A one-PR entry under `What's new?` or `Incompatible changes` is unfinished when it has no summary and its title is the formatted cached PR title.
6. Choose the editing mode:
   - **Initial polish:** When the file contains only generated entries, classify and rewrite all PR entries.
   - **Incremental polish:** When polished items already exist, classify only unfinished entries. Preserve every existing item unless an unfinished PR is added to it.
7. Read the JSON metadata for every PR being classified. Read its cached diff only when the title, body, and labels do not establish the user-visible or meaningful technical outcome.
8. Group unfinished PRs into coherent release-level topics and place them into the final sections. When the cached evidence does not support a confident grouping or summary, move the generated one-PR entry unchanged under `### UNGROUPED` instead of guessing. Remove all other generated headings and raw entries.
9. Run the bundled inventory validator with the pre-edit inventories. Then manually review classification, wording, formatting, the complete file, and its diff before finishing.

## Classify PRs

- Group by shared user-visible behavior or one meaningful technical outcome. Generated headings and labels are evidence, not mandatory final topic boundaries.
- In incremental mode, first try to place each unfinished PR into an existing topic. Treat the existing topic structure as stable, and create a new topic only when the PR introduces a distinct release-level outcome that cannot reasonably fit any existing topic.
- When adding a PR to an existing topic, add its PR link and leave the existing title and summary unchanged when they still represent the topic accurately. If the summary must change to cover the new PR, make only the smallest incremental edit needed; do not rewrite the summary for style or completeness.
- Put a PR under `### Incompatible changes` when its cached title starts with `break(` or its generated entry was placed there. Keep all other PRs under `### What's new?`.
- Use `General improvements` for minor maintenance, dependency, CI, test, and cleanup PRs that do not merit a distinct release-level topic.
- Never omit, duplicate, or split one PR across items.
- Keep `General improvements` last under `What's new?`.
- Include `### Incompatible changes` only when it contains at least one PR.
- Keep one uncertain PR per item under `### UNGROUPED`, preserve its generated title and link without adding a summary, and make `### UNGROUPED` the final section after `### Incompatible changes`. Omit the section when every PR was grouped confidently.

## Write topic items

Draft one release-level item from the complete set of cached PR evidence for each new topic. When extending an existing topic, revise that item only when necessary to represent the added PR.

- Produce one bullet title and one summary per topic.
- Begin non-General titles with an imperative verb.
- Remove Conventional Commit prefixes and scopes such as `feat(framework):`, `fix(framework):`, `refactor(...)`, and `break(...)`.
- Include every assigned PR exactly once, deduplicate links, and sort them in strictly ascending numerical order.
- Describe the combined outcome in present tense without enumerating implementation steps.
- Preserve Flower terminology and format commands, APIs, identifiers, and code symbols with backticks.
- Do not invent behavior beyond the cached metadata and diff.
- Indent every summary line with two spaces, not a tab.

Use this structure:

```markdown
- **[Imperative title]** ([#PR1](URL), [#PR2](URL))

  [Present-tense summary]
```

### Favor reader-facing wording

- Write from the reader's perspective: describe what changes for users or deployers, not merely what the implementation did.
- For user-facing topics, make the title describe the capability, action, or outcome readers gain rather than the underlying implementation. Prefer verbs such as `Run`, `Build`, `Use`, `Connect`, `Get`, or `Configure` when the evidence supports them.
- For internal technical work that merits a release-level topic, frame it around the supported outcome, such as reliability, security, performance, consistency, or operability. Omit low-level implementation details when they have no meaningful reader-facing effect.
- Preparatory or migration work may be worth calling out when it helps readers anticipate a larger user-visible change. Explain the direction of the change and, when relevant, what remains unchanged in this release for compatibility.
- Lead the summary with what users can now do, observe, or need to change. Follow with implementation details only when they help readers understand configuration, compatibility, operation, or the scope of the change.
- Prefer constructions such as “AgentApps can now ...” or “Run `flwr ...` to ...” over opening with an implementation inventory such as “Adds ... alongside ...” when both are equally accurate.
- Prefer neutral, forward-looking language that describes the resulting improvement instead of evaluating or criticizing the previous implementation.
- Preserve meaningful maturity qualifiers such as `(experimental)` or `(research preview)` when the evidence supports them.
- Use `you` sparingly when it makes an action or upgrade requirement clearer. Continue to avoid unnecessary `we`.
- Keep architecture and infrastructure topics technical when the technical change itself is useful release context, but do not manufacture a user benefit that is not supported by the evidence.
- For incompatible changes, state the observable impact and required migration action directly when the evidence establishes them.
- Keep the tone restrained and professional. Do not add promotional language, unsupported benefits, or subjective claims such as “seamless,” “powerful,” or “easy.”

Use this exact General item, replacing the sample link with its complete sorted PR set:

```markdown
- **General improvements** ([#123](https://github.com/flwrlabs/flower/pull/123))

  As always, many parts of the Flower framework and quality infrastructure were improved and updated.
```

## Protect incremental edits

When a mostly polished changelog contains only a few unfinished PRs:

- Preserve the existing topic structure by default. Do not regroup, rename, reorder, or rewrite existing items unless required to place a new PR correctly.
- Assign each unfinished PR to the best existing topic whenever there is a reasonable semantic fit. Create a new topic only when no existing topic can represent the PR without becoming misleading and the PR merits its own release-level topic.
- Do not update an existing summary merely to improve its wording or make it more comprehensive. If the new PR introduces information that the current summary needs to represent, update it incrementally while preserving the existing wording and structure as much as possible.
- Leave the release heading and contributor section unchanged.
- Do not edit `framework/docs/source/changelog/index.md` or another release file.

If the cached metadata and diff still do not support a confident classification or summary, move the PR under `### UNGROUPED` instead of guessing. Entries under `### UNGROUPED` remain unfinished and can be reconsidered on later incremental runs.

## Validate

Run from the repository root:

```bash
python3 .agents/skills/flower-release-changelog/scripts/validate_release_changelog.py \
  framework/docs/source/changelog/v<VERSION>.md \
  --expected-prs <ALL_PRE_EDIT_PR_NUMBERS> \
  --expected-incompatible-prs <ALL_INCOMPATIBLE_PR_NUMBERS>
```

Omit `--expected-incompatible-prs` when the final incompatible set is empty. Add to that set any unfinished PR classified as incompatible from a cached `break(...)` title. The validator checks only that PRs are complete and unique, each displayed PR number matches its link target, links within each item are in ascending PR-number order, incompatible PRs are in the right section, and generated headings other than `### UNGROUPED` are gone. Fix every validator error, then manually review the content and confirm the final diff changes only the intended release file.
