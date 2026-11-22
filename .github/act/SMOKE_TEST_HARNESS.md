# Smoke-Test Harness Guide

This directory provides a small local harness around `act` for validating that
GitHub Actions workflows are wired correctly for representative events.

The harness is intentionally a smoke-test layer. It is not a replacement for
GitHub-hosted CI, and it is not meant to duplicate every matrix entry or every
release permutation locally.

## What This Harness Checks

Use this harness to answer questions like:

- Does this workflow start for the intended event type?
- Does the event payload contain the fields the workflow/actions expect?
- Do repository variables, environment values, and secrets reach the job?
- Do reusable workflow inputs and secrets line up between caller and callee?
- Do publish jobs resolve the intended package registry, Docker registry, image
  namespace, tags, and credentials?
- Do early setup, validation, and matrix-generation jobs complete locally?

For publish workflows, this is especially useful because a large part of the
failure surface is configuration plumbing:

- `workflow_dispatch` inputs overriding repository `vars`
- Docker registry host, user, namespace, and token alignment
- Python package repository URL/name/user/token alignment
- reusable workflow `with` and `secrets` mappings
- generated Docker image repositories and tags

## Why Event Fixtures Are Committed

`act` creates synthetic GitHub events. Those events are often smaller than real
GitHub payloads, and many actions assume fields that only exist in the real
payload.

For example, `dorny/paths-filter` needs `repository.default_branch`. GitHub
provides that field, but a minimal `act` event may not. Keeping event payloads in
`.github/act/events/` makes that dependency explicit and repeatable.

Committed fixtures also make event-specific behavior reviewable. A workflow that
uses `github.event.pull_request.head.repo.fork`, `github.ref_name`, or
`github.event.inputs` should have a fixture that shows what local smoke tests
are simulating.

Use `*.local.json` for personal overrides. Those files are ignored by git.

## Why Profiles Exist

Generic local CI and publish validation should not share the same secret file.
Publish jobs can upload packages or push container images when supplied with
real credentials, so they need a more deliberate configuration boundary.

Profiles provide that boundary:

```text
.github/act/profiles/<profile>.env.local
.github/act/profiles/<profile>.vars.local
.github/act/profiles/<profile>.secrets.local
```

The publish wrapper requires a profile and fails early if no matching profile
files exist. This makes accidental publishing less likely and keeps credentials
for different registries separate.

Example profiles:

- `testpypi-gitlab`: package publish dry-runs with TestPyPI/devpi values and
  Docker image publishing to GitLab Container Registry
- `testpypi-dockerhub`: package publish dry-runs with TestPyPI/devpi values and
  Docker image publishing to Docker Hub

Use disposable repositories, namespaces, package versions, and tokens for these
profiles.

## Why Wrappers Are Thin Shell Scripts

The scripts in this directory only normalize repetitive `act` arguments:

- workflow path
- event fixture
- job selection
- runner image mapping
- env/vars/secrets files
- matrix narrowing

Keeping the scripts thin has two benefits:

- The real source of truth remains the GitHub workflow YAML.
- The command being run is still easy to reason about when debugging.

If a workflow needs special behavior to pass locally, prefer making that behavior
explicit in an event fixture or profile file rather than hiding it in shell
logic.

## Why Jobs Are Usually Selected Explicitly

Running an entire workflow through `act` can be slow and noisy. It can also cause
side effects for publish workflows.

Prefer running the smallest job that validates the layer you care about:

```bash
.github/act/run-framework.sh push-main changes
.github/act/run-publish.sh docker-main testpypi-gitlab prepare-docker-build-matrix
```

For matrix jobs, use one representative entry unless the change specifically
touches matrix behavior:

```bash
ACT_MATRIX=python:3.10 .github/act/run-framework.sh push-main test_core
```

GitHub-hosted CI remains responsible for the full matrix.

## Publish Workflow Strategy

Publish workflows should usually be tested in stages.

First, test non-publishing configuration jobs:

```bash
.github/act/run-publish.sh docker-main testpypi-gitlab prepare-docker-build-matrix
```

This validates Docker registry variables and generated image names/tags without
pushing images.

Second, dry-run dangerous jobs:

```bash
.github/act/run-publish.sh release testpypi-gitlab publish-wheel -- --dryrun
.github/act/run-publish.sh nightly testpypi-gitlab release-nightly -- --dryrun
```

This validates workflow rendering and job wiring without executing publish
commands.

Finally, run actual Docker publish jobs only when the profile points at
disposable test registries:

```bash
.github/act/run-publish.sh docker-main testpypi-gitlab all
```

The publish wrapper requires `--dryrun` for Python package publish jobs by
default. Set `ACT_ALLOW_PACKAGE_PUBLISH=1` only for an intentional publish to a
disposable package repository configured by the selected profile.

Do not use production PyPI or production container namespaces for local `act`
publish validation.

## Known Limits

`act` is close enough to catch many workflow wiring problems, but it is not a
perfect GitHub Actions runner.

Known differences include:

- runner images differ from GitHub-hosted images
- hosted cache behavior differs
- artifact behavior can differ
- permissions and default tokens are not identical
- service containers and Docker-in-Docker behavior may differ
- ARM runner labels need local image mapping
- some actions behave differently in post steps under lightweight images

For publish workflows, the harness defaults to `catthehacker/ubuntu:full-22.04`
because setup actions and post steps need a more complete runner image. The
generic workflow wrapper keeps the lighter `act-22.04` default for faster smoke
checks.

## When To Add More Fixtures Or Profiles

Add an event fixture when a workflow depends on a specific GitHub event shape.
Examples:

- pull request from a fork
- release event with a tag name
- scheduled nightly event
- workflow dispatch inputs

Add a profile when a workflow needs a different external integration target.
Examples:

- GitLab Container Registry vs Docker Hub
- TestPyPI vs devpi
- staging credentials vs disposable local-test credentials

Do not commit real credentials. Local profile files and local event overrides are
ignored by git.
