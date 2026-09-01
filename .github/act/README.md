# Local GitHub Actions with act

This directory contains small wrappers for smoke-testing GitHub Actions locally
with `act`.

The wrappers always pass a committed event fixture and optionally pass local
environment, variable, and secret files:

- `.github/act/env.local` via `--env-file`
- `.github/act/vars.local` via `--var-file`
- `.github/act/secrets.local` via `--secret-file`

The `*.local` files are ignored by git. Copy the matching `*.example` file and
fill in only the values required by the workflow/job you are running.

```bash
cp .github/act/env.example .github/act/env.local
cp .github/act/vars.example .github/act/vars.local
cp .github/act/secrets.example .github/act/secrets.local
```

Run the framework workflow path-filter job:

```bash
.github/act/run-framework.sh push-main changes
```

Run one framework matrix entry:

```bash
ACT_MATRIX=python:3.10 .github/act/run-framework.sh push-main test_core
```

Run an arbitrary workflow:

```bash
.github/act/run.sh push-main .github/workflows/framework-test.yml changes
```

Use `all` as the job argument to let `act` choose all runnable jobs for the
workflow/event:

```bash
.github/act/run.sh push-main .github/workflows/framework-test.yml all
```

## Publish Workflows

Publish workflows should use named profiles so package and container
credentials stay separate from generic local CI settings.

Create a GitLab Container Registry + package dry-run profile:

```bash
cp .github/act/profiles/publish.env.example .github/act/profiles/testpypi-gitlab.env.local
cp .github/act/profiles/publish-gitlab.vars.example .github/act/profiles/testpypi-gitlab.vars.local
cp .github/act/profiles/publish.secrets.example .github/act/profiles/testpypi-gitlab.secrets.local
```

Create a Docker Hub + package dry-run profile:

```bash
cp .github/act/profiles/publish.env.example .github/act/profiles/testpypi-dockerhub.env.local
cp .github/act/profiles/publish-dockerhub.vars.example .github/act/profiles/testpypi-dockerhub.vars.local
cp .github/act/profiles/publish.secrets.example .github/act/profiles/testpypi-dockerhub.secrets.local
```

Then edit the copied `*.local` files. For Docker, set:

```bash
DOCKER_IMAGE_REGISTRY=registry.gitlab.com
DOCKER_IMAGE_REGISTRY_USER=<gitlab-user-or-deploy-token-user>
DOCKER_IMAGE_NAMESPACE=<gitlab-group>/<gitlab-project>/flower-act
DOCKER_IMAGE_REGISTRY_PASSWORD=<token>
```

For Python package dry-runs, set:

```bash
PYPI_REPOSITORY_NAME=testpypi
PYPI_REPOSITORY_URL=https://test.pypi.org/legacy/
PYPI_REPOSITORY_USERNAME=__token__
PYPI_REPOSITORY_PASSWORD=<token>
```

Run publish workflow checks:

```bash
.github/act/run-publish.sh docker-main testpypi-gitlab prepare-docker-build-matrix
.github/act/run-publish.sh release testpypi-gitlab publish-wheel -- --dryrun
.github/act/run-publish.sh nightly testpypi-gitlab release-nightly -- --dryrun
```

Run the Docker publishing jobs for the configured test registry:

```bash
.github/act/run-publish.sh docker-main testpypi-gitlab all
```

The Docker build jobs push images when supplied with valid Docker credentials.
Use disposable repositories/tags for local validation.

The `release` and `nightly` Python package jobs require `--dryrun` by default.
Set `ACT_ALLOW_PACKAGE_PUBLISH=1` only for an intentional publish to the
configured non-production package repository. Never use production package
credentials for local `act` validation.

Publish runs default to `catthehacker/ubuntu:full-22.04` because these workflows
use setup actions and post steps that need a more complete runner image. Override
with `ACT_UBUNTU_22_04_IMAGE=...` if you need a different image.

Notes:

- Event fixtures live in `.github/act/events/`.
- `push-main.json` includes `repository.default_branch`, which is required by
  `dorny/paths-filter` under `act`.
- Publish event fixtures leave registry inputs empty so workflows read values
  from the selected profile's `vars.local`. Copy an event fixture to
  `.github/act/events/*.local.json` if you want to test explicit
  `workflow_dispatch` input overrides.
- Release/deploy workflows may contact external services. Use dummy secrets for
  wiring checks and real secrets only when you intentionally want those side
  effects.
