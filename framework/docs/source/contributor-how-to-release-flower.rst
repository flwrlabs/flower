################
 Release Flower
################

Framework minor releases are mostly automated. The manual release process has two steps:

1. Trigger the release preparation workflow to create a release pull request.
2. Review the generated changelog, then approve and merge the pull request.

After the release pull request is merged, the remaining release steps are performed
automatically by GitHub Actions.

************************
 Prepare the release PR
************************

Trigger the release preparation workflow using the GitHub CLI:

.. code-block:: bash

    gh workflow run framework-release-prepare.yml \
      --repo flwrlabs/flower \
      -f version=X.Y.0

Alternatively, trigger the `Framework Prepare Minor Release workflow
<https://github.com/flwrlabs/flower/actions/workflows/framework-release-prepare.yml>`_
from the GitHub web UI.

The version must use the ``X.Y.0`` format, for example ``1.34.0``.

The workflow pins the current ``main`` commit as the release source and creates a draft
release PR. It generates and polishes the changelog and updates the version bookkeeping
needed for the release and the next development cycle.

If more changes land on ``main`` before the release PR is merged, trigger the workflow
again with the same version. The existing release PR is refreshed instead of creating a
new one, and the release source is re-pinned to the current ``main`` HEAD.

The release PR is also checked automatically. These checks validate the generated
version state and changelog and verify that the prebuilt release artifacts for the
pinned source commit are available.

*************************
 Review and merge the PR
*************************

Review the generated changelog in ``framework/docs/source/changelog/vX.Y.0.md`` and make
any edits needed before the release. When the PR is ready, mark it ready for review if
it is still a draft, approve it, and merge it into ``main``.

That is the end of the manual release process. Do not manually create release tags or
publish Python packages, Docker images, or a GitHub Release.

******************************
 What happens after the merge
******************************

Merging the release PR automatically triggers ``framework-release-finalize.yml``. The
workflow uses the release source commit recorded by the latest release preparation run
and:

- creates the ``framework-X.Y.0`` tag and the ``release/framework-X.Y`` maintenance
  branch;
- publishes the Python wheel and source distribution;
- promotes the prebuilt Docker images to their stable release tags;
- dispatches the Framework documentation build from the release branch; and
- publishes the GitHub Release using the release changelog as the release notes.

The release artifacts are built ahead of time for commits on ``main`` by
``framework-commit-artifacts.yml``. The finalization workflow promotes the artifacts for
the exact pinned release source instead of rebuilding them during the release.
