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
"""Current federation membership checks for run reads."""

from dataclasses import replace
from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    PullArtifactsRequest,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.federation import NoOpFederationManager

from .control_handlers import list_runs, pull_artifacts

_ACCOUNT = AccountInfo(flwr_aid=NOOP_FLWR_AID, account_name=NOOP_ACCOUNT_NAME)


@pytest.fixture(name="state", params=[FLWR_IN_MEMORY_DB_NAME, ":memory:"])
def state_fixture(request: pytest.FixtureRequest) -> LinkState:
    """Provide both in-memory and SQLite state with the default federation."""
    return LinkStateFactory(request.param, NoOpFederationManager(), Mock()).state()


def _create_run(state: LinkState, federation_id: str = NOOP_FEDERATION_ID) -> int:
    return state.create_run(
        "flwr/demo",
        "v0.0.1",
        "hash123",
        {},
        federation_id,
        None,
        _ACCOUNT.flwr_aid,
        TaskType.SERVER_APP,
    )


def test_list_runs_filters_current_membership_before_pagination(
    state: LinkState,
) -> None:
    """Hide submitted runs after membership ends and page accessible runs."""
    first_run_id = _create_run(state)
    second_run_id = _create_run(state)
    shared_federation = "@owner/shared"
    shared_run_id = _create_run(state, shared_federation)
    federations = state.federation_manager.get_federations(_ACCOUNT.flwr_aid)

    with patch.object(
        state.federation_manager,
        "get_federations",
        return_value=[*federations, replace(federations[0], id=shared_federation)],
    ) as get_federations:
        response = list_runs(ListRunsRequest(), _ACCOUNT, state)
        assert set(response.run_dict) == {first_run_id, second_run_id, shared_run_id}

        get_federations.return_value = federations
        response = list_runs(ListRunsRequest(), _ACCOUNT, state)
        assert set(response.run_dict) == {first_run_id, second_run_id}
        response = list_runs(ListRunsRequest(limit=1, skip=1), _ACCOUNT, state)
        assert set(response.run_dict) == {first_run_id}

        get_federations.return_value = []
        response = list_runs(ListRunsRequest(), _ACCOUNT, state)
        assert not response.run_dict


def test_list_owned_run_requires_current_membership(state: LinkState) -> None:
    """Require current membership even when listing a submitted run by ID."""
    run_id = _create_run(state)
    response = list_runs(ListRunsRequest(run_id=run_id), _ACCOUNT, state)
    assert set(response.run_dict) == {run_id}
    with patch.object(state.federation_manager, "has_member", return_value=False):
        with pytest.raises(FlowerError) as error:
            list_runs(ListRunsRequest(run_id=run_id), _ACCOUNT, state)
    assert error.value.code == ApiErrorCode.FEDERATION_NOT_FOUND


def test_pull_artifacts_requires_current_membership(state: LinkState) -> None:
    """Allow owned artifacts until membership ends, then never issue a URL."""
    run_id = _create_run(state)
    state.stop_run(run_id)
    provider = Mock(get_url=Mock(return_value="https://artifacts.example/run.zip"))
    request = PullArtifactsRequest(run_id=run_id)

    response = pull_artifacts(request, _ACCOUNT, state, provider)
    assert response.url == "https://artifacts.example/run.zip"
    provider.get_url.assert_called_once_with(run_id)
    provider.get_url.reset_mock()

    with patch.object(state.federation_manager, "has_member", return_value=False):
        with pytest.raises(FlowerError) as error:
            pull_artifacts(request, _ACCOUNT, state, provider)
    assert error.value.code == ApiErrorCode.FEDERATION_NOT_FOUND
    provider.get_url.assert_not_called()


def test_pull_artifacts_still_requires_run_ownership(state: LinkState) -> None:
    """Federation membership alone does not authorize artifact downloads."""
    run_id = _create_run(state)
    state.stop_run(run_id)
    provider = Mock()
    with patch.object(state.federation_manager, "has_member", return_value=True):
        with pytest.raises(FlowerError) as error:
            pull_artifacts(
                PullArtifactsRequest(run_id=run_id),
                replace(_ACCOUNT, flwr_aid="other-account"),
                state,
                provider,
            )
    assert error.value.code == ApiErrorCode.RUN_ID_NOT_BELONG_TO_ACCOUNT
    provider.get_url.assert_not_called()


def test_owned_space_runs_keep_listing_and_artifact_access(state: LinkState) -> None:
    """Keep owned Space runs visible without requiring federation membership."""
    federation_run_id = _create_run(state)
    removed_run_id = _create_run(state, "@owner/removed")
    space_id = UUID(state.create_space(participants=[]).id)
    space_run_ids = [
        state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            None,
            None,
            flwr_aid,
            TaskType.AGENT_APP,
            space_id=space_id,
            user_prompt="Hello",
        )
        for flwr_aid in [_ACCOUNT.flwr_aid, "other-account"]
    ]
    owned_space_run_id, other_space_run_id = space_run_ids

    # Scope selection and caller ownership apply before ordering and pagination.
    response = list_runs(ListRunsRequest(limit=1), _ACCOUNT, state)
    assert set(response.run_dict) == {owned_space_run_id}
    response = list_runs(ListRunsRequest(limit=1, skip=1), _ACCOUNT, state)
    assert set(response.run_dict) == {federation_run_id}
    with patch.object(state.federation_manager, "get_federations", return_value=[]):
        response = list_runs(ListRunsRequest(), _ACCOUNT, state)
    assert set(response.run_dict) == {owned_space_run_id}
    assert removed_run_id not in response.run_dict
    assert other_space_run_id not in response.run_dict

    # A Space run submitter can still obtain artifacts without a federation.
    state.stop_run(owned_space_run_id)
    provider = Mock(get_url=Mock(return_value="https://artifacts.example/space.zip"))
    with patch.object(state.federation_manager, "has_member") as has_member:
        response_artifacts = pull_artifacts(
            PullArtifactsRequest(run_id=owned_space_run_id), _ACCOUNT, state, provider
        )
    assert response_artifacts.url == "https://artifacts.example/space.zip"
    has_member.assert_not_called()
    provider.get_url.assert_called_once_with(owned_space_run_id)

    state.stop_run(other_space_run_id)
    with pytest.raises(FlowerError) as error:
        pull_artifacts(
            PullArtifactsRequest(run_id=other_space_run_id), _ACCOUNT, state, provider
        )
    assert error.value.code == ApiErrorCode.RUN_ID_NOT_BELONG_TO_ACCOUNT

    # Empty filters still select no runs; an explicit None entry selects Space runs.
    assert not state.get_run_info(federation_ids=[])
    assert {run.run_id for run in state.get_run_info(federation_ids=[None])} == set(
        space_run_ids
    )
    assert {run.run_id for run in state.get_run_info(federation_ids=None)} == {
        federation_run_id,
        removed_run_id,
        *space_run_ids,
    }
