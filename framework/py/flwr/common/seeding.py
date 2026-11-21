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
"""Seed RNGs used by a Flower run."""


from __future__ import annotations

import hashlib
import random
import sys
from collections.abc import Iterable, Mapping
from logging import DEBUG, INFO
from typing import TypeVar

import numpy as np

from flwr.app.message import Context, Message
from flwr.common.constant import PARTITION_ID_KEY
from flwr.supercore import log

RANDOM_SEED_KEY = "random-seed"
_MAX_SEED = 2**32 - 1


class _SamplingState:
    """Hold the seeded sampler used for client selection.

    Heartbeat jitter and other framework code keep using the global ``random``
    module, so they cannot change which clients are selected.
    """

    generator: random.Random | None = None


_SAMPLING_STATE = _SamplingState()

_T = TypeVar("_T", int, str)


def read_random_seed(run_config: object) -> int | None:
    """Return ``random-seed`` from a run config.

    Parameters
    ----------
    run_config : object
        Value stored on ``Context.run_config``. Objects that are not mappings
        are treated as "no seed" so test doubles do not enable seeding.

    Returns
    -------
    int | None
        The seed, or ``None`` when the key is absent.

    Raises
    ------
    ValueError
        If the key is present but is not a non-negative 32-bit integer.
    """
    if not isinstance(run_config, Mapping) or RANDOM_SEED_KEY not in run_config:
        return None
    return _parse_seed(run_config[RANDOM_SEED_KEY])


def seed_random(seed: int) -> None:
    """Seed the process RNGs used by Flower and by user code.

    Parameters
    ----------
    seed : int
        Non-negative seed no larger than ``2**32 - 1``.

    Notes
    -----
    Python's ``random`` and the global NumPy RNG are always seeded. PyTorch
    and TensorFlow are seeded only when they are already in ``sys.modules``.
    This helper does not import them. A framework imported later stays
    unseeded until the next call. The app handlers make that call after the
    app module has been imported. Weights created while the module was
    importing the framework are not redone.
    """
    parsed = _parse_seed(seed)
    random.seed(parsed)
    _SAMPLING_STATE.generator = random.Random(parsed)
    np.random.seed(parsed)
    _seed_imported_frameworks(parsed)


def clear_run_seed() -> None:
    """Drop the dedicated client-sampling generator.

    Tests call this so a seeded run does not affect later cases. Experiment
    code does not need to call it.
    """
    _SAMPLING_STATE.generator = None


def derive_seed(base_seed: int, *parts: str) -> int:
    """Mix a base seed with identity strings into a 32-bit seed.

    Parameters
    ----------
    base_seed : int
        Run-level seed.
    *parts : str
        Strings that distinguish this RNG stream. Order is significant.

    Returns
    -------
    int
        Seed in the range ``[0, 2**32 - 1]``.
    """
    material = "|".join((str(_parse_seed(base_seed)), *parts))
    digest = hashlib.sha256(material.encode()).digest()
    return int.from_bytes(digest[:4], "little")


def sample_population(population: Iterable[_T], k: int) -> list[_T]:
    """Sample ``k`` candidates.

    Parameters
    ----------
    population : Iterable[int] | Iterable[str]
        Node ids or client ids to sample from.
    k : int
        Number of items to draw.

    Returns
    -------
    list[int] | list[str]
        Sampled items. When a run seed is active, candidates are sorted before
        sampling so set iteration order cannot change the result.

    Notes
    -----
    Without a run seed this calls :func:`random.sample` and preserves the
    input order, matching previous Flower behavior.
    """
    ordered: list[_T] = list(population)
    if _SAMPLING_STATE.generator is None:
        return random.sample(ordered, k)
    ordered.sort(key=_candidate_sort_key)
    return _SAMPLING_STATE.generator.sample(ordered, k)


def seed_server(run_config: object, *, process: str = "server") -> int | None:
    """Seed a Flower process from ``random-seed`` when it is configured.

    Parameters
    ----------
    run_config : object
        ``Context.run_config`` for this process.
    process : str (default: "server")
        Name used in the log line. ServerApp uses ``"server"`` and AgentApp
        uses ``"agent"``.

    Returns
    -------
    int | None
        The applied seed, or ``None`` when seeding is disabled. An absent key
        also clears a sampler left by an earlier seeded run in this process.
    """
    seed = read_random_seed(run_config)
    if seed is None:
        # A previous seeded run in this process must not keep the dedicated
        # sampler once this run opts out.
        clear_run_seed()
        return None
    seed_random(seed)
    log(INFO, "Seeded %s RNGs from run config %s=%s", process, RANDOM_SEED_KEY, seed)
    return seed


def seed_client(context: Context, message: Message) -> int | None:
    """Seed the client process for one message when ``random-seed`` is set.

    Parameters
    ----------
    context : Context
        Client context. ``partition-id`` identifies the client when present.
    message : Message
        Message about to be handled. Its type, group id, and ``server-round``
        config distinguish rounds and actions.

    Returns
    -------
    int | None
        The derived seed applied to this process, or ``None`` when the run
        config does not request seeding.
    """
    base_seed = read_random_seed(context.run_config)
    if base_seed is None:
        return None
    derived = client_seed(base_seed, context, message)
    seed_random(derived)
    log(
        DEBUG,
        "Seeded client RNGs from %s=%s (derived=%s)",
        RANDOM_SEED_KEY,
        base_seed,
        derived,
    )
    return derived


def client_seed(base_seed: int, context: Context, message: Message) -> int:
    """Derive the seed for one client message.

    Parameters
    ----------
    base_seed : int
        Run-level ``random-seed``.
    context : Context
        Client context.
    message : Message
        Message that is about to be handled.

    Returns
    -------
    int
        Deterministic 32-bit seed for this client message.
    """
    node_config = context.node_config
    if isinstance(node_config, Mapping) and PARTITION_ID_KEY in node_config:
        identity = f"partition:{node_config[PARTITION_ID_KEY]}"
    else:
        identity = f"node:{context.node_id}"
    return derive_seed(
        base_seed,
        identity,
        str(message.metadata.message_type),
        str(message.metadata.group_id),
        _server_round_token(message),
    )


def _parse_seed(value: object) -> int:
    """Validate a user-supplied seed.

    Parameters
    ----------
    value : object
        Config value to validate.

    Returns
    -------
    int
        The seed as an ``int``.

    Raises
    ------
    ValueError
        If ``value`` is not a non-negative integer within 32 bits.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{RANDOM_SEED_KEY} must be a non-negative integer, got {value!r}."
        )
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"{RANDOM_SEED_KEY} must be a non-negative integer, got {value!r}."
        )
    seed = int(value)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(
            f"{RANDOM_SEED_KEY} must be between 0 and {_MAX_SEED}, got {seed}."
        )
    return seed


def _candidate_sort_key(value: int | str) -> tuple[int, int | str, str]:
    """Return a sort key that orders numeric ids numerically.

    Parameters
    ----------
    value : int | str
        Node id or client id.

    Returns
    -------
    tuple[int, int | str, str]
        Sort key. ASCII decimal ids sort by numeric value, then by their
        original text, so ``"1"`` and ``"01"`` stay distinct. Other strings,
        including non-ASCII digits, keep their text order.
    """
    if isinstance(value, int):
        return (0, value, "")
    if value.isascii() and value.isdecimal():
        return (0, int(value), value)
    return (1, value, "")


def _server_round_token(message: Message) -> str:
    """Return ``server-round`` from the first config record that has it.

    Parameters
    ----------
    message : Message
        Message whose config records may contain ``server-round``.

    Returns
    -------
    str
        The round token, or an empty string when the key is absent.
    """
    if not message.has_content():
        return ""
    records = message.content.config_records
    if not isinstance(records, Mapping):
        return ""
    for record in records.values():
        if isinstance(record, Mapping) and "server-round" in record:
            return str(record["server-round"])
    return ""


def _seed_imported_frameworks(seed: int) -> None:
    """Seed PyTorch and TensorFlow when they are already in ``sys.modules``.

    Parameters
    ----------
    seed : int
        Seed passed to the framework helpers.

    Notes
    -----
    This does not import either library. A framework that appears after this
    call stays unseeded until the next call.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and bool(cuda.is_available()):
            cuda.manual_seed_all(seed)

    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        tensorflow.random.set_seed(seed)
