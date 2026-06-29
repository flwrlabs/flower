# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `federation` command."""


from . import invite as invite
from .add_supernode import add_supernode as add_supernode
from .archive import archive as archive
from .create import create as create
from .ls import ls as ls
from .remove_account import remove_account as remove_account
from .remove_supernode import remove_supernode as remove_supernode
from .simulation_config import simulation_config as simulation_config

__all__ = [
    "add_supernode",
    "archive",
    "create",
    "invite",
    "ls",
    "remove_account",
    "remove_supernode",
    "simulation_config",
]
