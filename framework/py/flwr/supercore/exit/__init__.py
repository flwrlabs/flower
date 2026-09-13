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
"""Flower exit functionality."""


from .exit import flwr_exit
from .exit_code import ExitCode
from .exit_handler import add_exit_handler
from .signal_handler import register_signal_handlers

__all__ = [
    "ExitCode",
    "add_exit_handler",
    "flwr_exit",
    "register_signal_handlers",
]
