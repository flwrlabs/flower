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
"""Flower task process components."""


<<<<<<< HEAD:framework/py/flwr/supercore/task_process/__init__.py
from .agent import run_agentapp
from .model import run_model
=======
from .model.run import run_model
from .run_agentapp import run_agentapp
>>>>>>> refs/remotes/origin/add-flwr-model-logic:framework/py/flwr/supercore/executors/__init__.py

__all__ = [
    "run_agentapp",
    "run_model",
]
