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
"""Custom types for Flower clients."""


from collections.abc import Callable

from flwr.app import Context, Message

from .client import Client as Client

# Compatibility
ClientFn = Callable[[str], Client]
ClientFnExt = Callable[[Context], Client]

ClientAppCallable = Callable[[Message, Context], Message]
Mod = Callable[[Message, Context, ClientAppCallable], Message]
