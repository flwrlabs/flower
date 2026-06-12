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
"""Flower Built-in Mods."""


from flwr.clientapp.mod import arrays_size_mod, message_size_mod

from .centraldp_mods import adaptiveclipping_mod, fixedclipping_mod
from .localdp_mod import LocalDpMod
from .secure_aggregation import secagg_mod, secaggplus_mod
from .utils import make_ffn

__all__ = [
    "LocalDpMod",
    "adaptiveclipping_mod",
    "arrays_size_mod",
    "fixedclipping_mod",
    "make_ffn",
    "message_size_mod",
    "secagg_mod",
    "secaggplus_mod",
]
