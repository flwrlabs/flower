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
"""Register the Google Workspace connectors as one package."""

from .calendar import CONNECTOR as GOOGLE_CALENDAR_CONNECTOR
from .drive import CONNECTOR as GOOGLE_DRIVE_CONNECTOR
from .gmail import CONNECTOR as GMAIL_CONNECTOR

CONNECTORS = (GMAIL_CONNECTOR, GOOGLE_CALENDAR_CONNECTOR, GOOGLE_DRIVE_CONNECTOR)
