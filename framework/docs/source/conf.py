# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


import os
import sys

# Add current directory to sys.path to import conf_base.py
sys.path.insert(0, os.path.dirname(__file__))
from conf_base import *  # noqa: F403

# The full version of the next release, including alpha/beta/rc tags
release = "1.29.0"
# The current released version
rst_prolog = """
.. |stable_flwr_version| replace:: 1.29.0
.. The SuperLink Docker image digest is version-independent and does not necessarily track |stable_flwr_version|.
.. |stable_flwr_superlink_docker_digest| replace:: 4b317d5b6030710b476f4dbfab2c3a33021ad40a0fcfa54d7edd45e0c51d889c
.. |ubuntu_version| replace:: 24.04
.. |setuptools_version| replace:: 82.0.0
.. |pip_version| replace:: 26.0.1
.. |python_version| replace:: 3.10
.. |python_full_version| replace:: 3.10.19
"""

# Sphinx redirects, implemented after the doc filename changes.
# To prevent 404 errors and redirect to the new pages.
redirects = {
    **redirects,  # Keep existing redirects from conf_base.py
    # Renamed pages
    "how-to-authenticate-users": "how-to-authenticate-accounts.html",
    # Restructuring: contributor references
    "secagg": "explanation-ref-secure-aggregation-protocols.html",
}
