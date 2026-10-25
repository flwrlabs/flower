#!/bin/bash

# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

taplo fmt

# Python
echo "Formatting started: Python"
python -m devtool.check_copyright flwr_datasets/
python -m isort flwr_datasets/
python -m black -q flwr_datasets/
python -m ruff check --fix flwr_datasets/
echo "Formatting done: Python"

# Notebooks
echo "Formatting started: Notebooks"
python -m black --ipynb -q docs/source/*.ipynb
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode cell.metadata.pycharm"
python -m nbstripout --keep-output docs/source/*.ipynb --extra-keys "$KEYS"
echo "Formatting done: Notebooks"
