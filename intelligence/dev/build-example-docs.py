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
"""Build the examples docs."""

import os
import re
import shutil

from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = os.path.join(ROOT, "docs", "source", "examples.rst")

initial_text = """
Examples
========

Below you will find a list of Flower Intelligence examples for Node.js and for
web apps  running in the browser (as a tab or WebExtension).

Node.js Examples
----------------

Those examples will run in the terminal and are mostly there to showcase some
features with very low overhead. You'll find more instruction in the
respective pages.

"""

table_headers = (
    "\n.. list-table::\n   :widths: 50 45 \n   "
    ":header-rows: 1\n\n   * - Title\n     - Tags\n\n"
)

categories = {
    "node": {"table": table_headers, "list": ""},
    "web": {"table": table_headers, "list": ""},
}


def _read_metadata(example):
    with open(os.path.join(example, "README.md")) as f:
        content = f.read()

    metadata_match = re.search(r"^---(.*?)^---", content, re.DOTALL | re.MULTILINE)
    if not metadata_match:
        raise ValueError("Metadata block not found")
    metadata = metadata_match.group(1)

    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if not title_match:
        raise ValueError("Title not found in metadata")
    title = title_match.group(1).strip()

    tags_match = re.search(r"^tags:\s*\[(.+?)\]$", metadata, re.MULTILINE)
    if not tags_match:
        raise ValueError("Tags not found in metadata")
    tags = tags_match.group(1).strip()

    return title, tags


def _add_table_entry(example, tag, table_var):
    title, tags = _read_metadata(example)
    example_name = Path(example).stem
    table_entry = f"   * - `{title} <{example_name}.html>`_ \n     - {tags}\n\n"
    if tag in tags:
        categories[table_var]["table"] += table_entry
        categories[table_var]["list"] += f"  {example_name}\n"
        return True
    return False


def _copy_markdown_files(example):
    for file in os.listdir(example):
        if file.endswith(".md"):
            src = os.path.join(example, file)
            dest = os.path.join(
                ROOT, "docs", "source", os.path.basename(example) + ".md"
            )
            shutil.copyfile(src, dest)


def _add_gh_button(example):
    gh_text = f'[<img src="_static/view-gh.png" alt="View on GitHub" width="200"/>](https://github.com/flwrlabs/flower/blob/main/intelligence/ts/examples/{example})'
    readme_file = os.path.join(ROOT, "docs", "source", example + ".md")
    with open(readme_file, "r+") as f:
        content = f.read()
        if gh_text not in content:
            content = re.sub(
                r"(^# .+$)", rf"\1\n\n{gh_text}", content, count=1, flags=re.MULTILINE
            )
            f.seek(0)
            f.write(content)
            f.truncate()


def _main():
    if os.path.exists(INDEX):
        os.remove(INDEX)

    with open(INDEX, "w") as index_file:
        index_file.write(initial_text)

    examples_dir = os.path.join(ROOT, "ts", "examples")
    for example in sorted(os.listdir(examples_dir)):
        example_path = os.path.join(examples_dir, example)
        if os.path.isdir(example_path):
            _copy_markdown_files(example_path)
            _add_gh_button(example)
            if not _add_table_entry(example_path, "node", "node"):
                _add_table_entry(example_path, "web", "web")

    with open(INDEX, "a") as index_file:
        index_file.write(categories["node"]["table"])

        index_file.write("\nWeb Examples\n------------\n")
        index_file.write(
            "Those examples will require you to use a browser. You'll find "
            "more instructions in the respective pages.\n"
        )
        index_file.write(categories["web"]["table"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Quickstart\n  :hidden:\n\n"
        )
        index_file.write(categories["node"]["list"])

        index_file.write(
            "\n.. toctree::\n  :maxdepth: 1\n  :caption: Advanced\n  :hidden:\n\n"
        )
        index_file.write(categories["web"]["list"])

        index_file.write("\n")


if __name__ == "__main__":
    _main()
