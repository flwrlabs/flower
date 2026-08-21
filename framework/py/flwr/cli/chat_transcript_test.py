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
"""Tests for Flower Chat transcript rendering."""

from flwr.cli.chat_transcript import MarkdownBlock, render_markdown


def test_render_markdown() -> None:
    """Markdown should produce prompt_toolkit style fragments."""
    fragments = render_markdown(MarkdownBlock("Hello **bold** and `code`."), 60)

    assert "**" not in "".join(text for _, text, *_ in fragments)
    assert any(text == "bold" and "bold" in style for style, text, *_ in fragments)
    assert any(text == "code" and "bg:" in style for style, text, *_ in fragments)
