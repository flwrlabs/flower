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
"""Transcript blocks and rendering for Flower Chat."""

from dataclasses import dataclass

from prompt_toolkit.formatted_text import StyleAndTextTuples
from rich.console import Console
from rich.markdown import Markdown


@dataclass
class MarkdownBlock:
    """Markdown-formatted assistant message shown in the transcript."""

    body: str = ""


def render_markdown(block: MarkdownBlock, width: int) -> StyleAndTextTuples:
    """Render Markdown as prompt_toolkit formatted-text fragments."""
    # Render Markdown with Rich using the transcript's current terminal width.
    console = Console(
        width=max(1, width),
        color_system="truecolor",
        force_terminal=True,
        markup=False,
    )
    fragments: StyleAndTextTuples = []
    for segment in console.render(Markdown(block.body), console.options):
        # Ignore Rich control sequences and segments without visible content.
        if segment.control or not segment.text:
            continue

        # Translate Rich text attributes to prompt_toolkit style syntax.
        style = segment.style
        if isinstance(style, str):
            style = console.get_style(style)
        attributes: list[str] = []
        if style is not None:
            for enabled, name in (
                (style.bold, "bold"),
                (style.italic, "italic"),
                (style.underline, "underline"),
                (style.strike, "strike"),
            ):
                if enabled:
                    attributes.append(name)

            # Preserve Rich's foreground and background colors as truecolor values.
            for color, prefix in ((style.color, "fg:"), (style.bgcolor, "bg:")):
                if color is None:
                    continue
                triplet = color.get_truecolor()
                if triplet is not None:
                    attributes.append(
                        f"{prefix}#{triplet.red:02x}{triplet.green:02x}"
                        f"{triplet.blue:02x}"
                    )
        fragments.append((" ".join(attributes), segment.text))

    # Rich terminates each rendered message with one newline. Retain the blank
    # row that separates messages in the transcript.
    fragments.append(("", "\n"))
    return fragments
