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
from rich.color import Color, ColorType
from rich.console import Console
from rich.markdown import Markdown

_ANSI_COLOR_NAMES = (
    "ansiblack",
    "ansired",
    "ansigreen",
    "ansiyellow",
    "ansiblue",
    "ansimagenta",
    "ansicyan",
    "ansiwhite",
    "ansibrightblack",
    "ansibrightred",
    "ansibrightgreen",
    "ansibrightyellow",
    "ansibrightblue",
    "ansibrightmagenta",
    "ansibrightcyan",
    "ansibrightwhite",
)


@dataclass
class MarkdownBlock:
    """Markdown-formatted assistant message shown in the transcript."""

    body: str = ""


def render_markdown(block: MarkdownBlock, width: int) -> StyleAndTextTuples:
    """Render Markdown as prompt_toolkit formatted-text fragments."""
    # Render Markdown with Rich using the transcript's current terminal width.
    console = Console(
        width=width,
        color_system="truecolor",
        force_terminal=True,
        markup=False,
    )
    fragments: StyleAndTextTuples = []
    links: dict[str, str] = {}
    for segment in console.render(Markdown(block.body), console.options):
        # Ignore Rich control sequences and segments without visible content.
        if segment.control or not segment.text:
            continue

        # Translate Rich text attributes to prompt_toolkit style syntax.
        style = segment.style
        link = style.link if style is not None else None
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

            # Keep ANSI colors theme-aware and convert extended colors to RGB.
            for color, prefix in ((style.color, "fg:"), (style.bgcolor, "bg:")):
                if color is None:
                    continue
                if color_value := _to_prompt_toolkit_color(color):
                    attributes.append(f"{prefix}{color_value}")
        fragments.append((" ".join(attributes), segment.text))
        if link is not None:
            links[link] = f"{links.get(link, '')}{segment.text}"

    # Keep destinations visible and copyable because prompt_toolkit fragments
    # cannot represent Rich's hyperlink metadata. Avoid repeating bare URLs.
    for link, text in links.items():
        if text.strip() != link:
            fragments.append(("", f"{link}\n"))

    # Rich terminates each rendered message with one newline. Retain the blank
    # row that separates messages in the transcript.
    fragments.append(("", "\n"))
    return fragments


def _to_prompt_toolkit_color(color: Color) -> str | None:
    """Translate a Rich color to prompt_toolkit style syntax."""
    if color.type == ColorType.DEFAULT:
        return None
    if color.type == ColorType.STANDARD:
        assert color.number is not None
        return _ANSI_COLOR_NAMES[color.number]
    triplet = color.get_truecolor()
    return f"#{triplet.red:02x}{triplet.green:02x}{triplet.blue:02x}"
