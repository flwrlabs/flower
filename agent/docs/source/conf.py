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

"""Sphinx configuration for the standalone Flower Agent documentation."""

import datetime

project = "Flower Agent"
copyright = f"{datetime.date.today().year} Flower Labs GmbH"
author = "The Flower Authors"

extensions = ["myst_parser"]

source_suffix = {
    ".md": "markdown",
}
root_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "Flower Agent"
html_favicon = "_static/favicon.ico"
html_baseurl = "https://flower.ai/docs/agent/"

html_theme_options = {
    "light_logo": "flower-agent-logo-light.png",
    "dark_logo": "flower-agent-logo-dark.png",
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-announcement-text": "#ffffff",
        # Left sidebar
        "color-sidebar-link-text": "#5e5e5e",
        "color-sidebar-link-text--top-level": "#404040",
        "color-sidebar-item-background--hover": "#e5e5e5",
        "color-sidebar-search-background": "#f2f2f2",
        "color-sidebar-search-background--focus": "#e2e2e2",
        "color-sidebar-background": "#f2f2f2",
        # Right sidebar (On this page)
        "color-toc-item-text--active": "#404040",
    },
    "dark_css_variables": {
        "color-announcement-text": "#ffffff",
        "color-announcement-background": "#17222d",
        # Left sidebar
        "color-sidebar-link-text": "#ffffff",
        "color-sidebar-link-text--top-level": "#ababab",
        "color-sidebar-item-background--hover": "#222222",
        "color-sidebar-background": "#161616",
        "color-sidebar-search-background": "#161616",
        "color-sidebar-search-background--focus": "#1c1c1c",
        # Right sidebar (On this page)
        "color-toc-title-text": "#ffffff",
        "color-toc-item-text": "#ababab",
        "color-toc-item-text--hover": "#d2d2d2",
        "color-toc-item-text--active": "#fff5bf",
    },
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
