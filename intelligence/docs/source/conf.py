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


import datetime
import os

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Fixing path issue for autodoc
# sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "Flower Intelligence"
copyright = f"{datetime.date.today().year} Flower Labs GmbH"
author = "The Flower Authors"

# The full version, including alpha/beta/rc tags
release = "0.2.6"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_reredirects",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Flower Intelligence {release}"
html_favicon = "_static/favicon.ico"
html_baseurl = "https://flower.ai/docs/intelligence/"

html_theme_options = {
    "light_logo": "fi-light-mode.png",
    "dark_logo": "fi-dark-mode.png",
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
        "color-sidebar-link-text": "#7c7c7c",
        "color-sidebar-link-text--top-level": "#ababab",
        "color-sidebar-item-background--hover": "#222222",
        "color-sidebar-background": "#161616",
        "color-sidebar-search-background": "#161616",
        "color-sidebar-search-background--focus": "#1c1c1c",
        # Right sidebar (On this page)
        "color-toc-title-text": "#7c7c7c",
        "color-toc-item-text": "#ababab",
        "color-toc-item-text--hover": "#d2d2d2",
        "color-toc-item-text--active": "#fff5bf",
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for nbsphinx -------------------------------------------------

nbsphinx_execute = "never"

_open_in_colab_button = """
.. raw:: html

    <br/>
    <a href="https://colab.research.google.com/github/flwrlabs/flower/blob/main/datasets/docs/source/{{ env.doc2path(env.docname, base=None) }}">
        <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"/>
    </a>
"""
nbsphinx_prolog = _open_in_colab_button
nbsphinx_epilog = _open_in_colab_button

# -- Options for sphinxcontrib-mermaid -------------------------------------
# Don't load it automatically through the extension as we are loading it through the
# theme (see base.html) as the inclusion of require.js by the extension `nbsphinx`
# breaks the way mermaid is loaded. The solution is to load mermaid before the
# require.js script added by `nbsphinx`. We can only enforce this in the theme
# itself.
mermaid_version = ""

# -- Options for MyST config  -------------------------------------
# Enable this option to link to headers (`#`, `##`, or `###`)
myst_heading_anchors = 3

# -- Options for sphinx_copybutton -------------------------------------
copybutton_exclude = ".linenos, .gp, .go"
copybutton_prompt_text = ">>> "
