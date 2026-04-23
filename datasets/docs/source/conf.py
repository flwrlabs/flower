# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# Fixing path issue for autodoc
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../flwr_datasets"))


# -- Project information -----------------------------------------------------

project = "Flower Datasets"
copyright = f"{datetime.date.today().year} Flower Labs GmbH"
author = "The Flower Authors"

# Read version from pyproject.toml (single source of truth)
_pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(_pyproject_path, "rb") as f:
    _pyproject = tomllib.load(f)
release = _pyproject["project"]["version"]

# Make version available as a substitution in rst files (e.g., |release|)
rst_prolog = f"""
.. |release| replace:: {release}
"""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinxarg.ext",
    "myst_parser",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_reredirects",
    "nbsphinx",
]

# Generate .rst files
autosummary_generate = True

# Document ONLY the objects from __all__ (present in __init__ files).
# It will be done recursively starting from flwr_dataset.__init__
# It's controlled in the index.rst file.
autosummary_ignore_module_all = False

# Each class and function docs start with the path to it
# Make the flwr_datasets.federated_dataset.FederatedDataset appear as FederatedDataset
# The full name is still at the top of the page
add_module_names = False


def find_test_modules(package_path):
    """Go through the python files and exclude every *_test.py file."""
    full_path_modules = []
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith("_test.py"):
                # Construct the module path relative to the package directory
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, package_path)
                # Convert file path to dotted module path
                module_path = os.path.splitext(relative_path)[0].replace(os.sep, ".")
                full_path_modules.append(module_path)
    modules = []
    for full_path_module in full_path_modules:
        parts = full_path_module.split(".")
        for i in range(len(parts)):
            modules.append(".".join(parts[i:]))
    return modules


# Stop from documenting the *_test.py files.
# That's the only way to do that in autosummary (make the modules as mock_imports).
autodoc_mock_imports = find_test_modules(os.path.abspath("../../"))

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Sphinx redirects, implemented after the doc filename changes.
# To prevent 404 errors and redirect to the new pages.
redirects = {
    "how-to-visualize-label-distribution": "tutorial-visualize-label-distribution.html",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = f"Flower Datasets {release}"
html_favicon = "_static/favicon.ico"
html_baseurl = "https://flower.ai/docs/datasets/"

html_theme_options = {
    #
    # Sphinx Book Theme
    #
    # https://sphinx-book-theme.readthedocs.io/en/latest/configure.html
    # "repository_url": "https://github.com/flwrlabs/flower",
    # "repository_branch": "main",
    # "path_to_docs": "doc/source/",
    # "home_page_in_toc": True,
    # "use_repository_button": True,
    # "use_issues_button": True,
    # "use_edit_page_button": True,
    #
    # Furo
    #
    # https://pradyunsg.me/furo/customisation/
    # "light_css_variables": {
    #     "color-brand-primary": "#292F36",
    #     "color-brand-content": "#292F36",
    #     "color-admonition-background": "#F2B705",
    # },
    "light_logo": "datasets-light-mode.png",
    "dark_logo": "datasets-dark-mode.png",
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
