"""Tests for HTML theme updates."""

from pathlib import Path

from devtool import update_html_themes


def test_update_conf_file_inserts_fields_after_nested_theme_options(
    tmp_path: Path,
) -> None:
    """Generated fields should stay at the top level of html_theme_options."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "light_logo": "examples-light-mode.png",
    "light_css_variables": {
        "color-sidebar-background": "#f2f2f2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#161616",
    },
}
""",
        encoding="utf-8",
    )

    update_html_themes.update_conf_file(conf_file, '"announcement": "Banner"')

    assert (
        conf_file.read_text(encoding="utf-8")
        == """html_theme_options = {
    "light_logo": "examples-light-mode.png",
    "light_css_variables": {
        "color-sidebar-background": "#f2f2f2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#161616",
    },
    "announcement": "Banner",
}
"""
    )
