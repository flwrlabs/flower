"""Tests for HTML theme updates."""

from pathlib import Path

from devtool import update_html_themes


def test_update_conf_file_merges_nested_theme_variables(
    tmp_path: Path,
) -> None:
    """Generated theme variables should not overwrite existing values."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "light_logo": "examples-light-mode.png",
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-sidebar-background": "#f2f2f2",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#161616",
    },
}
""",
        encoding="utf-8",
    )

    update_html_themes.update_conf_file(
        conf_file,
        {
            "light_css_variables": {
                "color-announcement-background": "#292f36",
                "color-announcement-text": "#ffffff",
            },
            "dark_css_variables": {
                "color-announcement-background": "#292f36",
                "color-announcement-text": "#ffffff",
            },
            "announcement": "Banner",
        },
    )

    assert (
        conf_file.read_text(encoding="utf-8")
        == """html_theme_options = {
    "light_logo": "examples-light-mode.png",
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-sidebar-background": "#f2f2f2",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-sidebar-background": "#161616",
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff",
    },
    "announcement": "Banner",
}
"""
    )


def test_update_conf_file_appends_missing_theme_variable_dictionaries(
    tmp_path: Path,
) -> None:
    """Generated theme dictionaries should still be appended when missing."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "light_logo": "model-light-mode.png",
    "dark_logo": "model-dark-mode.png",
}
""",
        encoding="utf-8",
    )

    update_html_themes.update_conf_file(
        conf_file,
        {
            "light_css_variables": {
                "color-announcement-background": "#292f36",
            },
            "dark_css_variables": {
                "color-announcement-background": "#292f36",
            },
            "announcement": "Banner",
        },
    )

    assert (
        conf_file.read_text(encoding="utf-8")
        == """html_theme_options = {
    "light_logo": "model-light-mode.png",
    "dark_logo": "model-dark-mode.png",
    "light_css_variables": {
        "color-announcement-background": "#292f36"
    },
    "dark_css_variables": {
        "color-announcement-background": "#292f36"
    },
    "announcement": "Banner",
}
"""
    )
