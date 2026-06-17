"""Tests for HTML theme updates."""

from pathlib import Path

import pytest

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


def test_update_conf_file_does_not_rewrite_when_no_changes_needed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing generated fields should not cause a file rewrite."""
    conf_file = tmp_path / "conf.py"
    content = """html_theme_options = {
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-announcement-text": "#ffffff",
    },
}
"""
    conf_file.write_text(content, encoding="utf-8")
    modified_time = conf_file.stat().st_mtime_ns

    update_html_themes.update_conf_file(
        conf_file,
        {
            "light_css_variables": {
                "color-announcement-background": "#292f36",
                "color-announcement-text": "#ffffff",
            },
        },
    )

    assert conf_file.read_text(encoding="utf-8") == content
    assert conf_file.stat().st_mtime_ns == modified_time
    assert f"No changes needed in: {conf_file}" in capsys.readouterr().out


def test_update_conf_file_reports_missing_theme_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Files without html_theme_options should still be reported distinctly."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text('html_theme = "furo"\n', encoding="utf-8")

    update_html_themes.update_conf_file(conf_file, {"announcement": "Banner"})

    assert (
        f"No html_theme_options block found in: {conf_file}" in capsys.readouterr().out
    )
