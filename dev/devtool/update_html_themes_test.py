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


def test_update_conf_file_skips_complete_theme_variables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Complete theme dictionaries should not be rewritten."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-announcement-background": "#17222d",
        "color-announcement-text": "#ffffff",
    },
}
""",
        encoding="utf-8",
    )
    original_content = conf_file.read_text(encoding="utf-8")

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
        },
    )

    assert conf_file.read_text(encoding="utf-8") == original_content
    assert capsys.readouterr().out == f"No changes needed in: {conf_file}\n"


def test_update_conf_file_skips_existing_top_level_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing top-level generated fields should not be duplicated."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "announcement": "Existing banner",
}
""",
        encoding="utf-8",
    )
    original_content = conf_file.read_text(encoding="utf-8")

    update_html_themes.update_conf_file(
        conf_file,
        {
            "announcement": "Generated banner",
        },
    )

    assert conf_file.read_text(encoding="utf-8") == original_content
    assert capsys.readouterr().out == f"No changes needed in: {conf_file}\n"


def test_update_conf_file_adds_comma_before_merged_theme_variables(
    tmp_path: Path,
) -> None:
    """Merged theme variables should keep valid Python without trailing commas."""
    conf_file = tmp_path / "conf.py"
    conf_file.write_text(
        """html_theme_options = {
    "light_css_variables": {
        "color-announcement-background": "#17222d"
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
        },
    )

    assert (
        conf_file.read_text(encoding="utf-8")
        == """html_theme_options = {
    "light_css_variables": {
        "color-announcement-background": "#17222d",
        "color-announcement-text": "#ffffff",
    },
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
