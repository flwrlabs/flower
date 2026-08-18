"""Tests for Flower App Bundle file selection."""

from io import BytesIO
from zipfile import ZipFile

from flwr.cli.build import build_fab_from_files


def test_build_fab_includes_jinja_templates() -> None:
    """FABs include scheduler templates used by deployed apps."""
    fab_bytes = build_fab_from_files(
        {
            "pyproject.toml": b"[project]\nname='test-app'\nversion='1.0.0'\n",
            "app.py": b"print('ok')\n",
            "templates/train.sh.j2": b"#!/usr/bin/env bash\n",
        }
    )

    with ZipFile(BytesIO(fab_bytes)) as fab:
        assert "templates/train.sh.j2" in fab.namelist()
