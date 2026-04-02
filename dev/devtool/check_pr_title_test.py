"""Tests for PR title validation."""

import sys

import pytest

from devtool import check_pr_title


def test_main_accepts_valid_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid title should pass without exiting."""
    monkeypatch.setattr(
        sys, "argv", ["python", "feat(framework): Add test-devtool pytest coverage"]
    )

    check_pr_title.main()


def test_main_rejects_star_without_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wildcard project must use :skip."""
    monkeypatch.setattr(sys, "argv", ["python", "feat(*): Add test-devtool pytest"])

    with pytest.raises(SystemExit) as exc_info:
        check_pr_title.main()

    assert exc_info.value.code == 1
