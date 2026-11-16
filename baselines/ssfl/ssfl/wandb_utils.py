"""Weights & Biases logging (opt-in via wandb-mode)."""

from __future__ import annotations

from logging import INFO, WARNING
from typing import Any

from flwr.common import log


class WandbSession:
    """No-op unless wandb-mode is online or offline."""

    def __init__(self) -> None:
        self._run: Any | None = None
        self.enabled = False

    def start(self, cfg: dict[str, Any]) -> None:
        """Start a W&B run from the application config when logging is enabled."""
        mode = str(cfg.get("wandb-mode", "disabled")).lower()
        if mode in {"", "disabled", "off", "false", "none"}:
            return
        try:
            import wandb  # pylint: disable=import-outside-toplevel
        except ImportError:
            # wandb is a project dependency, but don't crash a paper run if a
            # hand-built env omitted it. Local summary.json / metrics.jsonl
            # still work.
            log(
                WARNING,
                "wandb-mode=%s but wandb is not installed; continuing with "
                "local metrics only. Reinstall the baseline with "
                "`python -m pip install -e .`.",
                mode,
            )
            return

        run_name = str(cfg.get("wandb-run-name", "")).strip() or None
        entity = str(cfg.get("wandb-entity", "")).strip() or None
        project = str(cfg.get("wandb-project", "ssfl-flower"))
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name,
            "mode": mode if mode in {"online", "offline"} else "offline",
            "config": dict(cfg),
        }
        if entity:
            init_kwargs["entity"] = entity
        self._run = wandb.init(**init_kwargs)
        self.enabled = True
        log(
            INFO,
            "W&B logging enabled (mode=%s, entity=%s, project=%s)",
            mode,
            entity or "(default)",
            project,
        )

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        """Log metrics when a W&B run is active.

        Pass ``commit=False`` to keep the current step open so later calls can
        add metrics at the same step. W&B otherwise treats a repeated step as
        out-of-order and drops the extra records.
        """
        if not self.enabled or self._run is None:
            return
        import wandb  # pylint: disable=import-outside-toplevel

        kwargs: dict[str, Any] = {}
        if step is not None:
            kwargs["step"] = step
        if commit is not None:
            kwargs["commit"] = commit
        wandb.log(metrics, **kwargs)

    def finish(self) -> None:
        """Finish the active W&B run, if any."""
        if not self.enabled or self._run is None:
            return
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.finish()
        self.enabled = False
        self._run = None
