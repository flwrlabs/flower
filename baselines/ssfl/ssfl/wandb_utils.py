"""Optional Weights & Biases logging (opt-in)."""

from __future__ import annotations

from logging import INFO, WARNING
from typing import Any

from flwr.common import log


class WandbSession:
    """No-op unless wandb-mode is online/offline and wandb is installed."""

    def __init__(self) -> None:
        self._run: Any | None = None
        self.enabled = False

    def start(self, cfg: dict[str, Any]) -> None:
        """Start an optional W&B run from the application config."""
        mode = str(cfg.get("wandb-mode", "disabled")).lower()
        if mode in {"", "disabled", "off", "false", "none"}:
            return
        try:
            import wandb  # pylint: disable=import-outside-toplevel
        except ImportError:
            # Flower's per-run env only installs main deps; don't crash paper runs
            # if wandb is missing — local summary.json / metrics.jsonl still work.
            log(
                WARNING,
                "wandb-mode=%s but wandb is not installed in this environment; "
                "continuing with local metrics only. Add wandb to project "
                "dependencies or `uv pip install wandb` in the app env.",
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

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics when a W&B run is active."""
        if not self.enabled or self._run is None:
            return
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish the active W&B run, if any."""
        if not self.enabled or self._run is None:
            return
        import wandb  # pylint: disable=import-outside-toplevel

        wandb.finish()
        self.enabled = False
        self._run = None
