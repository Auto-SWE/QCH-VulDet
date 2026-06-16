from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def get_wandb_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = dict(cfg.get("wandb") or {})
    settings.setdefault("enabled", bool(os.environ.get("WANDB_API_KEY")))
    settings.setdefault("project", "qch-vuldet")
    settings.setdefault("entity", "automating-swe")
    settings.setdefault("tags", [])
    return settings


def is_wandb_enabled(cfg: dict[str, Any]) -> bool:
    settings = get_wandb_settings(cfg)
    return bool(settings.get("enabled"))


def init_wandb_run(
    cfg: dict[str, Any],
    config_path: str,
    job_type: str,
    extra_config: dict[str, Any] | None = None,
):
    settings = get_wandb_settings(cfg)
    if not settings.get("enabled"):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging is enabled, but the 'wandb' package is not installed."
        ) from exc

    os.environ.setdefault("WANDB_PROJECT", str(settings["project"]))
    os.environ.setdefault("WANDB_ENTITY", str(settings["entity"]))
    os.environ.setdefault("WANDB_WATCH", "false")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")

    mode = settings.get("mode")
    if mode:
        os.environ.setdefault("WANDB_MODE", str(mode))

    run_name = settings.get("name") or Path(config_path).stem
    run_config = {
        **cfg,
        "config_path": config_path,
        "job_type": job_type,
    }
    if extra_config:
        run_config.update(extra_config)

    if wandb.run is not None:
        wandb.config.update(run_config, allow_val_change=True)
        return wandb.run

    return wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        name=run_name,
        group=settings.get("group"),
        job_type=job_type,
        tags=settings.get("tags") or None,
        notes=settings.get("notes"),
        config=run_config,
    )


def finish_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()


def log_metrics(
    cfg: dict[str, Any],
    prefix: str,
    metrics: dict[str, Any],
) -> None:
    settings = get_wandb_settings(cfg)
    if not settings.get("enabled"):
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    wandb.log({f"{prefix}/{key}": value for key, value in metrics.items()})
