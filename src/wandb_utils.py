from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def get_wandb_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg["wandb"])


def is_wandb_enabled(cfg: dict[str, Any]) -> bool:
    settings = get_wandb_settings(cfg)
    return bool(settings["enabled"])


def init_wandb_run(
    cfg: dict[str, Any],
    config_path: str,
    job_type: str,
    extra_config: dict[str, Any] | None = None,
):
    settings = get_wandb_settings(cfg)
    if not settings["enabled"]:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B logging is enabled, but the 'wandb' package is not installed."
        ) from exc

    run_name = Path(config_path).stem
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
        job_type=job_type,
        tags=settings["tags"] or None,
        config=run_config,
    )


def finish_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        wandb.finish()


def log_eval_outputs(
    cfg: dict[str, Any],
    prefix: str,
    metrics: dict[str, Any],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> None:
    settings = get_wandb_settings(cfg)
    if not settings["enabled"]:
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}

    if settings["log_eval_plots"]:
        labels = np.asarray(labels)
        preds = np.asarray(preds)
        probs = np.asarray(probs)
        proba_2class = np.column_stack([1.0 - probs, probs])

        payload[f"{prefix}/confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true=labels,
            preds=preds,
            class_names=["safe", "vulnerable"],
        )

        try:
            payload[f"{prefix}/pr_curve"] = wandb.plot.pr_curve(
                labels,
                proba_2class,
                labels=["safe", "vulnerable"],
                classes_to_plot=[1],
            )
        except Exception:
            pass

        try:
            payload[f"{prefix}/roc_curve"] = wandb.plot.roc_curve(
                labels,
                proba_2class,
                labels=["safe", "vulnerable"],
                classes_to_plot=[1],
            )
        except Exception:
            pass

    wandb.log(payload)
