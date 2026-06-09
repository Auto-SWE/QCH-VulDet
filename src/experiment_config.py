from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sync_resolved_wandb_config(cfg: dict[str, Any]) -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    wandb.config.update(
        {
            "resolved_config": dict(cfg),
            "resolved_output_dir": cfg["output_dir"],
        },
        allow_val_change=True,
    )
