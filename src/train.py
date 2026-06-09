from __future__ import annotations

import argparse
from pathlib import Path

from data import load_processed_dataset, tokenize_splits
from experiment_config import load_config, sync_resolved_wandb_config
from model_utils import build_model_and_tokenizer
from training import build_trainer, log_final_validation, save_model
from wandb_utils import finish_wandb_run, init_wandb_run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_training_data(cfg: dict, tokenizer):
    train_raw, val_raw, test_raw = load_processed_dataset(
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg["dataset_config"],
        train_split=cfg["train_split"],
        eval_split=cfg["eval_split"],
        test_split=cfg["test_split"],
    )

    train_ds, val_ds, _ = tokenize_splits(
        tokenizer=tokenizer,
        train=train_raw,
        validation=val_raw,
        test=test_raw,
        max_length=int(cfg["max_length"]),
        train_subset=cfg["train_subset"],
        eval_subset=cfg["eval_subset"],
    )

    return train_ds, val_ds


def main():
    args = parse_args()
    cfg = load_config(args.config)

    init_wandb_run(cfg, args.config, job_type="train")
    sync_resolved_wandb_config(cfg)

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds, val_ds = load_training_data(cfg, tokenizer)
    trainer = build_trainer(cfg, model, tokenizer, train_ds, val_ds)

    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))

    save_model(cfg, trainer, tokenizer)
    log_final_validation(cfg, trainer, val_ds)
    finish_wandb_run()


if __name__ == "__main__":
    main()
