from __future__ import annotations

import numpy as np
import torch
from transformers import DataCollatorWithPadding, TrainingArguments

from loss import WeightedClassificationTrainer
from metrics import compute_binary_metrics, softmax_np
from model_utils import get_pad_token_id
from wandb_utils import (
    get_wandb_settings,
    is_wandb_enabled,
    log_eval_outputs,
)


def get_class_weights(train_dataset):
    """Compute inverse-frequency class weights for binary classification."""
    labels = np.asarray(train_dataset["labels"])

    num_neg = int((labels == 0).sum())
    num_pos = int((labels == 1).sum())
    total = num_neg + num_pos

    # both classes must be present for balanced weighting to be meaningful.
    if num_neg == 0 or num_pos == 0:
        raise ValueError(f"Bad label counts: neg={num_neg}, pos={num_pos}")

    # weight each class inversely proportional to its frequency.
    weights = torch.tensor(
        [
            total / (2.0 * num_neg),
            total / (2.0 * num_pos),
        ],
        dtype=torch.float32,
    )

    print(f"Label counts: neg={num_neg}, pos={num_pos}")
    print(f"Class weights: {weights.tolist()}")

    return weights


def build_training_args(cfg: dict) -> TrainingArguments:
    """Create Hugging Face training arguments from the project config."""
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        optim=cfg["optim"],
        max_grad_norm=float(cfg["max_grad_norm"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        fp16=True,
        logging_steps=int(cfg["logging_steps"]),

        # Evaluate and checkpoint on the same step schedule so the best model
        # can be restored at the end of training.
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,

        # Only report to W&B when enabled in config.
        report_to="wandb" if is_wandb_enabled(cfg) else "none",

        # Keep dataset columns available for custom trainer/collator behavior.
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(cfg["seed"]),
        eval_accumulation_steps=cfg["eval_accumulation_steps"],
    )


def build_trainer(cfg: dict, model, tokenizer, train_ds, val_ds):
    """Build the weighted binary-classification trainer."""
    class_weights = None

    # Optional automatic class weighting helps with imbalanced datasets.
    if cfg["class_weighting"] == "auto":
        class_weights = get_class_weights(train_ds)

    return WeightedClassificationTrainer(
        model=model,
        args=build_training_args(cfg),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_binary_metrics,
        class_weights=class_weights,
        pad_token_id=get_pad_token_id(tokenizer),
    )


def save_model(cfg: dict, trainer, tokenizer) -> None:
    """Save the trained model artifacts and tokenizer to the output directory."""
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


def log_final_validation(cfg: dict, trainer, val_ds) -> None:
    """Run final validation and optionally log prediction plots to W&B."""
    print("Final validation evaluation:")
    final_metrics = trainer.evaluate(metric_key_prefix="final_validation")
    print(final_metrics)

    # Skip prediction-level logging unless configured.
    if not get_wandb_settings(cfg)["log_eval_plots"]:
        return

    # Generate final validation predictions for plots and detailed diagnostics.
    final_predictions = trainer.predict(
        val_ds,
        metric_key_prefix="final_validation_predict",
    )

    logits = np.asarray(final_predictions.predictions)
    labels = np.asarray(final_predictions.label_ids)

    # Convert class logits into positive-class probabilities and hard labels.
    probs = softmax_np(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = compute_binary_metrics((logits, labels))

    log_eval_outputs(
        cfg,
        prefix="final_validation",
        metrics=metrics,
        labels=labels,
        preds=preds,
        probs=probs,
    )