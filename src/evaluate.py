from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from peft import PeftModel
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data import load_processed_split, maybe_subset, tokenize_dataset
from metrics import compute_metrics_at_threshold, find_best_threshold
from modeling import get_quantization_config, load_tokenizer, set_model_pad_token_id
from wandb_utils import finish_wandb_run, init_wandb_run, log_metrics


def predict_logits(model, tokenizer, dataset, cfg: dict):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="eval_outputs/.trainer",
            per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
            eval_accumulation_steps=cfg.get("eval_accumulation_steps"),
            report_to="none",
            remove_unused_columns=False,
            label_names=["labels"],
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(dataset)
    return np.asarray(predictions.predictions), np.asarray(predictions.label_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.split == "test" and args.threshold is None:
        raise ValueError(
            "Test evaluation requires --threshold. Select it on validation first."
        )

    init_wandb_run(
        cfg,
        args.config,
        job_type="evaluate",
        extra_config={
            "adapter_dir": args.adapter_dir,
            "eval_split": args.split,
            "eval_limit": args.limit,
            "requested_threshold": args.threshold,
        },
    )

    tokenizer_source = args.adapter_dir or cfg["model_name"]
    tokenizer = load_tokenizer(tokenizer_source)
    pad_token_id = int(tokenizer.pad_token_id)

    split_name = cfg.get("test_split") if args.split == "test" else cfg["eval_split"]
    if not split_name:
        raise ValueError(f"No dataset split configured for {args.split} evaluation.")

    eval_raw = load_processed_split(
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg.get("dataset_config"),
        split_name=split_name,
    )
    eval_raw = maybe_subset(
        eval_raw,
        args.limit,
        seed=int(cfg.get("seed", 42)),
    )
    eval_ds = tokenize_dataset(tokenizer, eval_raw, int(cfg["max_length"]))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        quantization_config=get_quantization_config(cfg),
        dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    set_model_pad_token_id(base_model, pad_token_id)

    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        model = base_model
        print(
            "Evaluating base model without an adapter. "
            "For causal/base checkpoints, the sequence-classification head is "
            "randomly initialized unless the checkpoint includes one."
        )
    set_model_pad_token_id(model, pad_token_id)
    model.eval()

    logits, labels = predict_logits(
        model,
        tokenizer,
        eval_ds,
        cfg,
    )
    probs = softmax(logits, axis=1)[:, 1]

    if args.threshold is None:
        best = find_best_threshold(labels, probs)
        threshold = best["threshold"]

        print("Best threshold on this split:")
        print(json.dumps(best, indent=2))
    else:
        threshold = args.threshold

    y_pred = (probs >= threshold).astype(int)

    print(f"Using threshold: {threshold:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels, y_pred))

    metrics = compute_metrics_at_threshold(labels, probs, threshold)

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))

    log_metrics(
        cfg,
        prefix=args.split,
        metrics=metrics,
    )
    finish_wandb_run()


if __name__ == "__main__":
    main()
