from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from data import load_training_splits, tokenize_splits
from metrics import compute_binary_metrics
from modeling import get_quantization_config, load_tokenizer, set_model_pad_token_id
from wandb_utils import (
    finish_wandb_run,
    init_wandb_run,
    is_wandb_enabled,
)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_class_weights(train_dataset):
    labels = np.asarray(train_dataset["labels"])

    num_neg = int((labels == 0).sum())
    num_pos = int((labels == 1).sum())
    total = num_neg + num_pos

    if num_neg == 0 or num_pos == 0:
        raise ValueError(f"Bad label counts: neg={num_neg}, pos={num_pos}")

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


def build_loss_function(class_weights):
    if class_weights is None:
        return None

    def weighted_cross_entropy(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            weight=class_weights.to(logits.device),
        )

    return weighted_cross_entropy


def cast_trainable_params_to_fp32(model):
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)


def build_model_and_tokenizer(cfg):
    tokenizer = load_tokenizer(cfg["model_name"])
    pad_token_id = int(tokenizer.pad_token_id)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        quantization_config=get_quantization_config(cfg),
        dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    set_model_pad_token_id(base_model, pad_token_id)

    if cfg.get("use_qlora", True):
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    elif cfg.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable()

    resume_adapter_dir = cfg.get("resume_adapter_dir")

    if resume_adapter_dir:
        print(f"Loading adapter from {resume_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            resume_adapter_dir,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["score"],
        )

        model = get_peft_model(base_model, peft_config)

    set_model_pad_token_id(model, pad_token_id)

    if cfg.get("fp16") and not cfg.get("bf16"):
        cast_trainable_params_to_fp32(model)

    model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    init_wandb_run(cfg, args.config, job_type="train")

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    model, tokenizer = build_model_and_tokenizer(cfg)

    train_raw, val_raw = load_training_splits(
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg.get("dataset_config"),
        train_split=cfg["train_split"],
        eval_split=cfg["eval_split"],
    )

    train_ds, val_ds = tokenize_splits(
        tokenizer=tokenizer,
        train=train_raw,
        validation=val_raw,
        max_length=int(cfg["max_length"]),
        train_subset=cfg.get("train_subset"),
        eval_subset=cfg.get("eval_subset"),
    )

    class_weights = None
    if cfg.get("class_weighting") == "auto":
        class_weights = get_class_weights(train_ds)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        weight_decay=float(cfg["weight_decay"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        optim=cfg.get("optim", "adamw_torch_fused"),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        logging_steps=int(cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(cfg["eval_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        report_to="wandb" if is_wandb_enabled(cfg) else "none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=int(cfg.get("seed", 42)),
        eval_accumulation_steps=cfg.get("eval_accumulation_steps"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_binary_metrics,
        compute_loss_func=build_loss_function(class_weights),
    )

    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    finish_wandb_run()


if __name__ == "__main__":
    main()
