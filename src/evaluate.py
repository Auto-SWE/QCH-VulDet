from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm

from data import load_processed_dataset, maybe_subset, tokenize_splits
from experiment_config import load_config
from metrics import compute_binary_metrics, find_best_threshold, softmax_np
from model_utils import build_eval_model_and_tokenizer
from wandb_utils import finish_wandb_run, init_wandb_run, log_eval_outputs


def get_model_input_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_logits(model, tokenizer, dataset, batch_size: int):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    input_device = get_model_input_device(model)
    print(
        f"Running inference on {len(dataset)} examples "
        f"with batch_size={batch_size} on {input_device}."
    )

    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            labels = batch.pop("labels")
            batch = {
                key: value.to(input_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            outputs = model(**batch)
            all_logits.append(outputs.logits.detach().float().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return np.column_stack([np.log1p(-probs), np.log(probs)])


def resolve_eval_subset(args, cfg: dict):
    if args.all_examples:
        return None

    if args.eval_subset is not None:
        return args.eval_subset

    return cfg["eval_subset"]


def get_probability_csv_fields(raw_dataset):
    metadata_columns = [
        "idx",
        "project",
        "commit_id",
        "file_name",
        "func_hash",
        "cwe",
        "cve",
    ]
    present_metadata_columns = [
        column for column in metadata_columns if column in raw_dataset.column_names
    ]

    return [
        "row_id",
        *present_metadata_columns,
        "label",
        "pred_label",
        "prob_safe",
        "prob_vulnerable",
    ]


def build_probability_row(raw_row, row_id: int, label: int, prob: float, threshold: float):
    out = {"row_id": row_id}
    for column in [
        "idx",
        "project",
        "commit_id",
        "file_name",
        "func_hash",
        "cwe",
        "cve",
    ]:
        if column not in raw_row:
            continue

        value = raw_row.get(column)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        out[column] = "" if value is None else value

    out.update(
        {
            "label": int(label),
            "pred_label": int(prob >= threshold),
            "prob_safe": float(1.0 - prob),
            "prob_vulnerable": float(prob),
        }
    )
    return out


def ensure_output_path_writable(output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline=""):
        pass

    return path


def read_probability_csv(output_path: str):
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    labels = []
    probs = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"row_id", "label", "prob_vulnerable"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Existing probability CSV missing columns: {missing}")

        expected_row_id = 0
        for row in reader:
            row_id = int(row["row_id"])
            if row_id != expected_row_id:
                raise ValueError(
                    "Existing probability CSV is not a contiguous prefix: "
                    f"expected row_id {expected_row_id}, found {row_id}."
                )
            labels.append(int(row["label"]))
            probs.append(float(row["prob_vulnerable"]))
            expected_row_id += 1

    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def append_probability_csv(
    raw_dataset,
    labels,
    probs,
    output_path: str,
    threshold: float,
    start_row_id: int = 0,
) -> None:
    path = ensure_output_path_writable(output_path)
    fieldnames = get_probability_csv_fields(raw_dataset)
    write_header = path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for offset, raw_row in enumerate(raw_dataset):
            writer.writerow(
                build_probability_row(
                    raw_row,
                    row_id=start_row_id + offset,
                    label=int(labels[offset]),
                    prob=float(probs[offset]),
                    threshold=threshold,
                )
            )

    print(f"Wrote probabilities: {path}")


def predict_logits_and_stream_probs(
    model,
    tokenizer,
    dataset,
    raw_dataset,
    batch_size: int,
    output_path: str,
    threshold: float,
    start_row_id: int,
):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    input_device = get_model_input_device(model)
    output_path = ensure_output_path_writable(output_path)
    fieldnames = get_probability_csv_fields(raw_dataset)
    write_header = output_path.stat().st_size == 0

    print(
        f"Running inference on {len(dataset)} examples "
        f"with batch_size={batch_size} on {input_device}."
    )
    print(f"Streaming probabilities to {output_path}.")

    all_logits = []
    all_labels = []
    row_id = start_row_id
    raw_offset = 0

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
                labels = batch.pop("labels")
                batch_size_actual = int(labels.shape[0])
                batch_inputs = {
                    key: value.to(input_device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }

                outputs = model(**batch_inputs)
                batch_logits = outputs.logits.detach().float().cpu().numpy()
                batch_labels = labels.detach().cpu().numpy()
                batch_probs = softmax_np(batch_logits)[:, 1]

                all_logits.append(batch_logits)
                all_labels.append(batch_labels)

                for offset in range(batch_size_actual):
                    writer.writerow(
                        build_probability_row(
                            raw_dataset[raw_offset + offset],
                            row_id=row_id + offset,
                            label=int(batch_labels[offset]),
                            prob=float(batch_probs[offset]),
                            threshold=threshold,
                        )
                    )

                f.flush()
                row_id += batch_size_actual
                raw_offset += batch_size_actual

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eval-subset", type=int, default=None)
    parser.add_argument("--all-examples", action="store_true")
    parser.add_argument("--save-probs", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    init_wandb_run(
        cfg,
        args.config,
        job_type="evaluate",
        extra_config={
            "adapter_dir": args.adapter_dir,
            "eval_split": args.split,
            "eval_subset_override": args.eval_subset,
            "all_examples": args.all_examples,
            "requested_threshold": args.threshold,
            "save_probs": args.save_probs,
        },
    )

    train_raw, val_raw, test_raw = load_processed_dataset(
        dataset_name=cfg["dataset_name"],
        dataset_config=cfg["dataset_config"],
        train_split=cfg["train_split"],
        eval_split=cfg["eval_split"],
        test_split=cfg["test_split"],
    )

    if args.split == "test" and test_raw is None:
        raise ValueError("Requested test split, but dataset/config has no test split.")

    eval_raw = test_raw if args.split == "test" else val_raw
    eval_raw = maybe_subset(eval_raw, resolve_eval_subset(args, cfg))

    stream_probs = bool(args.save_probs and args.threshold is not None)
    existing_labels = np.array([], dtype=np.int64)
    existing_probs = np.array([], dtype=np.float64)
    resume_offset = 0

    if args.save_probs:
        ensure_output_path_writable(args.save_probs)
        if stream_probs:
            existing_labels, existing_probs = read_probability_csv(args.save_probs)
            resume_offset = len(existing_labels)
            if resume_offset > len(eval_raw):
                raise ValueError(
                    f"Existing probability CSV has {resume_offset} rows, but "
                    f"the requested split has only {len(eval_raw)} examples."
                )
            if resume_offset:
                print(
                    f"Resuming probability export from row_id {resume_offset}; "
                    f"{len(eval_raw) - resume_offset} examples remain."
                )

    eval_raw_to_predict = eval_raw
    if resume_offset:
        if resume_offset < len(eval_raw):
            eval_raw_to_predict = eval_raw.select(range(resume_offset, len(eval_raw)))
        else:
            eval_raw_to_predict = None

    if eval_raw_to_predict is not None and len(eval_raw_to_predict):
        model, tokenizer = build_eval_model_and_tokenizer(cfg, args.adapter_dir)

        _, eval_ds, _ = tokenize_splits(
            tokenizer=tokenizer,
            train=train_raw.select(range(1)),
            validation=eval_raw_to_predict,
            test=None,
            max_length=int(cfg["max_length"]),
            train_subset=1,
            eval_subset=None,
        )

        batch_size = int(cfg["per_device_eval_batch_size"])
        if stream_probs:
            logits, labels = predict_logits_and_stream_probs(
                model,
                tokenizer,
                eval_ds,
                eval_raw_to_predict,
                batch_size=batch_size,
                output_path=args.save_probs,
                threshold=float(args.threshold),
                start_row_id=resume_offset,
            )
        else:
            logits, labels = predict_logits(
                model,
                tokenizer,
                eval_ds,
                batch_size=batch_size,
            )
        probs = softmax_np(logits)[:, 1]
    else:
        print("Probability CSV already contains all requested examples.")
        logits = np.empty((0, 2), dtype=np.float64)
        labels = np.array([], dtype=np.int64)
        probs = np.array([], dtype=np.float64)

    if resume_offset:
        logits = np.concatenate([logits_from_probs(existing_probs), logits], axis=0)
        labels = np.concatenate([existing_labels, labels], axis=0)
        probs = np.concatenate([existing_probs, probs], axis=0)

    if args.threshold is None:
        if args.split == "test":
            print(
                "WARNING: selecting the best threshold on the test split leaks test "
                "labels. Use validation to choose a threshold, then pass it with "
                "--threshold for test metrics."
            )

        best = find_best_threshold(labels, probs)
        threshold = best["threshold"]

        print("Best threshold on this split:")
        print(json.dumps(best, indent=2))
    else:
        threshold = args.threshold

    y_pred = (probs >= threshold).astype(int)

    if args.save_probs and not stream_probs:
        append_probability_csv(eval_raw, labels, probs, args.save_probs, threshold)

    print(f"Using threshold: {threshold:.4f}")

    print("Confusion matrix:")
    print(confusion_matrix(labels, y_pred))

    print("Classification report:")
    print(
        classification_report(
            labels,
            y_pred,
            labels=[0, 1],
            target_names=["safe", "vulnerable"],
            digits=4,
            zero_division=0,
        )
    )

    report = classification_report(
        labels,
        y_pred,
        labels=[0, 1],
        target_names=["safe", "vulnerable"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    threshold_metrics = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, y_pred),
        "balanced_accuracy": balanced_accuracy_score(labels, y_pred),
        "precision_vulnerable": precision_score(labels, y_pred, zero_division=0),
        "recall_vulnerable": recall_score(labels, y_pred, zero_division=0),
        "f1_vulnerable": f1_score(labels, y_pred, zero_division=0),
        "precision_safe": precision_score(
            labels, y_pred, pos_label=0, zero_division=0
        ),
        "recall_safe": recall_score(labels, y_pred, pos_label=0, zero_division=0),
        "f1_safe": f1_score(labels, y_pred, pos_label=0, zero_division=0),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "mcc": matthews_corrcoef(labels, y_pred),
        "support_safe": report["safe"]["support"],
        "support_vulnerable": report["vulnerable"]["support"],
    }

    if args.threshold is None:
        threshold_metrics.update({f"best_{key}": value for key, value in best.items()})

    metrics_at_0_5 = {
        f"{key}_at_default_threshold": value
        for key, value in compute_binary_metrics((logits, labels)).items()
    }
    log_eval_outputs(
        cfg,
        prefix=args.split,
        metrics={**threshold_metrics, **metrics_at_0_5},
        labels=labels,
        preds=y_pred,
        probs=probs,
    )
    finish_wandb_run()


if __name__ == "__main__":
    main()
