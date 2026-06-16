from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

METADATA_FIELDS = [
    "idx",
    "project",
    "commit_id",
    "project_url",
    "commit_url",
    "commit_message",
    "func_hash",
    "file_name",
    "file_hash",
    "cwe",
    "cve",
    "cve_desc",
    "nvd_url",
]


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc

    return rows


def convert_rows(rows: list[dict[str, Any]], split_source: str) -> list[dict[str, Any]]:
    converted = []

    for row in rows:
        if "func" not in row:
            raise ValueError(f"Missing 'func'. Keys: {sorted(row.keys())}")

        if "target" not in row:
            raise ValueError(f"Missing 'target'. Keys: {sorted(row.keys())}")

        label = int(row["target"])
        if label not in (0, 1):
            raise ValueError(f"Expected target 0/1, got {row['target']!r}")

        item = {
            "text": str(row["func"] or ""),
            "label": label,
            "raw_func": str(row["func"] or ""),
            "split_source": split_source,
        }

        for field in METADATA_FIELDS:
            value = row.get(field)

            if value is None:
                value = [] if field == "cwe" else ""
            elif field == "cwe" and not isinstance(value, list):
                value = [str(value)]
            elif field == "cwe":
                value = [str(x) for x in value]
            else:
                value = str(value)

            item[field] = value

        converted.append(item)

    return converted


def load_split(path: str | None, split_source: str) -> Dataset | None:
    if path is None:
        return None

    rows = read_jsonl(path)
    converted = convert_rows(rows, split_source=split_source)

    if not converted:
        raise ValueError(f"No rows found in {path}")

    return Dataset.from_list(converted)


def add_split(
    dataset_dict: dict[str, Dataset],
    split_name: str,
    path: str | None,
    split_source: str,
) -> None:
    dataset = load_split(path, split_source)
    if dataset is not None:
        dataset_dict[split_name] = dataset


def print_stats(ds: DatasetDict, config_name: str) -> None:
    for split_name, split in ds.items():
        labels = [int(x) for x in split["label"]]
        total = len(labels)
        pos = sum(x == 1 for x in labels)
        neg = total - pos
        pos_rate = pos / total if total else 0.0

        print(
            f"{config_name}/{split_name}: n={total:,} "
            f"safe={neg:,} unsafe={pos:,} "
            f"unsafe_rate={pos_rate:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--full-train")
    parser.add_argument("--full-validation")
    parser.add_argument("--full-test")

    parser.add_argument("--paired-train")
    parser.add_argument("--paired-validation")
    parser.add_argument("--paired-test")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")

    args = parser.parse_args()

    default_splits: dict[str, Dataset] = {}
    paired_splits: dict[str, Dataset] = {}

    add_split(default_splits, "train", args.full_train, "full")
    add_split(default_splits, "validation", args.full_validation, "full")
    add_split(default_splits, "test", args.full_test, "full")

    add_split(paired_splits, "train", args.paired_train, "paired")
    add_split(paired_splits, "validation", args.paired_validation, "paired")
    add_split(paired_splits, "test", args.paired_test, "paired")

    configs = {
        "default": DatasetDict(default_splits) if default_splits else None,
        "paired": DatasetDict(paired_splits) if paired_splits else None,
    }
    configs = {name: ds for name, ds in configs.items() if ds is not None}

    if not configs:
        raise ValueError("No input split files were provided.")

    for config_name, ds in configs.items():
        print_stats(ds, config_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_name, ds in configs.items():
        ds.save_to_disk(str(output_dir / config_name))
    print(f"Saved dataset to {output_dir}")

    if args.repo_id:
        for config_name, ds in configs.items():
            ds.push_to_hub(args.repo_id, config_name=config_name, private=args.private)
        print(f"Uploaded to Hugging Face dataset repo: {args.repo_id}")


if __name__ == "__main__":
    main()
