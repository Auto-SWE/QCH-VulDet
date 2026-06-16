from __future__ import annotations

from datasets import Dataset, load_dataset


def validate_dataset(dataset: Dataset, split_name: str) -> None:
    missing_columns = {"text", "label"} - set(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Split {split_name} missing columns {missing_columns}. "
            f"Columns: {dataset.column_names}"
        )


def load_training_splits(
    dataset_name: str,
    dataset_config: str | None,
    train_split: str,
    eval_split: str,
):
    ds = load_dataset(dataset_name, dataset_config)
    print(ds)

    required_splits = [train_split, eval_split]
    missing_splits = [split for split in required_splits if split not in ds]

    if missing_splits:
        raise ValueError(
            f"Missing split(s): {missing_splits}. "
            f"Available splits: {list(ds.keys())}"
        )

    train = ds[train_split]
    validation = ds[eval_split]

    validate_dataset(train, train_split)
    validate_dataset(validation, eval_split)

    return train, validation


def load_processed_split(
    dataset_name: str,
    dataset_config: str | None,
    split_name: str,
) -> Dataset:
    dataset = load_dataset(dataset_name, dataset_config, split=split_name)
    validate_dataset(dataset, split_name)
    return dataset


def maybe_subset(dataset: Dataset, subset_size: int | None, seed: int = 42) -> Dataset:
    if subset_size is None:
        return dataset

    subset_size = min(subset_size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(subset_size))


def tokenize_dataset(tokenizer, dataset: Dataset, max_length: int) -> Dataset:
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = [int(label) for label in batch["label"]]
        return tokenized

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def tokenize_splits(
    tokenizer,
    train,
    validation,
    max_length: int,
    train_subset: int | None = None,
    eval_subset: int | None = None,
):
    train = maybe_subset(train, train_subset)
    validation = maybe_subset(validation, eval_subset)

    train_tok = tokenize_dataset(tokenizer, train, max_length)
    val_tok = tokenize_dataset(tokenizer, validation, max_length)
    return train_tok, val_tok
