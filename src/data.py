from __future__ import annotations

from datasets import Dataset, load_dataset


def load_processed_dataset(
    dataset_name: str,
    train_split: str,
    eval_split: str,
    test_split: str | None,
):
    ds = load_dataset(dataset_name)
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
    test = ds[test_split] if test_split and test_split in ds else None

    for split_name, split in [(train_split, train), (eval_split, validation)]:
        missing_columns = {"text", "label"} - set(split.column_names)
        if missing_columns:
            raise ValueError(
                f"Split {split_name} missing columns {missing_columns}. "
                f"Columns: {split.column_names}"
            )

    return train, validation, test


def maybe_subset(dataset: Dataset, subset_size: int | None, seed: int = 42) -> Dataset:
    if subset_size is None:
        return dataset

    subset_size = min(subset_size, len(dataset))
    return dataset.shuffle(seed=seed).select(range(subset_size))


def tokenize_splits(
    tokenizer,
    train,
    validation,
    test,
    max_length: int,
    train_subset: int | None = None,
    eval_subset: int | None = None,
):
    train = maybe_subset(train, train_subset)
    validation = maybe_subset(validation, eval_subset)

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        tokenized["labels"] = [int(x) for x in batch["label"]]

        return tokenized

    train_tok = train.map(
        tokenize_batch,
        batched=True,
        remove_columns=train.column_names,
        desc="Tokenizing train",
    )

    val_tok = validation.map(
        tokenize_batch,
        batched=True,
        remove_columns=validation.column_names,
        desc="Tokenizing validation",
    )

    test_tok = None
    if test is not None:
        test_tok = test.map(
            tokenize_batch,
            batched=True,
            remove_columns=test.column_names,
            desc="Tokenizing test",
        )

    return train_tok, val_tok, test_tok
