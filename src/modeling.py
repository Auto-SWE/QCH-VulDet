from __future__ import annotations

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    return tokenizer


def set_model_pad_token_id(model, pad_token_id: int) -> None:
    seen_config_ids = set()
    for candidate in [model, *model.modules()]:
        for attr_name in ("config", "generation_config"):
            config = getattr(candidate, attr_name, None)
            if config is None or id(config) in seen_config_ids:
                continue

            if hasattr(config, "pad_token_id"):
                config.pad_token_id = int(pad_token_id)
            seen_config_ids.add(id(config))


def get_quantization_config(cfg: dict):
    if not cfg.get("use_qlora", True):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
