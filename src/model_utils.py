from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_pad_token_id(tokenizer) -> int:
    """Return a valid padding token ID, falling back to EOS when needed."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    return int(pad_token_id)


def set_model_pad_token_id(model, pad_token_id: int) -> None:
    """Set pad_token_id on every model config object that exposes it."""
    seen_config_ids = set()

    # Walk the top-level model and all child modules because PEFT/wrapped models
    # may keep separate config or generation_config objects on nested modules.
    for candidate in [model, *model.modules()]:
        for attr_name in ("config", "generation_config"):
            config = getattr(candidate, attr_name, None)
            if config is None or id(config) in seen_config_ids:
                continue

            if hasattr(config, "pad_token_id"):
                config.pad_token_id = pad_token_id
            seen_config_ids.add(id(config))


def load_tokenizer(source: str):
    """Load the tokenizer and ensure it has a usable padding token."""
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        use_fast=True,
    )

    # many causal LMs do not define a pad token. For classification batches,
    # reusing EOS as PAD is a common fallback.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer pad_token_id: {get_pad_token_id(tokenizer)}")
    return tokenizer


def get_quantization_config(cfg: dict):
    """Create the 4-bit BitsAndBytes quantization config used for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_base_classifier(cfg: dict, pad_token_id: int):
    """Load the base sequence-classification model in 4-bit precision."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=2,
        quantization_config=get_quantization_config(cfg),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Keep tokenizer/model padding behavior aligned before training or eval.
    set_model_pad_token_id(model, pad_token_id)
    return model


def prepare_for_qlora_training(base_model):
    """Prepare a quantized model for k-bit adapter training."""
    return prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )


def cast_trainable_params_to_fp32(model) -> None:
    """Cast trainable fp16 parameters to fp32 for more stable optimization."""
    for param in model.parameters():
        if param.requires_grad and param.dtype == torch.float16:
            param.data = param.data.to(torch.float32)


def add_qlora_adapter(base_model, cfg: dict):
    """Attach a LoRA sequence-classification adapter to the base model."""
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
        # Keep the classifier head trainable and save it with the adapter.
        modules_to_save=["score"],
    )

    return get_peft_model(base_model, peft_config)


def build_model_and_tokenizer(cfg: dict):
    """Build the tokenizer and training model, optionally resuming an adapter."""
    tokenizer = load_tokenizer(cfg["model_name"])
    pad_token_id = get_pad_token_id(tokenizer)

    # Load the quantized classifier, then prepare it for QLoRA training.
    base_model = build_base_classifier(cfg, pad_token_id)
    base_model = prepare_for_qlora_training(base_model)

    resume_adapter_dir = cfg.get("resume_adapter_dir")
    if resume_adapter_dir:
        print(f"Loading adapter from {resume_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            resume_adapter_dir,
            is_trainable=True,
        )
    else:
        model = add_qlora_adapter(base_model, cfg)

    # Re-apply padding after PEFT wrapping so all nested configs stay aligned.
    set_model_pad_token_id(model, pad_token_id)

    # Only trainable parameters are cast; frozen quantized weights remain k-bit.
    cast_trainable_params_to_fp32(model)

    model.print_trainable_parameters()
    return model, tokenizer


def build_eval_model_and_tokenizer(cfg: dict, adapter_dir: str | None):
    """Build the tokenizer and evaluation model, with or without an adapter."""
    tokenizer = load_tokenizer(adapter_dir or cfg["model_name"])
    pad_token_id = get_pad_token_id(tokenizer)

    base_model = build_base_classifier(cfg, pad_token_id)

    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        print(
            "Evaluating base model without an adapter. "
            "For causal/base checkpoints, the sequence-classification head is "
            "randomly initialized unless the checkpoint includes one."
        )
        model = base_model

    # Ensure eval uses the same padding ID as the tokenizer.
    set_model_pad_token_id(model, pad_token_id)
    model.eval()
    return model, tokenizer