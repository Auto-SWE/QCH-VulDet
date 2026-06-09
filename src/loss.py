from __future__ import annotations

import torch
from transformers import Trainer


class WeightedClassificationTrainer(Trainer):
    """Trainer that uses class-weighted cross-entropy loss."""

    # Initialize the trainer with optional class weights and a padding token ID.
    def __init__(self, class_weights=None, pad_token_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.pad_token_id = pad_token_id

    # Ensure every available model config uses the configured padding token ID.
    def _ensure_pad_token_id(self, model) -> None:
        """Apply pad_token_id to every model config that exposes it."""
        if self.pad_token_id is None:
            return

        seen_config_ids = set()

        for module in [model, *model.modules()]:
            for config_name in ("config", "generation_config"):
                config = getattr(module, config_name, None)

                if config is None:
                    continue

                if id(config) in seen_config_ids:
                    continue

                if hasattr(config, "pad_token_id"):
                    config.pad_token_id = int(self.pad_token_id)

                seen_config_ids.add(id(config))

    # Compute class-weighted cross-entropy loss for the current training batch.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss for a batch."""
        self._ensure_pad_token_id(model)

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs.logits

        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        loss = loss_fn(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss

    # Sync the padding token ID before delegating prediction to the base Trainer.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Run prediction after syncing pad_token_id."""
        self._ensure_pad_token_id(model)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=ignore_keys,
        )