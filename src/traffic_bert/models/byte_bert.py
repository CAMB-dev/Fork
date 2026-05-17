"""Byte-BERT models for MLM pretraining and hierarchical classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM, BertModel

from traffic_bert.tokenizer import ByteTokenizer


def create_bert_config(**overrides: Any) -> BertConfig:
    tokenizer = ByteTokenizer()
    defaults = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
        "max_position_embeddings": 512,
        "type_vocab_size": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
    }
    defaults.update(overrides)
    return BertConfig(**defaults)


def create_mlm_model(**config_overrides: Any) -> BertForMaskedLM:
    return BertForMaskedLM(create_bert_config(**config_overrides))


class ByteBertForHierarchicalClassification(nn.Module):
    """BERT encoder with major softmax and minor multi-label heads."""

    def __init__(
        self,
        bert_config: BertConfig,
        num_major_labels: int,
        num_minor_labels: int,
        minor_to_major: list[int] | None = None,
        lambda_minor: float = 1.0,
        pooling: str = "mean",
        major_class_weights: list[float] | None = None,
        context_feature_size: int = 0,
        context_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.bert = BertModel(bert_config)
        self.context_feature_size = int(context_feature_size)
        if context_hidden_size is None:
            context_hidden_size = bert_config.hidden_size
        self.context_projection: nn.Module | None = None
        classifier_input_size = bert_config.hidden_size
        if self.context_feature_size > 0:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(self.context_feature_size),
                nn.Linear(self.context_feature_size, int(context_hidden_size)),
                nn.GELU(),
                nn.Dropout(float(bert_config.hidden_dropout_prob)),
            )
            classifier_input_size += int(context_hidden_size)
        self.major_classifier = nn.Linear(classifier_input_size, num_major_labels)
        self.minor_classifier = nn.Linear(classifier_input_size, num_minor_labels)
        self.lambda_minor = lambda_minor
        self.pooling = pooling
        if minor_to_major is None:
            minor_to_major = [0] * num_minor_labels
        self.register_buffer(
            "minor_to_major",
            torch.tensor(minor_to_major, dtype=torch.long),
            persistent=False,
        )
        if major_class_weights is None:
            major_class_weights = []
        self.register_buffer(
            "major_class_weights",
            torch.tensor(major_class_weights, dtype=torch.float32),
            persistent=False,
        )

    def _encode_windows(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        window_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]

        batch_size, num_windows, seq_len = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * num_windows, seq_len)
        flat_attention_mask = attention_mask.reshape(batch_size * num_windows, seq_len)
        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        cls = outputs.last_hidden_state[:, 0, :].reshape(batch_size, num_windows, -1)

        if window_mask is None:
            window_mask = torch.ones(
                batch_size,
                num_windows,
                dtype=torch.bool,
                device=input_ids.device,
            )

        if self.pooling == "mean":
            weights = window_mask.to(dtype=cls.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (cls * weights).sum(dim=1) / denom
        if self.pooling == "max":
            masked = cls.masked_fill(~window_mask.unsqueeze(-1), torch.finfo(cls.dtype).min)
            return masked.max(dim=1).values
        raise ValueError(f"unsupported pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        window_mask: torch.Tensor | None = None,
        context_features: torch.Tensor | None = None,
        major_labels: torch.Tensor | None = None,
        minor_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        flow_repr = self._encode_windows(input_ids, attention_mask, window_mask)
        classifier_repr = flow_repr
        if self.context_projection is not None:
            if context_features is None:
                context_features = torch.zeros(
                    flow_repr.shape[0],
                    self.context_feature_size,
                    dtype=flow_repr.dtype,
                    device=flow_repr.device,
                )
            else:
                context_features = context_features.to(dtype=flow_repr.dtype, device=flow_repr.device)
            context_repr = self.context_projection(context_features)
            classifier_repr = torch.cat([flow_repr, context_repr], dim=-1)
        major_logits = self.major_classifier(classifier_repr)
        minor_logits = self.minor_classifier(classifier_repr)

        result = {
            "flow_repr": flow_repr,
            "classifier_repr": classifier_repr,
            "major_logits": major_logits,
            "minor_logits": minor_logits,
        }

        losses: list[torch.Tensor] = []
        if major_labels is not None:
            major_weight = (
                self.major_class_weights.to(device=major_logits.device)
                if self.major_class_weights.numel() == major_logits.shape[-1]
                else None
            )
            major_loss = F.cross_entropy(major_logits, major_labels, weight=major_weight)
            result["major_loss"] = major_loss
            losses.append(major_loss)

        if minor_labels is not None and minor_logits.numel() > 0:
            raw_minor_loss = F.binary_cross_entropy_with_logits(
                minor_logits,
                minor_labels,
                reduction="none",
            )
            if major_labels is not None and self.minor_to_major.numel() > 0:
                mask = self.minor_to_major.unsqueeze(0) == major_labels.unsqueeze(1)
                raw_minor_loss = raw_minor_loss * mask.to(dtype=raw_minor_loss.dtype)
                denom = mask.sum().clamp_min(1).to(dtype=raw_minor_loss.dtype)
                minor_loss = raw_minor_loss.sum() / denom
            else:
                minor_loss = raw_minor_loss.mean()
            result["minor_loss"] = minor_loss
            losses.append(self.lambda_minor * minor_loss)

        if losses:
            result["loss"] = sum(losses)
        return result
