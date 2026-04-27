"""Minimal training and evaluation loops."""

from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_classifier_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            window_mask=batch["window_mask"].to(device),
            major_labels=batch["major_labels"].to(device),
            minor_labels=batch["minor_labels"].to(device),
        )
        loss = outputs["loss"]
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return {"loss": total_loss / max(steps, 1)}


@torch.no_grad()
def collect_classifier_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, list]:
    model.eval()
    major_logits: list[torch.Tensor] = []
    minor_logits: list[torch.Tensor] = []
    major_labels: list[torch.Tensor] = []
    minor_labels: list[torch.Tensor] = []
    flow_ids: list[str] = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            window_mask=batch["window_mask"].to(device),
        )
        major_logits.append(outputs["major_logits"].cpu())
        minor_logits.append(outputs["minor_logits"].cpu())
        major_labels.append(batch["major_labels"].cpu())
        minor_labels.append(batch["minor_labels"].cpu())
        flow_ids.extend(batch["flow_ids"])

    return {
        "major_logits": torch.cat(major_logits) if major_logits else torch.empty(0),
        "minor_logits": torch.cat(minor_logits) if minor_logits else torch.empty(0),
        "major_labels": torch.cat(major_labels) if major_labels else torch.empty(0),
        "minor_labels": torch.cat(minor_labels) if minor_labels else torch.empty(0),
        "flow_ids": flow_ids,
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    extra: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def mask_mlm_inputs(
    input_ids: torch.Tensor,
    special_token_ids: set[int],
    mask_token_id: int,
    vocab_size: int,
    probability: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply BERT-style MLM masking and return masked inputs plus labels."""

    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, probability, device=input_ids.device)
    special_mask = torch.zeros(labels.shape, dtype=torch.bool, device=input_ids.device)
    for token_id in special_token_ids:
        special_mask |= labels == token_id
    probability_matrix.masked_fill_(special_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    masked_input_ids = input_ids.clone()
    replace_with_mask = torch.bernoulli(
        torch.full(labels.shape, 0.8, device=input_ids.device)
    ).bool() & masked_indices
    masked_input_ids[replace_with_mask] = mask_token_id

    replace_with_random = torch.bernoulli(
        torch.full(labels.shape, 0.5, device=input_ids.device)
    ).bool() & masked_indices & ~replace_with_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=input_ids.device)
    masked_input_ids[replace_with_random] = random_words[replace_with_random]

    return masked_input_ids, labels


def train_mlm_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    special_token_ids: set[int],
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float = 0.15,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="mlm-train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        window_mask = batch["window_mask"].to(device)

        batch_size, num_windows, seq_len = input_ids.shape
        input_ids = input_ids.reshape(batch_size * num_windows, seq_len)
        attention_mask = attention_mask.reshape(batch_size * num_windows, seq_len)
        keep = window_mask.reshape(batch_size * num_windows)
        input_ids = input_ids[keep]
        attention_mask = attention_mask[keep]
        if input_ids.numel() == 0:
            continue

        masked_input_ids, labels = mask_mlm_inputs(
            input_ids=input_ids,
            special_token_ids=special_token_ids,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            probability=mlm_probability,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        steps += 1

    return {"loss": total_loss / max(steps, 1)}
