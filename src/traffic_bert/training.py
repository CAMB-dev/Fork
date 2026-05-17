"""Minimal training and evaluation loops."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable
import torch
from torch.utils.data import DataLoader
from rich.progress import BarColumn, MofNCompleteColumn, Progress, ProgressColumn, Task
from rich.progress import TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text


def resolve_device(device: str = "cuda") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class BatchSpeedColumn(ProgressColumn):
    """Render dataloader throughput as batches per second."""

    def render(self, task: Task) -> Text:
        if task.speed is None:
            return Text("- batch/s")
        return Text(f"{task.speed:.2f} batch/s")


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        BatchSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("loss={task.fields[current_loss]}"),
        TextColumn("avg={task.fields[avg_loss]}"),
    )


def _add_task(progress: Progress, description: str, total: int | None) -> TaskID:
    return progress.add_task(
        description,
        total=total,
        current_loss="-",
        avg_loss="-",
    )


def _format_loss(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


STEP_HISTORY_FIELDS = [
    "epoch",
    "step",
    "global_step",
    "loss",
    "avg_loss",
    "learning_rate",
    "samples_seen",
    "examples_per_second",
    "batch_accuracy",
    "running_accuracy",
    "masked_token_accuracy",
    "masked_tokens",
]


class StepHistoryWriter:
    """Append per-step metrics to JSONL and CSV history files."""

    def __init__(self, history_dir: str | Path) -> None:
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.history_dir / "steps.jsonl"
        self.csv_path = self.history_dir / "steps.csv"
        self._csv_needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

    def write(self, row: dict[str, Any]) -> None:
        normalized = {
            key: (float(value) if isinstance(value, torch.Tensor) else value)
            for key, value in row.items()
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=STEP_HISTORY_FIELDS,
                extrasaction="ignore",
            )
            if self._csv_needs_header:
                writer.writeheader()
                self._csv_needs_header = False
            writer.writerow(normalized)


def _step_summary(losses: list[float]) -> dict[str, float | None]:
    if not losses:
        return {
            "min_loss": None,
            "max_loss": None,
            "last_loss": None,
            "mean_step_loss": None,
        }
    return {
        "min_loss": min(losses),
        "max_loss": max(losses),
        "last_loss": losses[-1],
        "mean_step_loss": sum(losses) / len(losses),
    }


def train_classifier_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    global_step_start: int = 0,
    learning_rate: float | None = None,
    step_writer: StepHistoryWriter | None = None,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    losses: list[float] = []
    correct = 0
    total_examples = 0
    started_at = time.perf_counter()
    with _progress() as progress:
        task = _add_task(progress, "train", len(dataloader))
        for batch in dataloader:
            step_started_at = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            major_labels = batch["major_labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                window_mask=batch["window_mask"].to(device),
                context_features=batch.get("context_features").to(device)
                if batch.get("context_features") is not None
                else None,
                major_labels=major_labels,
                minor_labels=batch["minor_labels"].to(device),
            )
            loss = outputs["loss"]
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            current_loss = float(loss.detach().cpu())
            total_loss += current_loss
            steps += 1
            losses.append(current_loss)
            batch_correct = int((outputs["major_logits"].argmax(dim=-1) == major_labels).sum().item())
            batch_size = int(major_labels.numel())
            correct += batch_correct
            total_examples += batch_size
            global_step = global_step_start + steps
            batch_elapsed = max(time.perf_counter() - step_started_at, 1e-12)
            batch_accuracy = batch_correct / max(batch_size, 1)
            running_accuracy = correct / max(total_examples, 1)
            avg_loss = total_loss / max(steps, 1)
            if step_writer is not None:
                step_writer.write(
                    {
                        "epoch": epoch,
                        "step": steps,
                        "global_step": global_step,
                        "loss": current_loss,
                        "avg_loss": avg_loss,
                        "learning_rate": learning_rate,
                        "samples_seen": total_examples,
                        "examples_per_second": batch_size / batch_elapsed,
                        "batch_accuracy": batch_accuracy,
                        "running_accuracy": running_accuracy,
                    }
                )
            progress.update(
                task,
                advance=1,
                current_loss=_format_loss(current_loss),
                avg_loss=_format_loss(avg_loss),
            )
    mean_loss = total_loss / max(steps, 1)
    return {
        "loss": mean_loss,
        "train_loss": mean_loss,
        "train_accuracy": correct / max(total_examples, 1),
        "final_running_accuracy": correct / max(total_examples, 1),
        "global_step": global_step_start + steps,
        "examples_per_second": total_examples / max(time.perf_counter() - started_at, 1e-12),
        **_step_summary(losses),
    }


@torch.no_grad()
def collect_classifier_outputs(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_loss: bool = True,
) -> dict[str, list]:
    model.eval()
    major_logits: list[torch.Tensor] = []
    minor_logits: list[torch.Tensor] = []
    major_labels: list[torch.Tensor] = []
    minor_labels: list[torch.Tensor] = []
    flow_ids: list[str] = []
    total_loss = 0.0
    steps = 0

    with _progress() as progress:
        task = _add_task(progress, "eval", len(dataloader))
        for batch in dataloader:
            batch_major_labels = batch["major_labels"].to(device)
            batch_minor_labels = batch["minor_labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                window_mask=batch["window_mask"].to(device),
                context_features=batch.get("context_features").to(device)
                if batch.get("context_features") is not None
                else None,
                major_labels=batch_major_labels if compute_loss else None,
                minor_labels=batch_minor_labels if compute_loss else None,
            )
            if compute_loss and "loss" in outputs:
                total_loss += float(outputs["loss"].detach().cpu())
                steps += 1
            major_logits.append(outputs["major_logits"].cpu())
            minor_logits.append(outputs["minor_logits"].cpu())
            major_labels.append(batch["major_labels"].cpu())
            minor_labels.append(batch["minor_labels"].cpu())
            flow_ids.extend(batch["flow_ids"])
            progress.update(task, advance=1)

    return {
        "major_logits": torch.cat(major_logits) if major_logits else torch.empty(0),
        "minor_logits": torch.cat(minor_logits) if minor_logits else torch.empty(0),
        "major_labels": torch.cat(major_labels) if major_labels else torch.empty(0),
        "minor_labels": torch.cat(minor_labels) if minor_labels else torch.empty(0),
        "flow_ids": flow_ids,
        "eval_loss": total_loss / max(steps, 1) if steps else None,
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


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def write_history_files(history_dir: str | Path, history: list[dict[str, Any]]) -> None:
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(history_dir / "train.jsonl", "w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    flat_rows = [flatten_metrics(row) for row in history]
    fieldnames: list[str] = []
    preferred = [
        "epoch",
        "loss",
        "train_loss",
        "train_accuracy",
        "learning_rate",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_weighted_f1",
    ]
    for name in preferred:
        if any(name in row for row in flat_rows):
            fieldnames.append(name)
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(history_dir / "train.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def save_epoch_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "history": history,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    save_checkpoint(Path(output_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt", model, payload)


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
    epoch: int = 1,
    global_step_start: int = 0,
    learning_rate: float | None = None,
    step_writer: StepHistoryWriter | None = None,
    mlm_probability: float = 0.15,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    losses: list[float] = []
    total_masked_tokens = 0
    total_masked_correct = 0
    total_examples = 0
    started_at = time.perf_counter()

    with _progress() as progress:
        task = _add_task(progress, "mlm-train", len(dataloader))
        for batch in dataloader:
            step_started_at = time.perf_counter()
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
                progress.update(task, advance=1)
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

            current_loss = float(loss.detach().cpu())
            total_loss += current_loss
            steps += 1
            losses.append(current_loss)
            masked = labels != -100
            masked_tokens = int(masked.sum().item())
            masked_correct = (
                int((outputs.logits.argmax(dim=-1)[masked] == labels[masked]).sum().item())
                if masked_tokens
                else 0
            )
            total_masked_tokens += masked_tokens
            total_masked_correct += masked_correct
            total_examples += int(batch["major_labels"].numel())
            avg_loss = total_loss / max(steps, 1)
            batch_elapsed = max(time.perf_counter() - step_started_at, 1e-12)
            if step_writer is not None:
                step_writer.write(
                    {
                        "epoch": epoch,
                        "step": steps,
                        "global_step": global_step_start + steps,
                        "loss": current_loss,
                        "avg_loss": avg_loss,
                        "learning_rate": learning_rate,
                        "samples_seen": total_examples,
                        "examples_per_second": int(batch["major_labels"].numel()) / batch_elapsed,
                        "masked_token_accuracy": masked_correct / max(masked_tokens, 1),
                        "masked_tokens": masked_tokens,
                    }
                )
            progress.update(
                task,
                advance=1,
                current_loss=_format_loss(current_loss),
                avg_loss=_format_loss(avg_loss),
            )

    mean_loss = total_loss / max(steps, 1)
    return {
        "loss": mean_loss,
        "train_loss": mean_loss,
        "masked_token_accuracy": total_masked_correct / max(total_masked_tokens, 1),
        "masked_tokens": total_masked_tokens,
        "global_step": global_step_start + steps,
        "examples_per_second": total_examples / max(time.perf_counter() - started_at, 1e-12),
        **_step_summary(losses),
    }


def rich_train_batches(
    dataloader: Iterable,
    description: str,
) -> Iterable:
    with _progress() as progress:
        total = len(dataloader) if hasattr(dataloader, "__len__") else None
        task = _add_task(progress, description, total)
        for batch in dataloader:
            yield batch, progress, task
