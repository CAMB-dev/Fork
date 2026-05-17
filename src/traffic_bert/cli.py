"""Command line entrypoint for the Traffic Byte-BERT project."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import typer

from traffic_bert import __version__
from traffic_bert.baselines import (
    evaluate_major_baseline,
    load_baseline,
    save_baseline,
    train_major_baseline,
)
from traffic_bert.calibration import calibrate_thresholds
from traffic_bert.config import append_jsonl, load_yaml, set_seed, write_json
from traffic_bert.data.build import BuildConfig, build_config_from_yaml, build_processed_dataset
from traffic_bert.data.dataset import CONTEXT_FEATURE_COLUMNS, FlowWindowDataset, flow_collate
from traffic_bert.data.pcap import PcapFlowExtractor
from traffic_bert.data.payload_csv import PayloadCsvBuildConfig, build_payload_csv_dataset
from traffic_bert.data.schema import InputView
from traffic_bert.data.split import (
    assign_file_time_split,
    assign_stratified_hash_split,
    processed_stats,
    split_run_stats,
    stratified_sample,
)
from traffic_bert.data.validate import validation_summary
from traffic_bert.inference import decode_hierarchical_prediction
from traffic_bert.labels import LabelMap
from traffic_bert.metrics import constrain_minor_predictions, major_classification_metrics
from traffic_bert.metrics import multilabel_f1, multilabel_report_from_predictions
from traffic_bert.metrics import multilabel_scores_from_predictions
from traffic_bert.metrics import attack_detection_metrics
from traffic_bert.metrics import multilabel_classification_report
from traffic_bert.metrics_io import export_major_metrics, export_minor_metrics
from traffic_bert.models import (
    ByteCnnClassifier,
    ByteGruClassifier,
    ByteTransformerClassifier,
    ByteBertForHierarchicalClassification,
    create_bert_config,
    create_mlm_model,
)
from traffic_bert.sce import (
    SemanticCodebook,
    learn_frequency_codebook,
    learn_label_lift_codebook,
)
from traffic_bert.tokenizer import (
    CLS_TOKEN,
    MASK_TOKEN,
    PAD_TOKEN,
    PKT_BWD_TOKEN,
    PKT_END_TOKEN,
    PKT_FWD_TOKEN,
    SEP_TOKEN,
    ByteTokenizer,
    PacketChunk,
)
from traffic_bert.training import (
    collect_classifier_outputs,
    resolve_device,
    save_epoch_checkpoint,
    save_checkpoint,
    StepHistoryWriter,
    train_classifier_epoch,
    train_mlm_epoch,
    write_history_files,
)

app = typer.Typer(help="Traffic Byte-BERT experiment toolkit.")
data_app = typer.Typer(help="Data preprocessing commands.")
vocab_app = typer.Typer(help="Vocabulary utilities.")
train_app = typer.Typer(help="Training commands.")
eval_app = typer.Typer(help="Evaluation commands.")
predict_app = typer.Typer(help="Prediction commands.")
baseline_app = typer.Typer(help="Classical baseline commands.")
sce_app = typer.Typer(help="Semantic Conversion Encoder utilities.")

app.add_typer(data_app, name="data")
app.add_typer(vocab_app, name="vocab")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(predict_app, name="predict")
app.add_typer(baseline_app, name="baseline")
app.add_typer(sce_app, name="sce")


@app.command()
def version() -> None:
    """Print the package version."""

    typer.echo(__version__)


@vocab_app.command("write")
def write_vocab(path: Path = typer.Argument(..., help="Output vocab text file.")) -> None:
    """Write the fixed byte vocabulary to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ByteTokenizer().save_vocab(str(path))
    typer.echo(f"wrote vocab to {path}")


def _read_label_reservoir_sample(
    *,
    input_path: Path,
    columns: list[str],
    view: str | None,
    label_column: str,
    max_rows_per_label: int,
    sample_seed: int,
) -> pd.DataFrame:
    rng = random.Random(sample_seed)
    seen: Counter[str] = Counter()
    reservoirs: dict[str, list[dict[str, object]]] = {}
    parquet = pq.ParquetFile(input_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        frame = batch.to_pandas()
        if view is not None:
            frame = frame[frame["view"] == view]
        if frame.empty:
            continue
        for record in frame[["bytes", label_column]].to_dict("records"):
            label = str(record[label_column])
            seen[label] += 1
            bucket = reservoirs.setdefault(label, [])
            if len(bucket) < max_rows_per_label:
                bucket.append(record)
                continue
            index = rng.randrange(seen[label])
            if index < max_rows_per_label:
                bucket[index] = record
    rows = [record for label in sorted(reservoirs) for record in reservoirs[label]]
    return pd.DataFrame(rows, columns=["bytes", label_column])


@sce_app.command("build-codebook")
def build_sce_codebook(
    input_path: Path = typer.Option(..., help="Processed Parquet file to learn from."),
    output_path: Path = typer.Option(..., help="Output SCE codebook JSON."),
    view: Optional[str] = typer.Option(None, help="Optional view filter."),
    ranking: str = typer.Option(
        "frequency",
        help="Codebook ranking: frequency or label_lift.",
    ),
    label_column: str = typer.Option("major_label", help="Label column used by label_lift."),
    target_label: Optional[str] = typer.Option(
        None,
        help="Optional target label for label_lift ranking.",
    ),
    chunk_size: int = typer.Option(4, min=1, help="Byte chunk size for frequency tokens."),
    max_entries: int = typer.Option(256, min=1, help="Maximum codebook entries."),
    min_count: int = typer.Option(2, min=1, help="Minimum chunk frequency to keep."),
    max_rows: Optional[int] = typer.Option(None, min=1, help="Optional row cap for quick builds."),
    max_rows_per_label: Optional[int] = typer.Option(
        None,
        min=1,
        help="For label_lift, deterministic reservoir cap per label.",
    ),
    sample_seed: int = typer.Option(42, help="Seed used by label-lift reservoir sampling."),
    drop_zero_chunks: bool = typer.Option(
        True,
        "--drop-zero-chunks/--keep-zero-chunks",
        help="Skip all-zero chunks that mostly come from masked header fields.",
    ),
) -> None:
    """Build a deterministic SCE byte-chunk codebook from processed flow bytes."""

    if ranking not in {"frequency", "label_lift"}:
        raise typer.BadParameter("ranking must be one of: frequency, label_lift")
    if max_rows is not None and max_rows_per_label is not None:
        raise typer.BadParameter("use either max_rows or max_rows_per_label, not both")
    if ranking != "label_lift" and max_rows_per_label is not None:
        raise typer.BadParameter("max_rows_per_label is only supported with label_lift ranking")
    columns = ["bytes"]
    schema = set(pq.read_schema(input_path).names)
    if view is not None:
        if "view" not in schema:
            raise typer.BadParameter("input parquet has no view column")
        columns.append("view")
    if ranking == "label_lift":
        if label_column not in schema:
            raise typer.BadParameter(f"input parquet has no label column: {label_column}")
        columns.append(label_column)
    if max_rows_per_label is not None:
        frame = _read_label_reservoir_sample(
            input_path=input_path,
            columns=columns,
            view=view,
            label_column=label_column,
            max_rows_per_label=max_rows_per_label,
            sample_seed=sample_seed,
        )
    else:
        frame = pd.read_parquet(input_path, columns=columns)
        if view is not None:
            frame = frame[frame["view"] == view]
        if max_rows is not None:
            frame = frame.head(max_rows)
    if ranking == "frequency":
        codebook = learn_frequency_codebook(
            frame["bytes"].tolist(),
            chunk_size=chunk_size,
            max_entries=max_entries,
            min_count=min_count,
            drop_zero_chunks=drop_zero_chunks,
        )
    else:
        codebook = learn_label_lift_codebook(
            frame["bytes"].tolist(),
            frame[label_column].astype(str).tolist(),
            chunk_size=chunk_size,
            max_entries=max_entries,
            min_count=min_count,
            drop_zero_chunks=drop_zero_chunks,
            target_label=target_label,
        )
    codebook.save(output_path)
    payload = {
        "output_path": str(output_path),
        "input_path": str(input_path),
        "view": view,
        "ranking": ranking,
        "label_column": label_column if ranking == "label_lift" else None,
        "target_label": target_label,
        "rows_used": int(len(frame)),
        "chunk_size": chunk_size,
        "max_entries": max_entries,
        "min_count": min_count,
        "max_rows_per_label": max_rows_per_label,
        "sample_seed": sample_seed if max_rows_per_label is not None else None,
        "drop_zero_chunks": drop_zero_chunks,
        "entries": len(codebook.entries),
        "top_entries": codebook.to_dict()["entries"][:10],
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@data_app.command("build")
def build_data(
    input_path: Path = typer.Option(..., help="PCAP file or directory."),
    output_path: Path = typer.Option(..., help="Output Parquet path."),
    label_map: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    source_dataset: str = typer.Option("custom", help="Dataset name stored in rows."),
    split: str = typer.Option("train", help="Split name stored in rows."),
    label_source: str = typer.Option(
        "filename",
        help="filename, parent, parent_filename, static, or cic_csv.",
    ),
    static_label: Optional[str] = typer.Option(None, help="Label used with label_source=static."),
    views: str = typer.Option(
        "payload_only,full_packet,masked_header_packet",
        help="Comma-separated input views.",
    ),
    keep_empty_payload: bool = typer.Option(True, help="Keep flows without L4 payload."),
    max_packets_per_flow: Optional[int] = typer.Option(None, help="Optional packet cap per flow."),
    max_packets_to_read: Optional[int] = typer.Option(
        None,
        help="Optional packet read cap for smoke builds.",
    ),
    max_packets_to_skip: int = typer.Option(0, help="Optional packet skip count for smoke builds."),
    min_packet_time: Optional[float] = typer.Option(
        None,
        help="Optional minimum packet Unix timestamp for smoke builds.",
    ),
    max_packet_time: Optional[float] = typer.Option(
        None,
        help="Optional maximum packet Unix timestamp for smoke builds.",
    ),
    flow_timeout_seconds: Optional[float] = typer.Option(
        120.0,
        help="Idle timeout in seconds before a repeated five-tuple starts a new flow session.",
    ),
    label_csv: Optional[Path] = typer.Option(None, help="CIC-style flow label CSV."),
    drop_unmatched_labels: bool = typer.Option(True, help="Drop flows without matched labels."),
) -> None:
    """Build processed Parquet data from PCAP files."""

    selected_views = tuple(InputView(item.strip()) for item in views.split(",") if item.strip())
    stats = build_processed_dataset(
        BuildConfig(
            input_path=input_path,
            output_path=output_path,
            label_map_path=label_map,
            source_dataset=source_dataset,
            split=split,
            label_source=label_source,
            static_label=static_label,
            label_csv_path=label_csv,
            drop_unmatched_labels=drop_unmatched_labels,
            views=selected_views,
            keep_empty_payload=keep_empty_payload,
            max_packets_per_flow=max_packets_per_flow,
            max_packets_to_read=max_packets_to_read,
            max_packets_to_skip=max_packets_to_skip,
            min_packet_time=min_packet_time,
            max_packet_time=max_packet_time,
            flow_timeout_seconds=flow_timeout_seconds,
        )
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@data_app.command("sample-stratified")
def sample_stratified_data(
    input_path: Path = typer.Option(..., help="Input processed Parquet."),
    output_dir: Path = typer.Option(..., help="Directory for sampled split files."),
    stratify_column: str = typer.Option("major_label", help="Column capped per class."),
    split_stratify_column: str = typer.Option(
        "source_label",
        help="Column used to preserve labels across train/val/test.",
    ),
    group_column: str = typer.Option("flow_id", help="Column kept within one split."),
    max_per_class: int = typer.Option(2_000, help="Maximum rows per class before splitting."),
    train_ratio: float = typer.Option(0.7, help="Training split ratio."),
    val_ratio: float = typer.Option(0.15, help="Validation split ratio."),
    seed: int = typer.Option(42, help="Deterministic sampling seed."),
) -> None:
    """Create a small stratified train/val/test subset from processed data."""

    frame = pd.read_parquet(input_path)
    input_frame = frame
    sampled = stratified_sample(
        frame,
        stratify_column=stratify_column,
        max_per_class=max_per_class,
        seed=seed,
    )
    sampled = assign_stratified_hash_split(
        sampled,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        group_column=group_column,
        stratify_column=split_stratify_column,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_frame = sampled[sampled["split"] == split_name]
        split_frame.to_parquet(output_dir / f"{split_name}.parquet", index=False)
        split_frame["major_label"].value_counts().rename_axis("major_label").reset_index(
            name="rows"
        ).to_csv(output_dir / f"{split_name}.class_distribution.csv", index=False)
    stats = split_run_stats(
        input_frame=input_frame,
        output_frame=sampled,
        split_method="random_flow_hash",
        max_per_class=max_per_class,
        stratify_column=stratify_column,
        split_stratify_column=split_stratify_column,
        group_column=group_column,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    write_json(output_dir / "split.stats.json", stats)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@data_app.command("build-config")
def build_data_config(config: Path = typer.Option(..., help="YAML build config.")) -> None:
    """Build one or more processed datasets from a YAML config."""

    results = []
    for build_config in build_config_from_yaml(config):
        stats = build_processed_dataset(build_config)
        results.append({"output_path": str(build_config.output_path), "stats": stats})
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))


@data_app.command("build-payload-csv")
def build_payload_csv(
    input_path: Path = typer.Option(..., help="Payload-Byte style CSV file."),
    output_path: Path = typer.Option(..., help="Output Parquet path."),
    label_map: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    source_dataset: str = typer.Option("payload-byte", help="Dataset name stored in rows."),
    split: str = typer.Option("train", help="Split name stored in rows."),
    label_column: str = typer.Option("label", help="CSV label column."),
    byte_prefix: str = typer.Option("payload_byte_", help="Prefix of byte columns."),
    chunksize: int = typer.Option(20_000, help="Rows read per CSV chunk."),
    keep_empty_payload: bool = typer.Option(False, help="Keep rows with all-zero payload bytes."),
    trim_trailing_zeros: bool = typer.Option(True, help="Trim zero padding at payload tail."),
    compression: str = typer.Option("zstd", help="Parquet compression codec."),
) -> None:
    """Build processed Parquet data from Payload-Byte packet CSV files."""

    stats = build_payload_csv_dataset(
        PayloadCsvBuildConfig(
            input_path=input_path,
            output_path=output_path,
            label_map_path=label_map,
            source_dataset=source_dataset,
            split=split,
            label_column=label_column,
            byte_prefix=byte_prefix,
            chunksize=chunksize,
            keep_empty_payload=keep_empty_payload,
            trim_trailing_zeros=trim_trailing_zeros,
            compression=compression,
        )
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@data_app.command("split")
def split_data(
    input_path: Path = typer.Option(..., help="Input processed Parquet."),
    output_dir: Path = typer.Option(..., help="Directory for train/val/test Parquet files."),
    group_column: str = typer.Option("source_file", help="Column kept within one split."),
    stratify_column: Optional[str] = typer.Option(
        None,
        help="Optional label column for stable stratified group splitting.",
    ),
    train_ratio: float = typer.Option(0.7, help="Training split ratio."),
    val_ratio: float = typer.Option(0.15, help="Validation split ratio."),
) -> None:
    """Assign stable group-wise train/val/test splits and write split Parquet files."""

    frame = pd.read_parquet(input_path)
    if stratify_column is None:
        frame = assign_file_time_split(
            frame,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            group_column=group_column,
        )
    else:
        frame = assign_stratified_hash_split(
            frame,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            group_column=group_column,
            stratify_column=stratify_column,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        frame[frame["split"] == split_name].to_parquet(output_dir / f"{split_name}.parquet", index=False)
    stats = processed_stats(frame)
    write_json(output_dir / "split.stats.json", stats)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@data_app.command("merge")
def merge_data(
    input_paths: list[Path] = typer.Option(..., "--input-path", help="Input Parquet path."),
    output_path: Path = typer.Option(..., help="Merged output Parquet path."),
    compression: str = typer.Option("zstd", help="Parquet compression codec."),
) -> None:
    """Merge multiple processed Parquet files with the same schema."""

    if not input_paths:
        raise typer.BadParameter("at least one --input-path is required")
    tables = [pq.read_table(path) for path in input_paths]
    table = pa.concat_tables(tables, promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression=compression)

    stats_columns = [
        column
        for column in [
            "flow_id",
            "split",
            "view",
            "major_label",
            "minor_labels",
            "source_label",
            "source_dataset",
            "packet_count",
            "payload_byte_length",
            "packet_byte_length",
        ]
        if column in table.column_names
    ]
    stats = processed_stats(table.select(stats_columns).to_pandas())
    result = {"output_path": str(output_path), **stats}
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@data_app.command("stats")
def data_stats(
    input_path: Path = typer.Option(..., help="Input processed Parquet."),
    output_dir: Optional[Path] = typer.Option(None, help="Optional output directory."),
) -> None:
    """Generate processed-data statistics and optional class distribution CSV."""

    frame = pd.read_parquet(input_path)
    stats = processed_stats(frame)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "stats.json", stats)
        distribution = (
            frame.groupby(["split", "view", "major_label"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@data_app.command("validate")
def validate_data(input_path: Path = typer.Option(..., help="Input processed Parquet.")) -> None:
    """Validate processed-data schema and common quality issues."""

    frame = pd.read_parquet(input_path)
    summary = validation_summary(frame)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise typer.Exit(code=1)


def _make_classifier(
    label_map: LabelMap,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 512,
    major_class_weights: Optional[list[float]] = None,
    vocab_size: int | None = None,
    context_feature_size: int = 0,
) -> ByteBertForHierarchicalClassification:
    config_kwargs = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
    }
    if vocab_size is not None:
        config_kwargs["vocab_size"] = int(vocab_size)
    config = create_bert_config(**config_kwargs)
    return ByteBertForHierarchicalClassification(
        bert_config=config,
        num_major_labels=len(label_map.major_labels),
        num_minor_labels=len(label_map.minor_labels),
        minor_to_major=label_map.minor_major_ids(),
        major_class_weights=major_class_weights,
        context_feature_size=context_feature_size,
    )


def _compute_major_class_weights(
    frame: pd.DataFrame,
    label_map: LabelMap,
    mode: str,
    cap: float,
) -> list[float] | None:
    if mode == "none":
        return None
    if mode not in {"balanced", "sqrt_balanced"}:
        raise typer.BadParameter(
            "major class weighting must be one of: none, balanced, sqrt_balanced"
        )

    label_ids = frame["major_label"].map(lambda value: label_map.major_id(str(value)))
    counts = label_ids.value_counts().to_dict()
    present = [idx for idx in range(len(label_map.major_labels)) if counts.get(idx, 0) > 0]
    if not present:
        return None

    total = float(sum(counts[idx] for idx in present))
    class_count = float(len(present))
    cap = float(cap)
    weights: list[float] = []
    for idx in range(len(label_map.major_labels)):
        count = int(counts.get(idx, 0))
        if count <= 0:
            weights.append(0.0)
            continue
        value = total / (class_count * float(count))
        if mode == "sqrt_balanced":
            value = value**0.5
        if cap > 0:
            value = min(value, cap)
            value = max(value, 1.0 / cap)
        weights.append(float(value))
    return weights


def _make_neural_baseline(
    model_name: str,
    vocab_size: int,
    num_labels: int,
    hidden_size: int,
    max_length: int,
) -> torch.nn.Module:
    if model_name == "cnn":
        return ByteCnnClassifier(vocab_size=vocab_size, num_labels=num_labels, hidden_size=hidden_size)
    if model_name == "gru":
        return ByteGruClassifier(vocab_size=vocab_size, num_labels=num_labels, hidden_size=hidden_size)
    if model_name == "transformer":
        return ByteTransformerClassifier(
            vocab_size=vocab_size,
            num_labels=num_labels,
            hidden_size=hidden_size,
            max_length=max_length,
        )
    raise ValueError(f"unsupported neural baseline: {model_name}")


def _loader_kwargs(num_workers: int, device: torch.device) -> dict:
    workers = max(0, int(num_workers))
    kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def _train_neural_baseline_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    global_step_start: int = 0,
    learning_rate: float | None = None,
    step_writer: StepHistoryWriter | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    losses = []
    correct = 0
    total_examples = 0
    import time

    started_at = time.perf_counter()
    from traffic_bert.training import rich_train_batches

    for batch, progress, task in rich_train_batches(dataloader, "neural-train"):
        step_started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        labels = batch["major_labels"].to(device)
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        steps += 1
        losses.append(current_loss)
        batch_correct = int((logits.argmax(dim=-1) == labels).sum().item())
        batch_size = int(labels.numel())
        correct += batch_correct
        total_examples += batch_size
        avg_loss = total_loss / max(steps, 1)
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
                    "examples_per_second": batch_size
                    / max(time.perf_counter() - step_started_at, 1e-12),
                    "batch_accuracy": batch_correct / max(batch_size, 1),
                    "running_accuracy": correct / max(total_examples, 1),
                }
            )
        progress.update(
            task,
            advance=1,
            current_loss=f"{current_loss:.4f}",
            avg_loss=f"{avg_loss:.4f}",
        )
    mean_loss = total_loss / max(steps, 1)
    return {
        "loss": mean_loss,
        "train_loss": mean_loss,
        "train_accuracy": correct / max(total_examples, 1),
        "final_running_accuracy": correct / max(total_examples, 1),
        "global_step": global_step_start + steps,
        "examples_per_second": total_examples / max(time.perf_counter() - started_at, 1e-12),
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_step_loss": mean_loss if losses else None,
    }


@torch.no_grad()
def _eval_neural_baseline(
    model: torch.nn.Module,
    data_path: Path,
    label_map: LabelMap,
    view: str,
    batch_size: int,
    max_length: int,
    stride: int,
    max_windows: Optional[int],
    device: torch.device,
    num_workers: int = 0,
    output_dir: Optional[Path] = None,
) -> dict:
    dataset = FlowWindowDataset.from_parquet(
        data_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device),
    )
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    steps = 0
    from traffic_bert.training import rich_train_batches

    for batch, progress, task in rich_train_batches(loader, "neural-eval"):
        labels = batch["major_labels"].to(device)
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        total_loss += float(torch.nn.functional.cross_entropy(logits, labels).detach().cpu())
        steps += 1
        y_true.extend(batch["major_labels"].tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        progress.update(task, advance=1)
    metrics = major_classification_metrics(y_true, y_pred, label_map.major_labels)
    metrics["eval_loss"] = total_loss / max(steps, 1) if steps else None
    if output_dir is not None:
        export_major_metrics(output_dir, metrics, y_true, y_pred, label_map.major_labels)
        from traffic_bert.metrics import attack_detection_metrics

        metrics["detection"] = attack_detection_metrics(y_true, y_pred, label_map.major_labels)
        write_json(output_dir / "metrics.json", metrics)
    return metrics


def _load_classifier(
    checkpoint: Path,
    label_map: LabelMap,
    max_length: int,
) -> ByteBertForHierarchicalClassification:
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = payload.get("model_config", {})
    vocab_size = model_config.get("vocab_size")
    if vocab_size is None:
        embedding = payload["model_state_dict"].get("bert.embeddings.word_embeddings.weight")
        if embedding is not None:
            vocab_size = int(embedding.shape[0])
    model = _make_classifier(
        label_map,
        hidden_size=model_config.get("hidden_size", 256),
        num_layers=model_config.get("num_hidden_layers", 4),
        num_heads=model_config.get("num_attention_heads", 4),
        intermediate_size=model_config.get("intermediate_size", 1024),
        max_position_embeddings=model_config.get("max_position_embeddings", max_length),
        major_class_weights=payload.get("major_class_weights"),
        vocab_size=vocab_size,
        context_feature_size=int(payload.get("context_feature_size") or 0),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


def _load_thresholds(path: Optional[Path]) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "thresholds" in payload:
        return {str(key): float(value) for key, value in payload["thresholds"].items()}
    return {str(key): float(value) for key, value in payload.items()}


def _load_semantic_codebook(path: Optional[Path]) -> SemanticCodebook | None:
    if path is None:
        return None
    return SemanticCodebook.load(path)


def _semantic_codebook_from_checkpoint(checkpoint: Path) -> SemanticCodebook | None:
    payload = torch.load(checkpoint, map_location="cpu")
    raw = payload.get("semantic_codebook")
    if isinstance(raw, dict):
        return SemanticCodebook.from_dict(raw)
    raw_path = payload.get("semantic_codebook_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return SemanticCodebook.load(path)
    return None


def _resolve_semantic_codebook(
    path: Optional[Path],
    *,
    checkpoint: Optional[Path] = None,
) -> SemanticCodebook | None:
    explicit = _load_semantic_codebook(path)
    if explicit is not None:
        return explicit
    if checkpoint is not None:
        return _semantic_codebook_from_checkpoint(checkpoint)
    return None


def _semantic_codebook_checkpoint_metadata(
    codebook: SemanticCodebook | None,
    path: Optional[Path],
) -> dict[str, object]:
    return {
        "semantic_codebook_path": str(path) if path is not None else None,
        "semantic_codebook": codebook.to_dict() if codebook is not None else None,
    }


def _write_classifier_predictions(
    path: Path,
    *,
    flow_ids: list[str],
    y_true: list[int],
    y_pred: list[int],
    major_prob: torch.Tensor,
    label_map: LabelMap,
) -> None:
    records: dict[str, object] = {
        "flow_id": flow_ids,
        "true_major_id": y_true,
        "pred_major_id": y_pred,
        "true_major_label": [label_map.major_labels[idx] for idx in y_true],
        "pred_major_label": [label_map.major_labels[idx] for idx in y_pred],
    }
    for idx, label in enumerate(label_map.major_labels):
        records[f"prob_{label}"] = major_prob[:, idx].tolist()
    frame = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _endpoint_host(endpoint: object) -> str:
    value = str(endpoint or "")
    if not value:
        return "unknown"
    if value.startswith("[") and "]:" in value:
        return value[1:].split("]:", 1)[0]
    if ":" in value:
        return value.rsplit(":", 1)[0]
    return value


def _summarize_pcap_host_windows(
    predictions: list[dict],
    *,
    risk_threshold: float,
    window_seconds: tuple[int, ...] = (30, 60),
    top_windows: int = 20,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for seconds in window_seconds:
        buckets: dict[tuple[str, str, int], dict[str, object]] = {}
        for item in predictions:
            host = _endpoint_host(item.get("initiator_endpoint") or item.get("endpoint_a"))
            source_file = str(item.get("source_file") or "")
            try:
                start_time = float(item.get("start_time") or 0.0)
            except (TypeError, ValueError):
                start_time = 0.0
            bucket_start = int(start_time // seconds) * int(seconds)
            stats = buckets.setdefault(
                (source_file, host, bucket_start),
                {
                    "host": host,
                    "source_file": source_file,
                    "window_seconds": int(seconds),
                    "window_start_time": float(bucket_start),
                    "window_end_time": float(bucket_start + int(seconds)),
                    "flow_count": 0,
                    "predicted_attack_flow_count": 0,
                    "high_risk_flow_count": 0,
                    "major_label_counts": Counter(),
                    "connection_type_counts": Counter(),
                    "max_major_prob": 0.0,
                },
            )
            stats["flow_count"] = int(stats["flow_count"]) + 1
            label = str(item.get("major_label", "unknown"))
            stats["major_label_counts"][label] += 1
            connection_type = str(item.get("connection_type", "unknown"))
            stats["connection_type_counts"][connection_type] += 1
            probability = float(item.get("major_prob") or 0.0)
            stats["max_major_prob"] = max(float(stats["max_major_prob"]), probability)
            if label != "benign":
                stats["predicted_attack_flow_count"] = int(stats["predicted_attack_flow_count"]) + 1
                if probability >= risk_threshold:
                    stats["high_risk_flow_count"] = int(stats["high_risk_flow_count"]) + 1

        rows = []
        for stats in buckets.values():
            stats["major_label_counts"] = dict(stats["major_label_counts"])
            stats["connection_type_counts"] = dict(stats["connection_type_counts"])
            stats["predicted_attack_ratio"] = int(stats["predicted_attack_flow_count"]) / max(
                int(stats["flow_count"]),
                1,
            )
            rows.append(stats)
        rows.sort(
            key=lambda item: (
                int(item["high_risk_flow_count"]),
                int(item["predicted_attack_flow_count"]),
                float(item["max_major_prob"]),
                int(item["flow_count"]),
            ),
            reverse=True,
        )
        output[str(seconds)] = rows[: max(0, int(top_windows))]
    return output


def _load_host_window_policies(path: Optional[Path]) -> list[dict[str, object]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("policies"), list):
        return [
            dict(item)
            for item in payload["policies"]
            if bool(item.get("enabled", item.get("test_passed_requirements", True)))
        ]
    target_label = str(payload.get("target_label") or "botnet_malware")
    min_recall = float(payload.get("min_recall", 0.0))
    min_f1 = float(payload.get("min_f1", 0.0))
    policies: list[dict[str, object]] = []
    for seconds, report in sorted(payload.get("bucket_window_detection", {}).items()):
        val_metrics = report.get("val_at_chosen") or report.get("val_scan", {}).get("chosen") or {}
        test_metrics = report.get("test_at_chosen") or {}
        threshold = val_metrics.get("threshold")
        if threshold is None:
            continue
        if test_metrics and (
            float(test_metrics.get("recall", 0.0)) < min_recall
            or float(test_metrics.get("f1", 0.0)) < min_f1
        ):
            continue
        policies.append(
            {
                "target_label": target_label,
                "window_seconds": int(seconds),
                "threshold": float(threshold),
                "source": str(path),
            }
        )
    return policies


def _major_probability_for_label(item: dict, label: str) -> float:
    major_probs = item.get("major_probs")
    if isinstance(major_probs, dict) and label in major_probs:
        return float(major_probs[label] or 0.0)
    if str(item.get("major_label")) == label:
        return float(item.get("major_prob") or 0.0)
    return 0.0


def _summarize_policy_host_windows(
    predictions: list[dict],
    policies: list[dict[str, object]],
    *,
    top_windows: int,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for policy in policies:
        target_label = str(policy["target_label"])
        seconds = int(policy["window_seconds"])
        threshold = float(policy["threshold"])
        policy_key = f"{target_label}@{seconds}s"
        buckets: dict[tuple[str, str, int], dict[str, object]] = {}
        for item in predictions:
            host = _endpoint_host(item.get("initiator_endpoint") or item.get("endpoint_a"))
            source_file = str(item.get("source_file") or "")
            try:
                start_time = float(item.get("start_time") or 0.0)
            except (TypeError, ValueError):
                start_time = 0.0
            bucket_start = int(start_time // seconds) * seconds
            stats = buckets.setdefault(
                (source_file, host, bucket_start),
                {
                    "target_label": target_label,
                    "host": host,
                    "source_file": source_file,
                    "window_seconds": seconds,
                    "window_start_time": float(bucket_start),
                    "window_end_time": float(bucket_start + seconds),
                    "threshold": threshold,
                    "flow_count": 0,
                    "target_argmax_flow_count": 0,
                    "policy_hit_flow_count": 0,
                    "max_target_prob": 0.0,
                    "major_label_counts": Counter(),
                },
            )
            stats["flow_count"] = int(stats["flow_count"]) + 1
            label = str(item.get("major_label", "unknown"))
            stats["major_label_counts"][label] += 1
            score = _major_probability_for_label(item, target_label)
            stats["max_target_prob"] = max(float(stats["max_target_prob"]), score)
            if label == target_label:
                stats["target_argmax_flow_count"] = int(stats["target_argmax_flow_count"]) + 1
            if score >= threshold:
                stats["policy_hit_flow_count"] = int(stats["policy_hit_flow_count"]) + 1

        rows = []
        for stats in buckets.values():
            stats["major_label_counts"] = dict(stats["major_label_counts"])
            stats["policy_triggered"] = int(stats["policy_hit_flow_count"]) > 0
            rows.append(stats)
        rows.sort(
            key=lambda item: (
                bool(item["policy_triggered"]),
                int(item["policy_hit_flow_count"]),
                float(item["max_target_prob"]),
                int(item["flow_count"]),
            ),
            reverse=True,
        )
        output[policy_key] = rows[: max(0, int(top_windows))]
    return output


def _summarize_pcap_predictions(
    predictions: list[dict],
    risk_threshold: float = 0.5,
    top_flows: int = 20,
    host_window_policies: list[dict[str, object]] | None = None,
) -> dict:
    label_counts = Counter(str(item.get("major_label", "unknown")) for item in predictions)
    attack_predictions = [
        item for item in predictions if str(item.get("major_label", "unknown")) != "benign"
    ]
    high_risk = [
        item
        for item in attack_predictions
        if float(item.get("major_prob") or 0.0) >= risk_threshold
    ]
    high_risk.sort(key=lambda item: float(item.get("major_prob") or 0.0), reverse=True)
    connection_counts = Counter(str(item.get("connection_type", "unknown")) for item in predictions)
    host_stats: dict[str, dict[str, object]] = {}
    for item in predictions:
        host = _endpoint_host(item.get("initiator_endpoint") or item.get("endpoint_a"))
        stats = host_stats.setdefault(
            host,
            {
                "host": host,
                "flow_count": 0,
                "predicted_attack_flow_count": 0,
                "high_risk_flow_count": 0,
                "major_label_counts": Counter(),
                "max_major_prob": 0.0,
            },
        )
        stats["flow_count"] = int(stats["flow_count"]) + 1
        label = str(item.get("major_label", "unknown"))
        stats["major_label_counts"][label] += 1
        probability = float(item.get("major_prob") or 0.0)
        stats["max_major_prob"] = max(float(stats["max_major_prob"]), probability)
        if label != "benign":
            stats["predicted_attack_flow_count"] = int(stats["predicted_attack_flow_count"]) + 1
            if probability >= risk_threshold:
                stats["high_risk_flow_count"] = int(stats["high_risk_flow_count"]) + 1
    host_risk_summary = []
    for stats in host_stats.values():
        stats["major_label_counts"] = dict(stats["major_label_counts"])
        stats["predicted_attack_ratio"] = int(stats["predicted_attack_flow_count"]) / max(
            int(stats["flow_count"]),
            1,
        )
        host_risk_summary.append(stats)
    host_risk_summary.sort(
        key=lambda item: (
            int(item["high_risk_flow_count"]),
            int(item["predicted_attack_flow_count"]),
            float(item["max_major_prob"]),
            int(item["flow_count"]),
        ),
        reverse=True,
    )
    flow_count = len(predictions)
    attack_count = len(attack_predictions)
    return {
        "flow_count": flow_count,
        "predicted_benign_flow_count": int(label_counts.get("benign", 0)),
        "predicted_attack_flow_count": attack_count,
        "predicted_attack_ratio": attack_count / max(flow_count, 1),
        "major_label_counts": dict(label_counts),
        "connection_type_counts": dict(connection_counts),
        "risk_threshold": risk_threshold,
        "high_risk_flow_count": len(high_risk),
        "top_high_risk_flows": high_risk[: max(0, int(top_flows))],
        "top_host_risks": host_risk_summary[: max(0, int(top_flows))],
        "top_host_windows": _summarize_pcap_host_windows(
            predictions,
            risk_threshold=risk_threshold,
            top_windows=top_flows,
        ),
        "policy_host_windows": _summarize_policy_host_windows(
            predictions,
            host_window_policies or [],
            top_windows=top_flows,
        ),
    }


def _load_bert_encoder_from_mlm(
    model: ByteBertForHierarchicalClassification,
    checkpoint: Path,
) -> None:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model_state_dict"]
    bert_state = {
        key.removeprefix("bert."): value
        for key, value in state.items()
        if key.startswith("bert.")
    }
    missing, unexpected = model.bert.load_state_dict(bert_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected BERT keys when loading MLM checkpoint: {unexpected}")
    # Pooler weights may be missing depending on the source checkpoint; that is acceptable.
    non_pooler_missing = [key for key in missing if not key.startswith("pooler.")]
    if non_pooler_missing:
        raise ValueError(f"missing BERT keys when loading MLM checkpoint: {non_pooler_missing}")


def _evaluate_classifier_model(
    model: ByteBertForHierarchicalClassification,
    data_path: Path,
    label_map: LabelMap,
    view: str,
    batch_size: int,
    max_length: int,
    stride: int,
    max_windows: Optional[int],
    device: torch.device,
    num_workers: int = 0,
    output_dir: Optional[Path] = None,
    output_predictions_path: Optional[Path] = None,
    use_connection_tokens: bool = False,
    use_context_tokens: bool = False,
    use_context_features: bool = False,
    semantic_codebook: SemanticCodebook | None = None,
) -> dict:
    dataset = FlowWindowDataset.from_parquet(
        data_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
        use_connection_tokens=use_connection_tokens,
        use_context_tokens=use_context_tokens,
        use_context_features=use_context_features,
        semantic_codebook=semantic_codebook,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device),
    )
    outputs = collect_classifier_outputs(model, loader, device)
    y_true = outputs["major_labels"].numpy()
    major_prob = torch.softmax(outputs["major_logits"].float(), dim=-1)
    y_pred = major_prob.argmax(dim=-1).numpy()
    metrics = major_classification_metrics(y_true, y_pred, label_map.major_labels)
    metrics["eval_loss"] = outputs["eval_loss"]
    metrics["detection"] = attack_detection_metrics(y_true, y_pred, label_map.major_labels)
    minor_true = outputs["minor_labels"].numpy()
    minor_prob = torch.sigmoid(outputs["minor_logits"]).numpy()
    minor_pred = (minor_prob >= 0.5).astype(int)
    metrics["minor"] = multilabel_f1(
        minor_true,
        minor_prob,
    )
    constrained_minor_pred = constrain_minor_predictions(
        minor_pred,
        y_pred,
        label_map.minor_major_ids(),
    )
    metrics["minor_constrained"] = multilabel_scores_from_predictions(
        minor_true,
        constrained_minor_pred,
    )
    minor_report = multilabel_classification_report(
        minor_true,
        minor_prob,
        label_map.minor_labels,
    )
    constrained_minor_report = multilabel_report_from_predictions(
        minor_true,
        constrained_minor_pred,
        label_map.minor_labels,
    )
    metrics["minor_report"] = minor_report
    metrics["minor_constrained_report"] = constrained_minor_report
    if output_dir is not None:
        export_major_metrics(output_dir, metrics, y_true.tolist(), y_pred.tolist(), label_map.major_labels)
        export_minor_metrics(output_dir, minor_report, label_map.minor_labels)
        export_minor_metrics(
            output_dir / "minor_constrained",
            constrained_minor_report,
            label_map.minor_labels,
        )
        write_json(output_dir / "metrics.json", metrics)
    if output_predictions_path is not None:
        _write_classifier_predictions(
            output_predictions_path,
            flow_ids=outputs["flow_ids"],
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            major_prob=major_prob,
            label_map=label_map,
        )
    return metrics


@train_app.command("classifier")
def train_classifier(
    train_path: Path = typer.Option(..., help="Training Parquet file."),
    output_dir: Path = typer.Option(Path("artifacts/classifier"), help="Output directory."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    val_path: Optional[Path] = typer.Option(None, help="Optional validation Parquet file."),
    view: str = typer.Option("masked_header_packet", help="Input view to train on."),
    epochs: int = typer.Option(3, help="Number of epochs."),
    batch_size: int = typer.Option(4, help="Batch size."),
    learning_rate: float = typer.Option(3e-5, help="Learning rate."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(8, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    seed: int = typer.Option(42, help="Random seed."),
    log_path: Optional[Path] = typer.Option(None, help="Optional JSONL training log."),
    history_dir: Optional[Path] = typer.Option(None, help="Directory for JSONL/CSV history."),
    save_each_epoch: bool = typer.Option(True, help="Save one checkpoint per epoch."),
    resume_checkpoint: Optional[Path] = typer.Option(None, help="Resume model weights."),
    init_bert_checkpoint: Optional[Path] = typer.Option(None, help="MLM checkpoint for BERT init."),
    major_class_weighting: str = typer.Option(
        "none",
        help="Major-label loss weighting: none, balanced, or sqrt_balanced.",
    ),
    major_class_weight_cap: float = typer.Option(
        20.0,
        help="Maximum and reciprocal-minimum cap for computed major-label weights.",
    ),
    use_connection_tokens: bool = typer.Option(
        False,
        "--use-connection-tokens/--no-connection-tokens",
        help="Prefix each flow with a connection-type token from processed metadata.",
    ),
    use_context_tokens: bool = typer.Option(
        False,
        "--use-context-tokens/--no-context-tokens",
        help="Prefix each flow with host-window context tokens when context columns exist.",
    ),
    use_context_features: bool = typer.Option(
        False,
        "--use-context-features/--no-context-features",
        help="Concatenate log-scaled host-window context features with the BERT flow representation.",
    ),
    semantic_codebook_path: Optional[Path] = typer.Option(
        None,
        help="Optional SCE codebook JSON. When set, byte chunks are encoded as SCE tokens with byte fallback.",
    ),
) -> None:
    """Train the hierarchical classifier."""

    set_seed(seed)
    label_map = LabelMap.from_yaml(label_map_path)
    semantic_codebook = _load_semantic_codebook(semantic_codebook_path)
    train_dataset = FlowWindowDataset.from_parquet(
        train_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
        use_connection_tokens=use_connection_tokens,
        use_context_tokens=use_context_tokens,
        use_context_features=use_context_features,
        semantic_codebook=semantic_codebook,
    )
    major_class_weights = _compute_major_class_weights(
        train_dataset.frame,
        label_map,
        major_class_weighting,
        major_class_weight_cap,
    )
    device_obj = resolve_device(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device_obj),
    )

    model = _make_classifier(
        label_map,
        max_position_embeddings=max_length,
        major_class_weights=major_class_weights,
        vocab_size=train_dataset.tokenizer.vocab_size,
        context_feature_size=len(CONTEXT_FEATURE_COLUMNS) if use_context_features else 0,
    )
    if init_bert_checkpoint is not None:
        _load_bert_encoder_from_mlm(model, init_bert_checkpoint)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"])
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    best_macro_f1 = -1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "class_weights.json",
        {
            "major_class_weighting": major_class_weighting,
            "major_class_weight_cap": major_class_weight_cap,
            "major_labels": label_map.major_labels,
            "major_class_weights": major_class_weights,
        },
    )
    history_dir = history_dir or (output_dir / "history")
    step_writer = StepHistoryWriter(history_dir)
    global_step = 0
    for epoch in range(epochs):
        learning_rate_value = optimizer.param_groups[0]["lr"]
        metrics = train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            device_obj,
            epoch=epoch + 1,
            global_step_start=global_step,
            learning_rate=learning_rate_value,
            step_writer=step_writer,
        )
        global_step = int(metrics.get("global_step", global_step))
        metrics["epoch"] = epoch + 1
        metrics["learning_rate"] = learning_rate_value
        eval_metrics = None
        if val_path is not None:
            eval_metrics = _evaluate_classifier_model(
                model=model,
                data_path=val_path,
                label_map=label_map,
                view=view,
                batch_size=batch_size,
                max_length=max_length,
                stride=stride,
                max_windows=max_windows,
                device=device_obj,
                num_workers=num_workers,
                output_dir=output_dir / "eval",
                use_connection_tokens=use_connection_tokens,
                use_context_tokens=use_context_tokens,
                use_context_features=use_context_features,
                semantic_codebook=semantic_codebook,
            )
            metrics.update(
                {
                    "val_loss": eval_metrics["eval_loss"],
                    "val_accuracy": eval_metrics["accuracy"],
                    "val_macro_precision": eval_metrics["macro_precision"],
                    "val_macro_recall": eval_metrics["macro_recall"],
                    "val_macro_f1": eval_metrics["macro_f1"],
                    "val_weighted_f1": eval_metrics["weighted_f1"],
                }
            )
            if eval_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = eval_metrics["macro_f1"]
                save_checkpoint(
                    output_dir / "classifier.best.pt",
                    model,
                    extra={
                        "label_map": label_map.as_dict(),
                        "epoch": epoch + 1,
                        "eval_metrics": eval_metrics,
                        "model_config": model.bert.config.to_dict(),
                        "major_class_weights": major_class_weights,
                        "use_connection_tokens": use_connection_tokens,
                        "use_context_tokens": use_context_tokens,
                        "use_context_features": use_context_features,
                        "context_feature_names": list(CONTEXT_FEATURE_COLUMNS),
                    "context_feature_size": len(CONTEXT_FEATURE_COLUMNS)
                    if use_context_features
                    else 0,
                    "tokenizer_vocab_size": train_dataset.tokenizer.vocab_size,
                    **_semantic_codebook_checkpoint_metadata(
                        semantic_codebook,
                        semantic_codebook_path,
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
        history.append(metrics)
        if log_path is not None:
            append_jsonl(log_path, metrics)
        write_history_files(history_dir, history)
        if save_each_epoch:
            save_epoch_checkpoint(
                output_dir,
                epoch + 1,
                model,
                optimizer,
                metrics,
                history,
                extra={
                    "label_map": label_map.as_dict(),
                    "eval_metrics": eval_metrics,
                    "model_config": model.bert.config.to_dict(),
                    "major_class_weights": major_class_weights,
                    "use_connection_tokens": use_connection_tokens,
                    "use_context_tokens": use_context_tokens,
                    "use_context_features": use_context_features,
                    "context_feature_names": list(CONTEXT_FEATURE_COLUMNS),
                    "context_feature_size": len(CONTEXT_FEATURE_COLUMNS)
                    if use_context_features
                    else 0,
                    "tokenizer_vocab_size": train_dataset.tokenizer.vocab_size,
                    **_semantic_codebook_checkpoint_metadata(
                        semantic_codebook,
                        semantic_codebook_path,
                    ),
                },
            )
        typer.echo(json.dumps(metrics, indent=2))

    eval_metrics = None
    if val_path is not None:
        eval_metrics = _evaluate_classifier_model(
            model=model,
            data_path=val_path,
            label_map=label_map,
            view=view,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            max_windows=max_windows,
            device=device_obj,
            num_workers=num_workers,
            output_dir=output_dir / "eval",
            use_connection_tokens=use_connection_tokens,
            use_context_tokens=use_context_tokens,
            use_context_features=use_context_features,
            semantic_codebook=semantic_codebook,
        )
        typer.echo(json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    save_checkpoint(
        output_dir / "classifier.pt",
        model,
        extra={
            "label_map": label_map.as_dict(),
            "history": history,
            "eval_metrics": eval_metrics,
            "model_config": model.bert.config.to_dict(),
            "major_class_weights": major_class_weights,
            "use_connection_tokens": use_connection_tokens,
            "use_context_tokens": use_context_tokens,
            "use_context_features": use_context_features,
            "context_feature_names": list(CONTEXT_FEATURE_COLUMNS),
            "context_feature_size": len(CONTEXT_FEATURE_COLUMNS) if use_context_features else 0,
            "tokenizer_vocab_size": train_dataset.tokenizer.vocab_size,
            **_semantic_codebook_checkpoint_metadata(
                semantic_codebook,
                semantic_codebook_path,
            ),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )


@train_app.command("classifier-config")
def train_classifier_config(config: Path = typer.Option(..., help="Classifier train YAML.")) -> None:
    """Train the classifier from a YAML config file."""

    raw = load_yaml(config)
    train_cfg = raw.get("train", {})
    data_cfg = raw.get("data", {})
    train_classifier(
        train_path=Path(data_cfg["train_path"]),
        output_dir=Path(train_cfg.get("output_dir", "artifacts/classifier")),
        label_map_path=Path(data_cfg.get("label_map", "configs/label_map.yaml")),
        val_path=Path(data_cfg["val_path"]) if data_cfg.get("val_path") else None,
        view=data_cfg.get("view", "masked_header_packet"),
        epochs=int(train_cfg.get("epochs", 3)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-5)),
        max_length=int(data_cfg.get("max_length", 512)),
        stride=int(data_cfg.get("stride", 384)),
        max_windows=data_cfg.get("max_windows", 8),
        num_workers=int(data_cfg.get("num_workers", train_cfg.get("num_workers", 0))),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 42)),
        log_path=Path(train_cfg["log_path"]) if train_cfg.get("log_path") else None,
        history_dir=Path(train_cfg["history_dir"]) if train_cfg.get("history_dir") else None,
        save_each_epoch=bool(train_cfg.get("save_each_epoch", True)),
        resume_checkpoint=Path(train_cfg["resume_checkpoint"])
        if train_cfg.get("resume_checkpoint")
        else None,
        init_bert_checkpoint=Path(train_cfg["init_bert_checkpoint"])
        if train_cfg.get("init_bert_checkpoint")
        else None,
        major_class_weighting=train_cfg.get("major_class_weighting", "none"),
        major_class_weight_cap=float(train_cfg.get("major_class_weight_cap", 20.0)),
        use_connection_tokens=bool(data_cfg.get("use_connection_tokens", False)),
        use_context_tokens=bool(data_cfg.get("use_context_tokens", False)),
        use_context_features=bool(data_cfg.get("use_context_features", False)),
        semantic_codebook_path=Path(data_cfg["semantic_codebook_path"])
        if data_cfg.get("semantic_codebook_path")
        else None,
    )


@train_app.command("mlm")
def train_mlm(
    train_path: Path = typer.Option(..., help="Training Parquet file."),
    output_dir: Path = typer.Option(Path("artifacts/mlm"), help="Output directory."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("masked_header_packet", help="Input view to train on."),
    epochs: int = typer.Option(1, help="Number of epochs."),
    batch_size: int = typer.Option(4, help="Batch size."),
    learning_rate: float = typer.Option(5e-5, help="Learning rate."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(8, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    seed: int = typer.Option(42, help="Random seed."),
    log_path: Optional[Path] = typer.Option(None, help="Optional JSONL training log."),
    history_dir: Optional[Path] = typer.Option(None, help="Directory for JSONL/CSV history."),
    save_each_epoch: bool = typer.Option(True, help="Save one checkpoint per epoch."),
    resume_checkpoint: Optional[Path] = typer.Option(None, help="Resume MLM checkpoint."),
) -> None:
    """Run MLM pretraining on processed flow data."""

    set_seed(seed)
    label_map = LabelMap.from_yaml(label_map_path)
    tokenizer = ByteTokenizer()
    dataset = FlowWindowDataset.from_parquet(
        train_path,
        label_map=label_map,
        view=view,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )
    device_obj = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device_obj),
    )
    model = create_mlm_model(max_position_embeddings=max_length)
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"])
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    special_token_ids = {
        tokenizer.token_to_id[token]
        for token in [
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            MASK_TOKEN,
            PKT_FWD_TOKEN,
            PKT_BWD_TOKEN,
            PKT_END_TOKEN,
        ]
    }

    history = []
    best_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = history_dir or (output_dir / "history")
    step_writer = StepHistoryWriter(history_dir)
    global_step = 0
    for epoch in range(epochs):
        learning_rate_value = optimizer.param_groups[0]["lr"]
        metrics = train_mlm_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            device=device_obj,
            special_token_ids=special_token_ids,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=tokenizer.vocab_size,
            epoch=epoch + 1,
            global_step_start=global_step,
            learning_rate=learning_rate_value,
            step_writer=step_writer,
        )
        global_step = int(metrics.get("global_step", global_step))
        metrics["epoch"] = epoch + 1
        metrics["learning_rate"] = learning_rate_value
        history.append(metrics)
        if log_path is not None:
            append_jsonl(log_path, metrics)
        write_history_files(history_dir, history)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(
                output_dir / "mlm.best.pt",
                model,
                extra={
                    "history": history,
                    "epoch": epoch + 1,
                    "model_config": model.config.to_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
        if save_each_epoch:
            save_epoch_checkpoint(
                output_dir,
                epoch + 1,
                model,
                optimizer,
                metrics,
                history,
                extra={"model_config": model.config.to_dict()},
            )
        typer.echo(json.dumps(metrics, indent=2))

    save_checkpoint(
        output_dir / "mlm.pt",
        model,
        extra={
            "history": history,
            "model_config": model.config.to_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
    )


@train_app.command("mlm-config")
def train_mlm_config(config: Path = typer.Option(..., help="MLM train YAML.")) -> None:
    """Train MLM from a YAML config file."""

    raw = load_yaml(config)
    train_cfg = raw.get("train", {})
    data_cfg = raw.get("data", {})
    train_mlm(
        train_path=Path(data_cfg["train_path"]),
        output_dir=Path(train_cfg.get("output_dir", "artifacts/mlm")),
        label_map_path=Path(data_cfg.get("label_map", "configs/label_map.yaml")),
        view=data_cfg.get("view", "masked_header_packet"),
        epochs=int(train_cfg.get("epochs", 1)),
        batch_size=int(train_cfg.get("batch_size", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        max_length=int(data_cfg.get("max_length", 512)),
        stride=int(data_cfg.get("stride", 384)),
        max_windows=data_cfg.get("max_windows", 8),
        num_workers=int(data_cfg.get("num_workers", train_cfg.get("num_workers", 0))),
        device=raw.get("device", "auto"),
        seed=int(raw.get("seed", 42)),
        log_path=Path(train_cfg["log_path"]) if train_cfg.get("log_path") else None,
        history_dir=Path(train_cfg["history_dir"]) if train_cfg.get("history_dir") else None,
        save_each_epoch=bool(train_cfg.get("save_each_epoch", True)),
        resume_checkpoint=Path(train_cfg["resume_checkpoint"])
        if train_cfg.get("resume_checkpoint")
        else None,
    )


@train_app.command("neural-baseline")
def train_neural_baseline(
    train_path: Path = typer.Option(..., help="Training Parquet file."),
    output_dir: Path = typer.Option(Path("artifacts/neural_baseline"), help="Output directory."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    val_path: Optional[Path] = typer.Option(None, help="Optional validation Parquet file."),
    view: str = typer.Option("payload_only", help="Input view."),
    model_name: str = typer.Option("cnn", "--model", help="cnn, gru, or transformer."),
    epochs: int = typer.Option(3, help="Number of epochs."),
    batch_size: int = typer.Option(8, help="Batch size."),
    learning_rate: float = typer.Option(1e-3, help="Learning rate."),
    hidden_size: int = typer.Option(128, help="Embedding/hidden size."),
    max_length: int = typer.Option(512, help="BERT-style max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(4, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    seed: int = typer.Option(42, help="Random seed."),
    log_path: Optional[Path] = typer.Option(None, help="Optional JSONL training log."),
    history_dir: Optional[Path] = typer.Option(None, help="Directory for JSONL/CSV history."),
    save_each_epoch: bool = typer.Option(True, help="Save one checkpoint per epoch."),
) -> None:
    """Train a neural major-label baseline."""

    set_seed(seed)
    tokenizer = ByteTokenizer()
    label_map = LabelMap.from_yaml(label_map_path)
    dataset = FlowWindowDataset.from_parquet(
        train_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )
    device_obj = resolve_device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device_obj),
    )
    model = _make_neural_baseline(
        model_name=model_name,
        vocab_size=tokenizer.vocab_size,
        num_labels=len(label_map.major_labels),
        hidden_size=hidden_size,
        max_length=max_length,
    )
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir = history_dir or (output_dir / "history")
    history = []
    best_macro_f1 = -1.0
    step_writer = StepHistoryWriter(history_dir)
    global_step = 0
    for epoch in range(epochs):
        learning_rate_value = optimizer.param_groups[0]["lr"]
        metrics = _train_neural_baseline_epoch(
            model,
            loader,
            optimizer,
            device_obj,
            epoch=epoch + 1,
            global_step_start=global_step,
            learning_rate=learning_rate_value,
            step_writer=step_writer,
        )
        global_step = int(metrics.get("global_step", global_step))
        metrics["epoch"] = epoch + 1
        metrics["learning_rate"] = learning_rate_value
        eval_metrics = None
        if val_path is not None:
            eval_metrics = _eval_neural_baseline(
                model=model,
                data_path=val_path,
                label_map=label_map,
                view=view,
                batch_size=batch_size,
                max_length=max_length,
                stride=stride,
                max_windows=max_windows,
                device=device_obj,
                num_workers=num_workers,
                output_dir=output_dir / "eval",
            )
            metrics.update(
                {
                    "val_loss": eval_metrics["eval_loss"],
                    "val_accuracy": eval_metrics["accuracy"],
                    "val_macro_precision": eval_metrics["macro_precision"],
                    "val_macro_recall": eval_metrics["macro_recall"],
                    "val_macro_f1": eval_metrics["macro_f1"],
                    "val_weighted_f1": eval_metrics["weighted_f1"],
                }
            )
            if eval_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = eval_metrics["macro_f1"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "hidden_size": hidden_size,
                        "max_length": max_length,
                        "label_map": label_map.as_dict(),
                        "eval_metrics": eval_metrics,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    output_dir / "neural_baseline.best.pt",
                )
        history.append(metrics)
        if log_path is not None:
            append_jsonl(log_path, metrics)
        write_history_files(history_dir, history)
        if save_each_epoch:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt",
                model,
                extra={
                    "epoch": epoch + 1,
                    "metrics": metrics,
                    "history": history,
                    "model_name": model_name,
                    "hidden_size": hidden_size,
                    "max_length": max_length,
                    "label_map": label_map.as_dict(),
                    "eval_metrics": eval_metrics,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
            )
        typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "hidden_size": hidden_size,
            "max_length": max_length,
            "history": history,
            "label_map": label_map.as_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output_dir / "neural_baseline.pt",
    )


@eval_app.command("classifier")
def eval_classifier(
    data_path: Path = typer.Option(..., help="Evaluation Parquet file."),
    checkpoint: Path = typer.Option(..., help="Classifier checkpoint."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("masked_header_packet", help="Input view."),
    batch_size: int = typer.Option(4, help="Batch size."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(8, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    output_dir: Optional[Path] = typer.Option(None, help="Optional metrics output directory."),
    output_predictions_path: Optional[Path] = typer.Option(
        None,
        help="Optional CSV/Parquet path for per-flow major-label probabilities.",
    ),
    use_connection_tokens: bool = typer.Option(
        False,
        "--use-connection-tokens/--no-connection-tokens",
        help="Prefix each flow with a connection-type token from processed metadata.",
    ),
    use_context_tokens: bool = typer.Option(
        False,
        "--use-context-tokens/--no-context-tokens",
        help="Prefix each flow with host-window context tokens when context columns exist.",
    ),
    use_context_features: bool = typer.Option(
        False,
        "--use-context-features/--no-context-features",
        help="Concatenate log-scaled host-window context features with the BERT flow representation.",
    ),
    semantic_codebook_path: Optional[Path] = typer.Option(
        None,
        help="Optional SCE codebook JSON. Defaults to checkpoint-embedded codebook when available.",
    ),
) -> None:
    """Evaluate a classifier checkpoint."""

    label_map = LabelMap.from_yaml(label_map_path)
    model = _load_classifier(checkpoint, label_map, max_length)
    semantic_codebook = _resolve_semantic_codebook(semantic_codebook_path, checkpoint=checkpoint)
    device_obj = resolve_device(device)
    model.to(device_obj)
    metrics = _evaluate_classifier_model(
        model=model,
        data_path=data_path,
        label_map=label_map,
        view=view,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
        device=device_obj,
        num_workers=num_workers,
        output_dir=output_dir,
        output_predictions_path=output_predictions_path,
        use_connection_tokens=use_connection_tokens,
        use_context_tokens=use_context_tokens,
        use_context_features=use_context_features,
        semantic_codebook=semantic_codebook,
    )
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))


@eval_app.command("neural-baseline")
def eval_neural_baseline(
    data_path: Path = typer.Option(..., help="Evaluation Parquet file."),
    checkpoint: Path = typer.Option(..., help="Neural baseline checkpoint."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("payload_only", help="Input view."),
    batch_size: int = typer.Option(8, help="Batch size."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(4, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    output_dir: Optional[Path] = typer.Option(None, help="Optional metrics output directory."),
) -> None:
    """Evaluate a neural baseline checkpoint."""

    label_map = LabelMap.from_yaml(label_map_path)
    payload = torch.load(checkpoint, map_location="cpu")
    model = _make_neural_baseline(
        model_name=payload["model_name"],
        vocab_size=ByteTokenizer().vocab_size,
        num_labels=len(label_map.major_labels),
        hidden_size=int(payload["hidden_size"]),
        max_length=int(payload["max_length"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    device_obj = resolve_device(device)
    model.to(device_obj)
    metrics = _eval_neural_baseline(
        model=model,
        data_path=data_path,
        label_map=label_map,
        view=view,
        batch_size=batch_size,
        max_length=int(payload["max_length"]),
        stride=stride,
        max_windows=max_windows,
        device=device_obj,
        num_workers=num_workers,
        output_dir=output_dir,
    )
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))


@eval_app.command("calibrate-thresholds")
def calibrate_classifier_thresholds(
    data_path: Path = typer.Option(..., help="Validation Parquet file."),
    checkpoint: Path = typer.Option(..., help="Classifier checkpoint."),
    output_path: Path = typer.Option(
        Path("artifacts/classifier/minor_thresholds.json"),
        help="Output JSON threshold path.",
    ),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("masked_header_packet", help="Input view."),
    batch_size: int = typer.Option(4, help="Batch size."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(8, help="Max windows per flow."),
    num_workers: int = typer.Option(0, help="DataLoader worker processes."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    use_connection_tokens: bool = typer.Option(
        False,
        "--use-connection-tokens/--no-connection-tokens",
        help="Prefix each flow with a connection-type token from processed metadata.",
    ),
    use_context_tokens: bool = typer.Option(
        False,
        "--use-context-tokens/--no-context-tokens",
        help="Prefix each flow with host-window context tokens when context columns exist.",
    ),
    use_context_features: bool = typer.Option(
        False,
        "--use-context-features/--no-context-features",
        help="Concatenate log-scaled host-window context features with the BERT flow representation.",
    ),
    semantic_codebook_path: Optional[Path] = typer.Option(
        None,
        help="Optional SCE codebook JSON. Defaults to checkpoint-embedded codebook when available.",
    ),
) -> None:
    """Calibrate per-minor-label thresholds on validation data."""

    label_map = LabelMap.from_yaml(label_map_path)
    device_obj = resolve_device(device)
    semantic_codebook = _resolve_semantic_codebook(semantic_codebook_path, checkpoint=checkpoint)
    dataset = FlowWindowDataset.from_parquet(
        data_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
        use_connection_tokens=use_connection_tokens,
        use_context_tokens=use_context_tokens,
        use_context_features=use_context_features,
        semantic_codebook=semantic_codebook,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=flow_collate,
        **_loader_kwargs(num_workers, device_obj),
    )
    model = _load_classifier(checkpoint, label_map, max_length)
    model.to(device_obj)
    outputs = collect_classifier_outputs(model, loader, device_obj)
    thresholds, results = calibrate_thresholds(
        y_true=outputs["minor_labels"].numpy(),
        y_prob=torch.sigmoid(outputs["minor_logits"]).numpy(),
        labels=label_map.minor_labels,
    )
    payload = {
        "thresholds": thresholds,
        "results": [result.__dict__ for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@predict_app.command("hex")
def predict_hex(
    payload_hex: str = typer.Argument(..., help="Hex payload string."),
    checkpoint: Path = typer.Option(..., help="Classifier checkpoint."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    thresholds: Optional[Path] = typer.Option(None, help="Minor threshold JSON file."),
    output: Optional[Path] = typer.Option(None, help="Optional prediction JSON path."),
    semantic_codebook_path: Optional[Path] = typer.Option(
        None,
        help="Optional SCE codebook JSON. Defaults to checkpoint-embedded codebook when available.",
    ),
) -> None:
    """Predict one hex payload as a single forward packet."""

    label_map = LabelMap.from_yaml(label_map_path)
    semantic_codebook = _resolve_semantic_codebook(semantic_codebook_path, checkpoint=checkpoint)
    tokenizer = ByteTokenizer(
        extra_tokens=semantic_codebook.tokens if semantic_codebook is not None else None
    )
    byte_token_encoder = None
    if semantic_codebook is not None:
        def encode_with_codebook(data: bytes) -> list[str]:
            return semantic_codebook.encode_bytes(data, tokenizer=tokenizer)

        byte_token_encoder = encode_with_codebook
    data = tokenizer.parse_hex(payload_hex)
    windows = tokenizer.encode_flow(
        [PacketChunk(direction="fwd", data=data)],
        max_length=max_length,
        stride=stride,
        padding=True,
        byte_token_encoder=byte_token_encoder,
    )
    input_ids = torch.tensor([[item.input_ids for item in windows]], dtype=torch.long)
    attention_mask = torch.tensor([[item.attention_mask for item in windows]], dtype=torch.long)
    window_mask = torch.ones(1, len(windows), dtype=torch.bool)

    model = _load_classifier(checkpoint, label_map, max_length)
    device_obj = resolve_device(device)
    model.to(device_obj)
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device_obj),
            attention_mask=attention_mask.to(device_obj),
            window_mask=window_mask.to(device_obj),
        )
    prediction = decode_hierarchical_prediction(
        outputs["major_logits"][0].cpu(),
        outputs["minor_logits"][0].cpu(),
        label_map,
        thresholds=_load_thresholds(thresholds),
    )
    payload = prediction.as_dict()
    if output is not None:
        write_json(output, payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@predict_app.command("pcap")
def predict_pcap(
    pcap_path: Path = typer.Argument(..., help="PCAP/PCAPNG file."),
    checkpoint: Path = typer.Option(..., help="Classifier checkpoint."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("masked_header_packet", help="Input view."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    max_windows: Optional[int] = typer.Option(2, help="Max windows per flow."),
    max_packets_per_flow: Optional[int] = typer.Option(None, help="Optional packet cap per flow."),
    flow_timeout_seconds: Optional[float] = typer.Option(
        120.0,
        help="Idle timeout in seconds before a repeated five-tuple starts a new flow session.",
    ),
    close_on_tcp_flags: bool = typer.Option(
        False,
        help="Split TCP sessions immediately on FIN/RST packets.",
    ),
    device: str = typer.Option("cuda", help="cuda, cpu, or auto."),
    thresholds: Optional[Path] = typer.Option(None, help="Minor threshold JSON file."),
    output: Optional[Path] = typer.Option(None, help="Optional JSONL output path."),
    summary_output: Optional[Path] = typer.Option(None, help="Optional JSON summary output path."),
    risk_threshold: float = typer.Option(0.5, help="Major probability threshold for high-risk flow summary."),
    top_flows: int = typer.Option(20, help="Maximum high-risk flows included in the summary."),
    host_window_policy_path: Optional[Path] = typer.Option(
        None,
        help="Optional host-window analysis/policy JSON for calibrated window alerts.",
    ),
    semantic_codebook_path: Optional[Path] = typer.Option(
        None,
        help="Optional SCE codebook JSON. Defaults to checkpoint-embedded codebook when available.",
    ),
) -> None:
    """Predict all reconstructed TCP/UDP flows from a PCAP file."""

    label_map = LabelMap.from_yaml(label_map_path)
    semantic_codebook = _resolve_semantic_codebook(semantic_codebook_path, checkpoint=checkpoint)
    tokenizer = ByteTokenizer(
        extra_tokens=semantic_codebook.tokens if semantic_codebook is not None else None
    )
    byte_token_encoder = None
    if semantic_codebook is not None:
        def encode_with_codebook(data: bytes) -> list[str]:
            return semantic_codebook.encode_bytes(data, tokenizer=tokenizer)

        byte_token_encoder = encode_with_codebook
    input_view = InputView(view)
    flows = PcapFlowExtractor(
        max_packets_per_flow=max_packets_per_flow,
        flow_timeout_seconds=flow_timeout_seconds,
        close_on_tcp_flags=close_on_tcp_flags,
    ).extract(pcap_path)
    model = _load_classifier(checkpoint, label_map, max_length)
    device_obj = resolve_device(device)
    model.to(device_obj)
    model.eval()
    loaded_thresholds = _load_thresholds(thresholds)
    host_window_policies = _load_host_window_policies(host_window_policy_path)

    predictions = []
    output_handle = None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = open(output, "w", encoding="utf-8")
    for flow in flows:
        chunks = [
            PacketChunk(direction=packet.direction, data=packet.bytes_for_view(input_view))
            for packet in flow.packets
        ]
        windows = tokenizer.encode_flow(
            chunks,
            max_length=max_length,
            stride=stride,
            padding=True,
            byte_token_encoder=byte_token_encoder,
        )
        if max_windows is not None:
            windows = windows[:max_windows]
        input_ids = torch.tensor([[item.input_ids for item in windows]], dtype=torch.long)
        attention_mask = torch.tensor([[item.attention_mask for item in windows]], dtype=torch.long)
        window_mask = torch.ones(1, len(windows), dtype=torch.bool)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device_obj),
                attention_mask=attention_mask.to(device_obj),
                window_mask=window_mask.to(device_obj),
            )
        prediction = decode_hierarchical_prediction(
            outputs["major_logits"][0].cpu(),
            outputs["minor_logits"][0].cpu(),
            label_map,
            thresholds=loaded_thresholds,
        )
        item = prediction.as_dict()
        item.update(
            {
                "flow_id": flow.flow_id,
                "source_file": flow.source_file,
                "protocol": flow.protocol,
                "connection_type": flow.connection_type,
                "endpoint_a": flow.endpoint_a,
                "endpoint_b": flow.endpoint_b,
                "initiator_endpoint": flow.initiator_endpoint,
                "responder_endpoint": flow.responder_endpoint,
                "start_time": flow.start_time,
                "end_time": flow.end_time,
                "packet_count": flow.packet_count,
                "observed_packet_count": flow.observed_packet_count,
                "was_packet_truncated": flow.was_packet_truncated,
                "payload_byte_length": flow.payload_byte_length,
                "packet_byte_length": flow.packet_byte_length,
            }
        )
        predictions.append(item)
        if output_handle is not None:
            output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if output_handle is not None:
        output_handle.close()
    payload = {
        "summary": _summarize_pcap_predictions(
            predictions,
            risk_threshold=risk_threshold,
            top_flows=top_flows,
            host_window_policies=host_window_policies,
        ),
        "flows": predictions,
    }
    if summary_output is not None:
        write_json(summary_output, payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@baseline_app.command("train")
def train_baseline(
    train_path: Path = typer.Option(..., help="Training Parquet file."),
    output_path: Path = typer.Option(
        Path("artifacts/baselines/tfidf_logreg.pkl"),
        help="Output pickle path.",
    ),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    val_path: Optional[Path] = typer.Option(None, help="Optional validation Parquet file."),
    view: str = typer.Option("payload_only", help="Input view."),
    model: str = typer.Option("logreg", help="logreg or linear_svm."),
    ngram_min: int = typer.Option(1, help="Minimum byte n-gram."),
    ngram_max: int = typer.Option(4, help="Maximum byte n-gram."),
    max_features: int = typer.Option(200_000, help="TF-IDF max features."),
) -> None:
    """Train a TF-IDF byte n-gram baseline for major-label classification."""

    label_map = LabelMap.from_yaml(label_map_path)
    train_frame = pd.read_parquet(train_path)
    train_frame = train_frame[train_frame["view"] == view]
    pipeline = train_major_baseline(
        train_frame,
        label_map=label_map,
        model=model,
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
    )
    save_baseline(output_path, pipeline)
    result = {"saved_to": str(output_path), "train_rows": int(len(train_frame))}
    if val_path is not None:
        val_frame = pd.read_parquet(val_path)
        val_frame = val_frame[val_frame["view"] == view]
        result["eval"] = evaluate_major_baseline(pipeline, val_frame, label_map).metrics
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@baseline_app.command("eval")
def eval_baseline(
    data_path: Path = typer.Option(..., help="Evaluation Parquet file."),
    checkpoint: Path = typer.Option(..., help="Baseline pickle file."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    view: str = typer.Option("payload_only", help="Input view."),
) -> None:
    """Evaluate a saved TF-IDF byte n-gram baseline."""

    label_map = LabelMap.from_yaml(label_map_path)
    frame = pd.read_parquet(data_path)
    frame = frame[frame["view"] == view]
    pipeline = load_baseline(checkpoint)
    result = evaluate_major_baseline(pipeline, frame, label_map)
    typer.echo(json.dumps(result.metrics, ensure_ascii=False, indent=2))
