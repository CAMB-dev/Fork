"""Command line entrypoint for the Traffic Byte-BERT project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
import typer

from traffic_bert import __version__
from traffic_bert.data.build import BuildConfig, build_processed_dataset
from traffic_bert.data.dataset import FlowWindowDataset, flow_collate
from traffic_bert.data.schema import InputView
from traffic_bert.inference import decode_hierarchical_prediction
from traffic_bert.labels import LabelMap
from traffic_bert.metrics import major_classification_metrics, multilabel_f1
from traffic_bert.models import (
    ByteBertForHierarchicalClassification,
    create_bert_config,
    create_mlm_model,
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
    save_checkpoint,
    train_classifier_epoch,
    train_mlm_epoch,
)

app = typer.Typer(help="Traffic Byte-BERT experiment toolkit.")
data_app = typer.Typer(help="Data preprocessing commands.")
vocab_app = typer.Typer(help="Vocabulary utilities.")
train_app = typer.Typer(help="Training commands.")
eval_app = typer.Typer(help="Evaluation commands.")
predict_app = typer.Typer(help="Prediction commands.")

app.add_typer(data_app, name="data")
app.add_typer(vocab_app, name="vocab")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(predict_app, name="predict")


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


@data_app.command("build")
def build_data(
    input_path: Path = typer.Option(..., help="PCAP file or directory."),
    output_path: Path = typer.Option(..., help="Output Parquet path."),
    label_map: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    source_dataset: str = typer.Option("custom", help="Dataset name stored in rows."),
    split: str = typer.Option("train", help="Split name stored in rows."),
    label_source: str = typer.Option(
        "filename",
        help="filename, parent, parent_filename, or static.",
    ),
    static_label: Optional[str] = typer.Option(None, help="Label used with label_source=static."),
    views: str = typer.Option(
        "payload_only,full_packet,masked_header_packet",
        help="Comma-separated input views.",
    ),
    keep_empty_payload: bool = typer.Option(True, help="Keep flows without L4 payload."),
    max_packets_per_flow: Optional[int] = typer.Option(None, help="Optional packet cap per flow."),
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
            views=selected_views,
            keep_empty_payload=keep_empty_payload,
            max_packets_per_flow=max_packets_per_flow,
        )
    )
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


def _make_classifier(
    label_map: LabelMap,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    intermediate_size: int = 1024,
    max_position_embeddings: int = 512,
) -> ByteBertForHierarchicalClassification:
    config = create_bert_config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    return ByteBertForHierarchicalClassification(
        bert_config=config,
        num_major_labels=len(label_map.major_labels),
        num_minor_labels=len(label_map.minor_labels),
        minor_to_major=label_map.minor_major_ids(),
    )


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
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
) -> None:
    """Train the hierarchical classifier."""

    label_map = LabelMap.from_yaml(label_map_path)
    train_dataset = FlowWindowDataset.from_parquet(
        train_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=flow_collate,
    )

    model = _make_classifier(label_map, max_position_embeddings=max_length)
    device_obj = resolve_device(device)
    model.to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(epochs):
        metrics = train_classifier_epoch(model, train_loader, optimizer, device_obj)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        typer.echo(json.dumps(metrics, indent=2))

    eval_metrics = None
    if val_path is not None:
        val_dataset = FlowWindowDataset.from_parquet(
            val_path,
            label_map=label_map,
            view=view,
            max_length=max_length,
            stride=stride,
            max_windows=max_windows,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=flow_collate,
        )
        outputs = collect_classifier_outputs(model, val_loader, device_obj)
        y_true = outputs["major_labels"].numpy()
        y_pred = outputs["major_logits"].argmax(dim=-1).numpy()
        eval_metrics = major_classification_metrics(y_true, y_pred, label_map.major_labels)
        eval_metrics["minor"] = multilabel_f1(
            outputs["minor_labels"].numpy(),
            torch.sigmoid(outputs["minor_logits"]).numpy(),
        )
        typer.echo(json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "classifier.pt",
        model,
        extra={
            "label_map": label_map.as_dict(),
            "history": history,
            "eval_metrics": eval_metrics,
            "model_config": model.bert.config.to_dict(),
        },
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
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
) -> None:
    """Run MLM pretraining on processed flow data."""

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=flow_collate)
    model = create_mlm_model(max_position_embeddings=max_length)
    device_obj = resolve_device(device)
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
    for epoch in range(epochs):
        metrics = train_mlm_epoch(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            device=device_obj,
            special_token_ids=special_token_ids,
            mask_token_id=tokenizer.mask_token_id,
            vocab_size=tokenizer.vocab_size,
        )
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        typer.echo(json.dumps(metrics, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        output_dir / "mlm.pt",
        model,
        extra={"history": history, "model_config": model.config.to_dict()},
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
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
) -> None:
    """Evaluate a classifier checkpoint."""

    label_map = LabelMap.from_yaml(label_map_path)
    dataset = FlowWindowDataset.from_parquet(
        data_path,
        label_map=label_map,
        view=view,
        max_length=max_length,
        stride=stride,
        max_windows=max_windows,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=flow_collate)
    payload = torch.load(checkpoint, map_location="cpu")
    model_config = payload.get("model_config", {})
    model = _make_classifier(
        label_map,
        hidden_size=model_config.get("hidden_size", 256),
        num_layers=model_config.get("num_hidden_layers", 4),
        num_heads=model_config.get("num_attention_heads", 4),
        intermediate_size=model_config.get("intermediate_size", 1024),
        max_position_embeddings=model_config.get("max_position_embeddings", max_length),
    )
    model.load_state_dict(payload["model_state_dict"])
    device_obj = resolve_device(device)
    model.to(device_obj)
    outputs = collect_classifier_outputs(model, loader, device_obj)
    y_true = outputs["major_labels"].numpy()
    y_pred = outputs["major_logits"].argmax(dim=-1).numpy()
    metrics = major_classification_metrics(y_true, y_pred, label_map.major_labels)
    metrics["minor"] = multilabel_f1(
        outputs["minor_labels"].numpy(),
        torch.sigmoid(outputs["minor_logits"]).numpy(),
    )
    typer.echo(json.dumps(metrics, ensure_ascii=False, indent=2))


@predict_app.command("hex")
def predict_hex(
    payload_hex: str = typer.Argument(..., help="Hex payload string."),
    checkpoint: Path = typer.Option(..., help="Classifier checkpoint."),
    label_map_path: Path = typer.Option(Path("configs/label_map.yaml"), help="Label map YAML."),
    max_length: int = typer.Option(512, help="BERT max sequence length."),
    stride: int = typer.Option(384, help="Sliding window stride."),
    device: str = typer.Option("auto", help="auto, cpu, or cuda."),
) -> None:
    """Predict one hex payload as a single forward packet."""

    label_map = LabelMap.from_yaml(label_map_path)
    tokenizer = ByteTokenizer()
    data = tokenizer.parse_hex(payload_hex)
    windows = tokenizer.encode_flow(
        [PacketChunk(direction="fwd", data=data)],
        max_length=max_length,
        stride=stride,
        padding=True,
    )
    input_ids = torch.tensor([[item.input_ids for item in windows]], dtype=torch.long)
    attention_mask = torch.tensor([[item.attention_mask for item in windows]], dtype=torch.long)
    window_mask = torch.ones(1, len(windows), dtype=torch.bool)

    payload = torch.load(checkpoint, map_location="cpu")
    model_config = payload.get("model_config", {})
    model = _make_classifier(
        label_map,
        hidden_size=model_config.get("hidden_size", 256),
        num_layers=model_config.get("num_hidden_layers", 4),
        num_heads=model_config.get("num_attention_heads", 4),
        intermediate_size=model_config.get("intermediate_size", 1024),
        max_position_embeddings=model_config.get("max_position_embeddings", max_length),
    )
    model.load_state_dict(payload["model_state_dict"])
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
    )
    typer.echo(json.dumps(prediction.as_dict(), ensure_ascii=False, indent=2))
