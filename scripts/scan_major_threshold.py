from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from traffic_bert.cli import _load_classifier, _loader_kwargs, _resolve_semantic_codebook
from traffic_bert.config import write_json
from traffic_bert.data.dataset import FlowWindowDataset, flow_collate
from traffic_bert.labels import LabelMap
from traffic_bert.metrics import attack_detection_metrics, major_classification_metrics
from traffic_bert.training import collect_classifier_outputs


def _collect_predictions(
    *,
    data_path: Path,
    checkpoint: Path,
    label_map: LabelMap,
    view: str,
    batch_size: int,
    max_length: int,
    stride: int,
    max_windows: int | None,
    num_workers: int,
    device_name: str,
    use_connection_tokens: bool,
    use_context_tokens: bool,
    use_context_features: bool,
    semantic_codebook_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = _load_classifier(checkpoint, label_map, max_length)
    model.to(device)
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
        **_loader_kwargs(num_workers, device),
    )
    outputs = collect_classifier_outputs(model, loader, device, compute_loss=False)
    probs = torch.softmax(outputs["major_logits"].float(), dim=-1).numpy()
    y_true = outputs["major_labels"].numpy()
    y_pred = probs.argmax(axis=1)
    return y_true, probs, y_pred


def _apply_threshold(
    base_pred: np.ndarray,
    probs: np.ndarray,
    *,
    target_idx: int,
    benign_idx: int,
    threshold: float,
    policy: str,
) -> np.ndarray:
    pred = base_pred.copy()
    if policy == "benign_only":
        mask = (base_pred == benign_idx) & (probs[:, target_idx] >= threshold)
    elif policy == "target_or_benign":
        mask = (
            ((base_pred == benign_idx) | (base_pred == target_idx))
            & (probs[:, target_idx] >= threshold)
        )
    elif policy == "any":
        mask = probs[:, target_idx] >= threshold
    else:
        raise ValueError(f"unsupported policy: {policy}")
    pred[mask] = target_idx
    return pred


def _target_scores(y_true: np.ndarray, y_pred: np.ndarray, target_idx: int) -> dict:
    tp = int(((y_true == target_idx) & (y_pred == target_idx)).sum())
    fp = int(((y_true != target_idx) & (y_pred == target_idx)).sum())
    fn = int(((y_true == target_idx) & (y_pred != target_idx)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int((y_true == target_idx).sum()),
    }


def _summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_map: LabelMap,
    target_idx: int,
) -> dict:
    metrics = major_classification_metrics(y_true, y_pred, label_map.major_labels)
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "target": _target_scores(y_true, y_pred, target_idx),
        "detection": attack_detection_metrics(y_true, y_pred, label_map.major_labels),
    }


def _candidate_thresholds(target_probs: np.ndarray) -> np.ndarray:
    # Scanning every unique probability is quadratic in practice on full CICIDS
    # splits. A dense grid plus empirical quantiles is enough for calibration
    # while keeping full val/test scans tractable.
    quantiles = np.quantile(target_probs, np.linspace(0.0, 1.0, 501))
    grid = np.linspace(0.001, 0.999, 1997)
    return np.unique(np.concatenate([grid, quantiles]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a one-vs-major decision threshold.")
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    parser.add_argument("--target-label", default="botnet_malware")
    parser.add_argument(
        "--policy",
        choices=["benign_only", "target_or_benign", "any"],
        default="benign_only",
    )
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-f1", type=float, default=0.75)
    parser.add_argument("--view", default="masked_header_packet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-connection-tokens", action="store_true")
    parser.add_argument("--use-context-tokens", action="store_true")
    parser.add_argument("--use-context-features", action="store_true")
    parser.add_argument("--semantic-codebook-path", type=Path, default=None)
    args = parser.parse_args()

    label_map = LabelMap.from_yaml(args.label_map)
    target_idx = label_map.major_id(args.target_label)
    benign_idx = label_map.major_id("benign")
    common = {
        "checkpoint": args.checkpoint,
        "label_map": label_map,
        "view": args.view,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "num_workers": args.num_workers,
        "device_name": args.device,
        "use_connection_tokens": args.use_connection_tokens,
        "use_context_tokens": args.use_context_tokens,
        "use_context_features": args.use_context_features,
        "semantic_codebook_path": args.semantic_codebook_path,
    }
    y_val, p_val, base_val = _collect_predictions(data_path=args.val_path, **common)
    y_test, p_test, base_test = _collect_predictions(data_path=args.test_path, **common)

    base = {
        "val": _summary(y_val, base_val, label_map=label_map, target_idx=target_idx),
        "test": _summary(y_test, base_test, label_map=label_map, target_idx=target_idx),
    }
    candidates = []
    for threshold in _candidate_thresholds(p_val[:, target_idx]):
        pred = _apply_threshold(
            base_val,
            p_val,
            target_idx=target_idx,
            benign_idx=benign_idx,
            threshold=float(threshold),
            policy=args.policy,
        )
        target = _target_scores(y_val, pred, target_idx)
        candidates.append({"threshold": float(threshold), **target})
    passing = [
        item
        for item in candidates
        if item["recall"] >= args.min_recall and item["f1"] >= args.min_f1
    ]
    pool = passing or candidates
    chosen = max(
        pool,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["precision"],
            -item["fp"],
            item["threshold"],
        ),
    )
    test_pred = _apply_threshold(
        base_test,
        p_test,
        target_idx=target_idx,
        benign_idx=benign_idx,
        threshold=float(chosen["threshold"]),
        policy=args.policy,
    )
    val_pred = _apply_threshold(
        base_val,
        p_val,
        target_idx=target_idx,
        benign_idx=benign_idx,
        threshold=float(chosen["threshold"]),
        policy=args.policy,
    )
    result = {
        "target_label": args.target_label,
        "target_idx": target_idx,
        "policy": args.policy,
        "min_recall": args.min_recall,
        "min_f1": args.min_f1,
        "base": base,
        "chosen": chosen,
        "chosen_passed_val_requirements": bool(passing),
        "val_at_chosen": _summary(y_val, val_pred, label_map=label_map, target_idx=target_idx),
        "test_at_chosen": _summary(
            y_test,
            test_pred,
            label_map=label_map,
            target_idx=target_idx,
        ),
        "top_candidates": sorted(
            candidates,
            key=lambda item: (item["f1"], item["recall"], item["precision"]),
            reverse=True,
        )[:20],
    }
    write_json(args.output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
