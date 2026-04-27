"""Classical byte n-gram baselines for flow classification."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from traffic_bert.labels import LabelMap


def bytes_to_hex_text(data: bytes | memoryview | bytearray) -> str:
    """Represent bytes as whitespace-separated byte tokens for sklearn."""

    if isinstance(data, memoryview):
        data = data.tobytes()
    return " ".join(f"{value:02x}" for value in bytes(data))


def frame_to_texts(frame: pd.DataFrame) -> list[str]:
    return [bytes_to_hex_text(value) for value in frame["bytes"]]


@dataclass(frozen=True)
class BaselineResult:
    model_name: str
    metrics: dict[str, Any]


def create_tfidf_linear_pipeline(
    model: str = "logreg",
    ngram_range: tuple[int, int] = (1, 4),
    max_features: int = 200_000,
) -> Pipeline:
    """Create a byte n-gram TF-IDF classifier pipeline."""

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
        dtype=np.float32,
    )
    if model == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
    elif model == "linear_svm":
        classifier = SGDClassifier(
            loss="hinge",
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
        )
    else:
        raise ValueError(f"unsupported baseline model: {model}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def train_major_baseline(
    train_frame: pd.DataFrame,
    label_map: LabelMap,
    model: str = "logreg",
    ngram_range: tuple[int, int] = (1, 4),
    max_features: int = 200_000,
) -> Pipeline:
    """Train a classical baseline on processed flow rows."""

    pipeline = create_tfidf_linear_pipeline(
        model=model,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    x_train = frame_to_texts(train_frame)
    y_train = [label_map.major_id(str(label)) for label in train_frame["major_label"]]
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_major_baseline(
    pipeline: Pipeline,
    eval_frame: pd.DataFrame,
    label_map: LabelMap,
) -> BaselineResult:
    x_eval = frame_to_texts(eval_frame)
    y_true = [label_map.major_id(str(label)) for label in eval_frame["major_label"]]
    y_pred = pipeline.predict(x_eval)
    return BaselineResult(
        model_name=str(pipeline.named_steps["classifier"].__class__.__name__),
        metrics=classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_map.major_labels))),
            target_names=label_map.major_labels,
            output_dict=True,
            zero_division=0,
        ),
    )


def save_baseline(path: str | Path, pipeline: Pipeline) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(pipeline, handle)


def load_baseline(path: str | Path) -> Pipeline:
    with open(path, "rb") as handle:
        return pickle.load(handle)
