"""Evaluate a trained model with metrics that are honest for imbalanced data.

Accuracy is meaningless here (~99% of labels are 0), so we report per-label
ROC-AUC, F1, precision and recall, plus the mean column-wise ROC-AUC used by
the Jigsaw competition.

Usage:
    python -m src.evaluate --model cnn_lstm
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.models import load_model

from config import (
    DEFAULT_MODEL,
    LABEL_COLUMNS,
    MODEL_NAMES,
    TEST_CSV,
    TEST_LABELS_CSV,
    model_path,
)
from src.losses import weighted_binary_crossentropy  # noqa: F401 (custom-object registration)
from src.models import Attention  # noqa: F401 (custom-object registration)
from src.preprocess import load_csv, load_tokenizer, texts_to_padded


def _load_labeled_test() -> pd.DataFrame:
    """Merge test comments with labels and drop unscored rows (label == -1)."""
    test_df = load_csv(TEST_CSV)
    labels_df = pd.read_csv(TEST_LABELS_CSV)
    merged = test_df.merge(labels_df, on="id")
    mask = (merged[LABEL_COLUMNS] != -1).all(axis=1)
    return merged[mask].reset_index(drop=True)


def evaluate(model_name: str, threshold: float = 0.5) -> pd.DataFrame:
    tokenizer = load_tokenizer()
    model = load_model(
        model_path(model_name),
        compile=False,  # we only need predictions; custom loss not required
    )

    df = _load_labeled_test()
    X = texts_to_padded(tokenizer, df["comment_text"])
    y_true = df[LABEL_COLUMNS].values.astype(int)
    y_prob = model.predict(X, batch_size=512)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for i, label in enumerate(LABEL_COLUMNS):
        yt, yp, pr = y_true[:, i], y_pred[:, i], y_prob[:, i]
        auc = roc_auc_score(yt, pr) if yt.sum() > 0 else float("nan")
        rows.append(
            {
                "label": label,
                "roc_auc": auc,
                "f1": f1_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "support": int(yt.sum()),
            }
        )

    report = pd.DataFrame(rows).set_index("label")
    mean_auc = report["roc_auc"].mean()
    macro_f1 = report["f1"].mean()

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\nEvaluation report for '{model_name}' (threshold={threshold}):\n")
    print(report)
    print(f"\nMean column-wise ROC-AUC : {mean_auc:.4f}")
    print(f"Macro F1                 : {macro_f1:.4f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a toxic-comment classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.model, args.threshold)


if __name__ == "__main__":
    main()
