"""Train a toxicity-classification model and save artifacts.

Usage:
    python -m src.train --model cnn_lstm
    python -m src.train --model gru --epochs 5

Saves ``models/<name>.keras`` and (once) ``models/tokenizer.pkl``.
"""
from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from config import (
    BATCH_SIZE,
    DEFAULT_MODEL,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAMES,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_CSV,
    VALIDATION_SPLIT,
    model_path,
)
from src.losses import compute_pos_weights, weighted_binary_crossentropy
from src.models import build_model
from src.preprocess import (
    build_tokenizer,
    get_labels,
    load_csv,
    save_tokenizer,
    texts_to_padded,
)


def train(model_name: str, epochs: int = EPOCHS) -> None:
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {TRAIN_CSV} ...")
    train_df = load_csv(TRAIN_CSV)

    print("Building tokenizer ...")
    tokenizer = build_tokenizer(train_df["comment_text"])
    save_tokenizer(tokenizer)

    X = texts_to_padded(tokenizer, train_df["comment_text"])
    y = get_labels(train_df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Per-label positive weights to counter class imbalance.
    pos_weights = compute_pos_weights(y)
    for col, w in zip(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], pos_weights):
        print(f"  pos_weight[{col}] = {w:.2f}")

    model = build_model(model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=weighted_binary_crossentropy(pos_weights),
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    out_path = model_path(model_name)
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        ModelCheckpoint(out_path, monitor="val_auc", mode="max", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    print(f"Training '{model_name}' for up to {epochs} epochs ...")
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best model; ensure a final copy exists.
    if not out_path.exists():
        model.save(out_path)
    print(f"Saved model to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toxic-comment classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.model, args.epochs)


if __name__ == "__main__":
    main()
