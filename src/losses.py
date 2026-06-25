"""Per-label class weighting for the imbalanced multilabel problem.

Keras' built-in ``class_weight`` argument does **not** work for a multilabel
``Dense(6, activation="sigmoid")`` head, so we implement weighting directly in
the loss: each label gets a positive-class weight inversely proportional to how
often that label is positive in the training set.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_pos_weights(labels: np.ndarray) -> np.ndarray:
    """Return per-label positive weights = (#neg / #pos), clipped for stability.

    ``labels`` is an array of shape (n_samples, n_labels) of 0/1 values.
    Rarer labels (e.g. ``threat``) get a larger weight so the model is penalised
    more for missing them.
    """
    labels = np.asarray(labels)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    # Avoid division by zero; clip so a single ultra-rare label can't dominate.
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(weights, 1.0, 100.0).astype("float32")


def weighted_binary_crossentropy(pos_weights):
    """Build a weighted BCE loss given per-label positive weights.

    Positive targets are scaled by ``pos_weight``; negative targets keep weight 1.
    """
    pos_weights = tf.constant(np.asarray(pos_weights), dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce = -(
            pos_weights * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(bce)

    return loss
