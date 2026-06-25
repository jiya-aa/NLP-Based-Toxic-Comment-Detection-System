"""Model architectures: GRU, CNN+LSTM hybrid and Attention-GRU.

All models share the same input (padded integer sequences of length ``MAX_LEN``)
and output (``Dense(NUM_CLASSES, activation="sigmoid")`` for multilabel
classification). Use :func:`build_model` to construct one by name.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, Sequential

from config import EMBEDDING_DIM, MAX_LEN, MAX_WORDS, NUM_CLASSES


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    """Additive (Bahdanau-style) attention over a sequence of RNN states.

    Collapses a (batch, timesteps, features) tensor into a (batch, features)
    context vector by learning a weight for each timestep.
    """

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)   # alignment scores
        alpha = tf.nn.softmax(e, axis=1)             # attention weights
        context = tf.reduce_sum(alpha * x, axis=1)   # context vector
        return context


def _build_gru() -> Model:
    return Sequential(
        [
            Input(shape=(MAX_LEN,)),
            Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
            GRU(units=64),
            Dropout(0.5),
            Dense(units=NUM_CLASSES, activation="sigmoid"),
        ],
        name="gru",
    )


def _build_cnn_lstm() -> Model:
    return Sequential(
        [
            Input(shape=(MAX_LEN,)),
            Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
            Conv1D(filters=64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=4),
            LSTM(units=64),
            Dropout(0.5),
            Dense(units=NUM_CLASSES, activation="sigmoid"),
        ],
        name="cnn_lstm",
    )


def _build_attention_gru() -> Model:
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM)(inputs)
    x = GRU(units=64, return_sequences=True)(x)  # sequences required for attention
    x = Attention()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(units=NUM_CLASSES, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs, name="attention_gru")


_BUILDERS = {
    "gru": _build_gru,
    "cnn_lstm": _build_cnn_lstm,
    "attention_gru": _build_attention_gru,
}


def build_model(name: str) -> Model:
    """Construct an (uncompiled) model by name."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_BUILDERS)}.")
    return _BUILDERS[name]()
