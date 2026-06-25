"""Data loading, text cleaning and tokenization.

The tokenizer is fit **once** on the training data and persisted to
``models/tokenizer.pkl`` so that training, evaluation and the Gradio app all
use the exact same vocabulary (the original notebook re-fit it several times,
which is a subtle source of bugs).
"""
from __future__ import annotations

import pickle
import re
import string

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import (
    LABEL_COLUMNS,
    MAX_LEN,
    MAX_WORDS,
    OOV_TOKEN,
    TEXT_COLUMN,
    TOKENIZER_PATH,
)

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_DIGIT_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits and collapse whitespace."""
    text = str(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _DIGIT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def load_csv(path) -> pd.DataFrame:
    """Load a CSV and clean its comment column (no chained-assignment warning)."""
    df = pd.read_csv(path)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").apply(clean_text)
    return df


def build_tokenizer(texts) -> Tokenizer:
    """Fit a new Keras tokenizer on the given texts."""
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path=TOKENIZER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh)


def load_tokenizer(path=TOKENIZER_PATH) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. Run `python -m src.train` first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def texts_to_padded(tokenizer: Tokenizer, texts) -> np.ndarray:
    """Convert raw (already-cleaned) texts to padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN)


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract the 6 label columns as a float32 array."""
    return df[LABEL_COLUMNS].values.astype("float32")
