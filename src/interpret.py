"""LIME-based interpretability for the toxic-comment models.

Provides a prediction function compatible with ``LimeTextExplainer`` and a
helper that returns per-class (word, weight) explanations.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from lime.lime_text import LimeTextExplainer

from config import DEFAULT_MODEL, LABEL_COLUMNS, model_path
from src.preprocess import clean_text, load_tokenizer, texts_to_padded


@lru_cache(maxsize=4)
def _load(model_name: str):
    """Load and cache (model, tokenizer) so repeated calls are cheap."""
    from tensorflow.keras.models import load_model  # local import: keep startup light
    from src.models import Attention  # noqa: F401 (custom-object registration)

    tokenizer = load_tokenizer()
    model = load_model(model_path(model_name), compile=False)
    return model, tokenizer


def make_predictor(model_name: str = DEFAULT_MODEL):
    """Return a function mapping a list of raw texts -> (n, NUM_CLASSES) probs."""
    model, tokenizer = _load(model_name)

    def predictor(texts):
        cleaned = [clean_text(t) for t in texts]
        X = texts_to_padded(tokenizer, cleaned)
        return model.predict(X, verbose=0)

    return predictor


def explain(text: str, model_name: str = DEFAULT_MODEL, num_features: int = 20,
            num_samples: int = 1000):
    """Return (cleaned_text, {label: [(word, weight), ...]}) LIME explanations."""
    predictor = make_predictor(model_name)
    explainer = LimeTextExplainer(class_names=LABEL_COLUMNS)
    cleaned = clean_text(text)

    explanation = explainer.explain_instance(
        cleaned,
        predictor,
        num_features=num_features,
        num_samples=num_samples,
        labels=list(range(len(LABEL_COLUMNS))),
    )

    per_class = {}
    for i, label in enumerate(LABEL_COLUMNS):
        try:
            per_class[label] = explanation.as_list(label=i)
        except KeyError:
            per_class[label] = []
    return cleaned, per_class
