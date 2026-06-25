"""Gradio app: live toxicity prediction + LIME explanation heatmap.

Usage:
    python -m src.app                 # local
    python -m src.app --share         # public share link
    python -m src.app --model gru
"""
from __future__ import annotations

import argparse
import html
import re
import string

import gradio as gr

from config import DEFAULT_MODEL, LABEL_COLUMNS, MODEL_NAMES
from src.interpret import explain, make_predictor

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def predict_toxicity(comment: str, model_name: str) -> dict:
    """Return {label: probability} for a single comment."""
    predictor = make_predictor(model_name)
    probs = predictor([comment])[0]
    return {label: float(p) for label, p in zip(LABEL_COLUMNS, probs)}


def _heatmap_html(cleaned: str, per_class: dict) -> str:
    """Render each label's LIME weights as colour-coded text (red=+, blue=-)."""
    out = ["<h4>LIME explanation heatmap</h4>"]
    words = cleaned.split()
    for label, exp_list in per_class.items():
        out.append(f"<h5>{label}</h5>")
        if not exp_list:
            out.append("<p><em>No explanation for this class.</em></p>")
            continue
        weights = {w: wt for w, wt in exp_list}
        max_abs = max((abs(wt) for wt in weights.values()), default=1.0) or 1.0
        spans = []
        for word in words:
            key = _PUNCT_RE.sub("", word.lower())
            wt = weights.get(key, 0.0)
            intensity = min(abs(wt) / max_abs, 1.0)
            if wt > 0:
                color = f"rgba(255,0,0,{intensity:.3f})"
            elif wt < 0:
                color = f"rgba(0,0,255,{intensity:.3f})"
            else:
                color = "transparent"
            spans.append(
                f'<span style="background-color:{color};padding:1px 2px;">'
                f"{html.escape(word)}</span>"
            )
        out.append("<p>" + " ".join(spans) + "</p>")
    return "".join(out)


def build_interface(model_name: str) -> gr.Interface:
    def analyze(comment: str):
        if not comment or not comment.strip():
            return {}, "<p>Enter a comment to analyze.</p>"
        predictions = predict_toxicity(comment, model_name)
        cleaned, per_class = explain(comment, model_name)
        return predictions, _heatmap_html(cleaned, per_class)

    return gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(lines=5, label="Enter a comment"),
        outputs=[
            gr.Label(label="Toxicity prediction"),
            gr.HTML(label="LIME explanation heatmap"),
        ],
        title="Toxic Comment Detection with Interpretability",
        description=(
            "Multilabel toxicity classifier (toxic, severe_toxic, obscene, "
            "threat, insult, identity_hate) with LIME word-level explanations."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio toxicity demo.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--share", action="store_true", help="Create a public link.")
    args = parser.parse_args()
    build_interface(args.model).launch(share=args.share)


if __name__ == "__main__":
    main()
