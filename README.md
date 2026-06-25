# NLP-Based Toxic Comment Detection System

Multilabel classification of online comments into six toxicity types —
**toxic, severe_toxic, obscene, threat, insult, identity_hate** — using deep
learning, with **LIME interpretability** and a **Gradio** web demo.

Built on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset.

## Features

- **Three selectable architectures** (`src/models.py`):
  - `gru` — embedding → GRU → dense
  - `cnn_lstm` — embedding → Conv1D → MaxPool → LSTM → dense (default)
  - `attention_gru` — embedding → GRU(return_sequences) → additive attention → dense
- **Imbalance handling** via per-label class weighting in a custom weighted
  binary cross-entropy loss (`src/losses.py`). Standard `class_weight` does not
  work for a multilabel sigmoid head, so weighting is applied inside the loss.
- **Honest evaluation** (`src/evaluate.py`): per-label ROC-AUC, F1, precision,
  recall, and the mean column-wise ROC-AUC used by the Jigsaw competition —
  rather than accuracy, which is misleading when ~99% of labels are 0.
- **Interpretability** (`src/interpret.py`): LIME word-level explanations.
- **Web demo** (`src/app.py`): Gradio UI with a colour-coded explanation heatmap.
- A single tokenizer is fit once and persisted, so training, evaluation and the
  app always share the same vocabulary.

## Project structure

```
.
├── config.py            # paths + hyperparameters (single source of truth)
├── requirements.txt
├── src/
│   ├── preprocess.py    # text cleaning, tokenizer build/save/load, padding
│   ├── models.py        # GRU / CNN+LSTM / Attention-GRU + Attention layer
│   ├── losses.py        # per-label weighted binary cross-entropy
│   ├── train.py         # train + callbacks, saves models/<name>.keras
│   ├── evaluate.py      # per-label ROC-AUC / F1 / precision / recall
│   ├── interpret.py     # LIME explanations
│   └── app.py           # Gradio demo with LIME heatmap
├── notebooks/
│   └── DL_heatmap.ipynb # original exploratory notebook
├── data/                # (gitignored) train.csv, test.csv, test_labels.csv
└── models/              # (gitignored) saved .keras model + tokenizer.pkl
```

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data

Download the dataset from the
[Jigsaw competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
and place the CSVs in `data/`:

```
data/train.csv
data/test.csv
data/test_labels.csv
```

## Usage

Train a model (saves `models/<name>.keras` and `models/tokenizer.pkl`):

```bash
python -m src.train --model cnn_lstm           # or gru / attention_gru
python -m src.train --model gru --epochs 5
```

Evaluate on the labeled test split:

```bash
python -m src.evaluate --model cnn_lstm
```

Launch the interactive demo:

```bash
python -m src.app --model cnn_lstm             # add --share for a public link
```

## Notes

- Metrics: prefer **mean column-wise ROC-AUC** for model selection on this
  dataset; accuracy is not informative under heavy class imbalance.
- The `data/` and `models/` directories are gitignored (large / regenerable).
- All commands are run from the repository root as modules (`python -m src.*`)
  so that `config.py` resolves correctly.
