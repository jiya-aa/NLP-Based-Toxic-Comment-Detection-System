"""Central configuration: paths and hyperparameters.

Everything that the training, evaluation and serving scripts need to agree on
lives here so the tokenizer, model and app never drift out of sync.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_LABELS_CSV = DATA_DIR / "test_labels.csv"

TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
# Model artifacts are saved as models/<name>.keras (e.g. models/cnn_lstm.keras)


def model_path(name: str) -> Path:
    """Path to the saved Keras artifact for a given model name."""
    return MODELS_DIR / f"{name}.keras"


# --- Data ------------------------------------------------------------------
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMN = "comment_text"
NUM_CLASSES = len(LABEL_COLUMNS)

# --- Preprocessing ---------------------------------------------------------
MAX_WORDS = 20000   # vocabulary size for the tokenizer
MAX_LEN = 100       # padded sequence length
OOV_TOKEN = "<OOV>"

# --- Training --------------------------------------------------------------
EMBEDDING_DIM = 128
BATCH_SIZE = 256
EPOCHS = 10
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Available architectures (see src/models.py)
MODEL_NAMES = ["gru", "cnn_lstm", "attention_gru"]
DEFAULT_MODEL = "cnn_lstm"
