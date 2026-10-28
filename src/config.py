"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

MNIST_NPZ = DATA_DIR / "mnist.npz"
LATEST_MODEL_POINTER = MODELS_DIR / "latest.txt"
EXPERIMENT_LOG = MODELS_DIR / "experiments.json"

IMAGE_SIZE = 28
N_FEATURES = IMAGE_SIZE * IMAGE_SIZE  # 784
CLASSES = list(range(10))

# Default hyper-parameters: chosen so training finishes in well under a minute
# on an ordinary laptop while still reaching ~93% test accuracy.
DEFAULT_N_ESTIMATORS = 50
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_MAX_DEPTH = 6
DEFAULT_TRAIN_SIZE = 10_000
DEFAULT_RANDOM_STATE = 42

# Above this many training samples we warn the user that training is slow.
SLOW_TRAINING_WARNING_SIZE = 25_000
