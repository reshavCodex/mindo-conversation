from pathlib import Path

# ==========================================================
# Master Datasets (Read-Only)
# ==========================================================

FER_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013_CLEAN"
)

FER_CSV = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\fer2013_clean.csv"
)

CK_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\CK+"
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATASET = PROJECT_ROOT / "dataset" / "processed"

# ==========================================================
# Dataset Split Settings
# ==========================================================

RANDOM_SEED = 42

TRAIN_SPLIT = 0.90
VALIDATION_SPLIT = 0.10

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 224
NUM_CLASSES = 7

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Official FER2013 label mapping
EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}