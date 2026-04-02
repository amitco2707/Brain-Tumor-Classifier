"""
config.py — Central place for all project hyperparameters and paths.

Why have this file?
Previously, settings like batch_size, num_epochs, and learning_rate were
scattered across main.py, loader.py, and trainer.py. If you wanted to
experiment with different values, you had to hunt through multiple files.
Now you change one number here and it flows everywhere automatically.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) means "the folder where this config.py lives"
# That is always the project root, no matter where you run Python from.
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ── Data loading ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 16   # how many images to feed the model at once
RANDOM_SEED = 42   # fixes the train/val/test split so it's the same every run

# Train / val / test split ratios (must add up to 1.0)
# 70 % of images for training, 15 % each for validation and test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Test ratio = whatever is left: 1.0 - 0.70 - 0.15 = 0.15

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES = 2   # tumor (1) or no tumor (0)

# ── Training hyperparameters ──────────────────────────────────────────────────
NUM_EPOCHS    = 15     # how many full passes through the training data
LEARNING_RATE = 0.001  # how large each weight-update step is
