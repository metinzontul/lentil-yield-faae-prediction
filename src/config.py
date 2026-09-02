# -*- coding: utf-8 -*-


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Data ----------------
DATA_PATH = os.path.join(BASE_DIR, "data", "Lentil.csv")
CSV_SEP = ";"
CSV_ENCODING = "utf-8-sig"

RESULTS_DIR = os.path.join(BASE_DIR, "results")


LEAKY_FEATURE = "Seed Yield per Plant (g)"

DROP_FEATURES = [
    "Seed Yield per Plant (g)",
]

# ---------------- Reproducibility ----------------
SEED = 42

# ---------------- Train / Test ----------------
TEST_SIZE = 0.20

# ---------------- Cross-validation ----------------
N_SPLITS = 5

# ---------------- Repeated cross-validation ----------------
N_REPEATS = 3
