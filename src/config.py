"""Central configuration: every constant and path lives here.

WHY one config module: reproducibility. Anything that could change a result
(random seed, n-gram range, smoothing) or a location on disk is defined once and
imported everywhere. When a future stage needs to tweak a hyperparameter, there
is a single obvious place to do it — no magic numbers scattered across files that
silently drift apart.
"""
from __future__ import annotations

from pathlib import Path

# --- Reproducibility / model hyperparameters ---
SEED: int = 42
# char_wb = character n-grams that do NOT cross word boundaries. Robust to Vietnamese
# obfuscation ("đ.m", "nguuuu") and needs no word segmentation. See plan Stage 4.
ANALYZER: str = "char_wb"
NGRAM_RANGE: tuple[int, int] = (2, 4)
ALPHA: float = 1.0  # Laplace smoothing for (Complement)NB: avoids zero probabilities.

# --- Paths (all derived from the project root so they work on any machine) ---
# parents[1] == the toxic-comment-filter/ project root (this file is src/config.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
PIPELINE_PATH: Path = MODELS_DIR / "pipeline.pkl"
THRESHOLD_PATH: Path = MODELS_DIR / "threshold.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"

# Canonical ViHSD split files, written by scripts/download_data.py.
# We keep the official train/dev/test split names — never re-split (see plan Stage 2).
TRAIN_CSV: Path = DATA_RAW_DIR / "train.csv"
DEV_CSV: Path = DATA_RAW_DIR / "dev.csv"
TEST_CSV: Path = DATA_RAW_DIR / "test.csv"
