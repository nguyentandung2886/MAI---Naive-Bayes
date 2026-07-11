"""Walking-skeleton trainer: raw ViHSD -> models/pipeline.pkl.

This is intentionally minimal (plan Stage 1). It trains on a small slice with
crude binary labels, only to prove the end-to-end wire works:
    data -> Pipeline(vectorizer + classifier) -> single .pkl
Proper column mapping (src/data.py), preprocessing, model comparison and
threshold tuning arrive in later stages. Do not optimise here — the skeleton is
allowed to be "ugly but running" on purpose.
"""
from __future__ import annotations

import sys
from pathlib import Path

# src/train.py run as `python src/train.py` puts src/ (not the project root) on
# sys.path, so `import src.config` would fail. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import (  # noqa: E402
    ALPHA,
    ANALYZER,
    NGRAM_RANGE,
    PIPELINE_PATH,
    SEED,
    TRAIN_CSV,
)

# Small slice keeps the skeleton fast; the full dataset is used from Stage 4 on.
SKELETON_SAMPLE_SIZE = 2000


def main() -> None:
    frame = pd.read_csv(TRAIN_CSV)

    # Fixed SEED so the sampled slice is identical on every run (reproducibility).
    sample = frame.sample(n=min(SKELETON_SAMPLE_SIZE, len(frame)), random_state=SEED)
    texts = sample["free_text"].astype(str)  # astype(str) guards against NaN cells

    # Crude binary label: OFFENSIVE(1) + HATE(2) -> toxic(1); CLEAN(0) -> clean(0).
    # Moderation only needs BLOCK / ALLOW, and merging grows the toxic class.
    labels = (sample["label_id"] > 0).astype(int)

    pipeline = Pipeline(
        [
            # char_wb: character n-grams that do not cross word boundaries — robust
            # to Vietnamese obfuscation and needs no word segmentation.
            ("vec", CountVectorizer(analyzer=ANALYZER, ngram_range=NGRAM_RANGE)),
            # ComplementNB estimates parameters from each class's COMPLEMENT, so it
            # leans less toward the majority class than MultinomialNB on skewed data.
            ("clf", ComplementNB(alpha=ALPHA)),
        ]
    )

    # No data leakage even in the skeleton: the vectorizer lives INSIDE the pipeline,
    # so .fit() learns the vocabulary only from this training slice — dev/test would
    # later only .transform(), never refit.
    pipeline.fit(texts, labels)

    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    n_features = len(pipeline.named_steps["vec"].get_feature_names_out())
    print(f"Trained on {len(sample)} rows | toxic={int(labels.sum())} clean={int((labels == 0).sum())}")
    print(f"Vocabulary size: {n_features} char n-grams")
    print(f"Saved pipeline -> {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
