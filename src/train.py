"""Production trainer (plan Stage 4): fit the official pipeline, compare two Naive
Bayes variants on dev with a leakage-free protocol, then freeze the winner to
models/pipeline.pkl.

WHAT CHANGED vs the Stage 1 skeleton: we now (a) load data through src.data so the
column mapping / binarisation is the single canonical one, (b) bake clean_text
INTO the vectorizer so the exact same normalisation runs at fit AND at predict
time (zero train-serve skew), (c) fit ONLY on train and judge on dev, and
(d) pick the primary model from evidence — dev PR-AUC — instead of assuming it.

ANTI-LEAKAGE (the one rule we never break): the vectorizer lives INSIDE the
Pipeline, so pipe.fit(X_train, y_train) learns the vocabulary + IDF-free counts
from train ONLY. Every dev score is produced by pipe.predict_proba(X_dev), which
merely .transform()s dev through that already-fitted vocabulary — the vectorizer
is never refitted on dev. TEST is not touched at all here; it is reserved for
Stage 5.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# Running `python src/train.py` puts src/ (not the project root) on sys.path, so
# `import src.config` would fail. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score  # noqa: E402
from sklearn.naive_bayes import ComplementNB, MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import (  # noqa: E402
    ALPHA,
    ANALYZER,
    NGRAM_RANGE,
    PIPELINE_PATH,
    REPORTS_DIR,
    SEED,
)
from src.data import load_split  # noqa: E402
from src.preprocessing import clean_text  # noqa: E402

# Reference decision threshold for the SECONDARY F1 metric only. The OPTIMAL
# threshold is tuned on dev in Stage 5 — 0.5 here is just a fixed yardstick so the
# two models are compared on identical footing.
REFERENCE_THRESHOLD = 0.5

# Tie-break policy. A dev PR-AUC gap below TIE_MARGIN is practically noise on a
# single 2.6k-row dev set (bootstrap CIs on average precision are far wider than
# this), so we do NOT let a sub-margin difference decide the model. When the top
# models are within TIE_MARGIN, we prefer ComplementNB by design: it estimates each
# class from its COMPLEMENT and so corrects the majority-class bias that plain
# MultinomialNB shows on our ~17% toxic data — the property this whole project was
# built around. A gap ABOVE the margin still wins outright on evidence.
TIE_MARGIN = 0.01
PREFERRED_ON_TIE = "ComplementNB"


def build_pipeline(clf) -> Pipeline:
    """Wrap a classifier in the official feature pipeline.

    preprocessor=clean_text bakes the SAME cleaning into fit and predict, so the
    serialized .pkl carries it too -> no train-serve skew. lowercase=False because
    clean_text already lowercases; letting CountVectorizer lowercase again would be
    redundant work (and could mask a future change to clean_text's casing).
    """
    return Pipeline(
        [
            # char_wb: character n-grams that do not cross word boundaries — robust to
            # Vietnamese obfuscation ("đ.m", "nguuuu") and needs no word segmentation.
            (
                "vec",
                CountVectorizer(
                    preprocessor=clean_text,
                    lowercase=False,
                    analyzer=ANALYZER,
                    ngram_range=NGRAM_RANGE,
                ),
            ),
            ("clf", clf),
        ]
    )


def evaluate_on_dev(pipe: Pipeline, X_dev, y_dev) -> dict[str, float]:
    """Score an already-fitted pipeline on dev. Only .transform()s dev — no refit.

    PR-AUC (average_precision_score) is the PRIMARY, threshold-independent metric.
    WHY not accuracy: dev is ~82% clean, so a model that predicts CLEAN for
    everything already scores ~0.82 — accuracy rewards ignoring the toxic class we
    actually care about. WHY not ROC-AUC: on a heavily imbalanced positive class it
    is optimistically inflated (the huge true-negative pool dominates the false-
    positive rate), so it flatters weak models. Average precision summarises the
    precision-recall curve directly on the rare toxic class.
    """
    proba_dev = pipe.predict_proba(X_dev)[:, 1]  # P(toxic) for each dev comment
    pr_auc = average_precision_score(y_dev, proba_dev)
    # SECONDARY reference point only: F1 on the toxic class at a fixed 0.5 cut. Not
    # the deciding metric — the real operating threshold is chosen in Stage 5.
    f1_toxic = f1_score(y_dev, proba_dev >= REFERENCE_THRESHOLD, pos_label=1)
    return {"pr_auc": float(pr_auc), "f1_toxic_at_0.5": float(f1_toxic)}


def main() -> None:
    # Canonical, source-agnostic data. astype(str) guards against NaN text cells,
    # which would otherwise crash the vectorizer.
    train = load_split("train")
    dev = load_split("dev")
    X_train, y_train = train["text"].astype(str), train["label"]
    X_dev, y_dev = dev["text"].astype(str), dev["label"]

    # Two candidates, SAME alpha and SAME features — the only variable is the NB
    # variant, so the dev comparison is apples-to-apples.
    #   ComplementNB: estimates each class from its COMPLEMENT, correcting the bias
    #   MultinomialNB shows toward the majority class on skewed data like ours.
    candidates = {
        "ComplementNB": ComplementNB(alpha=ALPHA),
        "MultinomialNB": MultinomialNB(alpha=ALPHA),
    }

    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)  # fit ONCE, on train ONLY — see module docstring
        fitted[name] = pipe
        results[name] = evaluate_on_dev(pipe, X_dev, y_dev)

    # --- Human-readable comparison table (stdout) ---
    print("\nDev comparison (fit on train, scored on dev — test untouched):")
    print(f"{'model':<15}{'PR-AUC':>10}{'F1_toxic@0.5':>16}")
    for name, m in results.items():
        print(f"{name:<15}{m['pr_auc']:>10.4f}{m['f1_toxic_at_0.5']:>16.4f}")

    # --- Evidence-based pick: higher dev PR-AUC wins, UNLESS the gap is within
    # noise (TIE_MARGIN) — then fall back to the design-preferred variant. ---
    leader = max(results, key=lambda n: results[n]["pr_auc"])
    best_pr = results[leader]["pr_auc"]
    within_margin = [n for n in results if best_pr - results[n]["pr_auc"] <= TIE_MARGIN]
    tie_broken = leader != PREFERRED_ON_TIE and PREFERRED_ON_TIE in within_margin
    primary = PREFERRED_ON_TIE if tie_broken else leader
    if tie_broken:
        gap = best_pr - results[PREFERRED_ON_TIE]["pr_auc"]
        print(
            f"\nPR-AUC gap {gap:.4f} <= TIE_MARGIN {TIE_MARGIN}: treating as a tie, "
            f"preferring {PREFERRED_ON_TIE} over leader {leader} by design."
        )
    print(f"Primary model: {primary}")

    # --- Persist the comparison so the choice is reproducible / auditable ---
    report = {
        "date": date.today().isoformat(),
        "seed": SEED,
        "analyzer": ANALYZER,
        "ngram_range": list(NGRAM_RANGE),
        "alpha": ALPHA,
        "reference_threshold": REFERENCE_THRESHOLD,
        "primary_metric": "pr_auc",
        "tie_margin": TIE_MARGIN,
        "pr_auc_leader": leader,
        "tie_broken_by_design": tie_broken,
        "primary_model": primary,
        "results": results,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_comparison_dev.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison -> {report_path}")

    # --- Freeze the winner. It is already fitted on train ONLY; we do NOT refit on
    # train+dev, because dev must stay held-out for Stage 5 threshold tuning. ---
    best_pipe = fitted[primary]
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, PIPELINE_PATH)
    n_features = len(best_pipe.named_steps["vec"].get_feature_names_out())
    print(f"Vocabulary size: {n_features} char n-grams")
    print(f"Saved pipeline ({primary}) -> {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
