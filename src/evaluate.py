"""Stage 5 evaluation: lock a decision threshold on dev, then touch test ONCE.

ANTI-LEAKAGE (the rule this whole stage exists to enforce): the model is LOADED
from models/pipeline.pkl and never refit here. Everything that could adapt to the
data — choosing the operating threshold — is done on DEV only. Test probabilities
are computed exactly once, AFTER the threshold is locked, and feed reporting only.
No test number ever flows back into a selection decision. That separation is what
makes the reported test figures an honest estimate of production performance.

WHY a precision–recall curve (not ROC) to pick the cut: dev is ~83% clean, so
ROC-AUC is optimistically inflated by the huge true-negative pool. The PR curve
looks straight at the rare toxic class we actually care about.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# Running `python src/evaluate.py` puts src/ (not the project root) on sys.path, so
# `import src.config` would fail. Add the project root explicitly — mirrors train.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

# Headless environment: choose the non-interactive Agg backend BEFORE importing
# pyplot, otherwise matplotlib tries to open a GUI backend and crashes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import joblib  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_PATH,
    PIPELINE_PATH,
    THRESHOLD_PATH,
)
from src.data import load_split  # noqa: E402

# Toxic is the positive/rare class every metric below is scored against.
POS_LABEL = 1
# Product-oriented alternative to F1: "rather miss some toxic than wrongly block
# clean users". Reported as a candidate, never used as the operating point.
PRECISION_FLOOR = 0.80


def toxic_class_scores(y_true, y_pred) -> dict[str, float]:
    """Precision / recall / F1 for the TOXIC (positive) class only.

    zero_division=0 so a degenerate predictor — the majority baseline never emits a
    toxic label — scores 0 instead of raising. That 0 is exactly the floor we want
    to contrast the real model against, not an error.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
    }


def main() -> None:
    # LOAD the frozen model — Stage 5 never fits or refits it. A missing file means
    # Stage 4 (src/train.py) has not run; stop loudly instead of fabricating numbers.
    if not PIPELINE_PATH.exists():
        raise SystemExit(
            f"Frozen pipeline not found at {PIPELINE_PATH}. Run Stage 4 (src/train.py) first."
        )
    pipe = joblib.load(PIPELINE_PATH)

    # Canonical splits. astype(str) mirrors train.py, guarding the vectorizer against
    # NaN text cells; the pipeline bakes clean_text internally, so raw text goes in.
    train = load_split("train")
    dev = load_split("dev")
    test = load_split("test")
    X_train, y_train = train["text"].astype(str), train["label"]
    X_dev, y_dev = dev["text"].astype(str), dev["label"]
    X_test, y_test = test["text"].astype(str), test["label"]

    # === DEV ONLY. Everything that SELECTS the threshold happens in this block; test
    # is not read until the operating threshold is locked further down. ===
    proba_dev = pipe.predict_proba(X_dev)[:, 1]  # P(toxic) for each dev comment
    dev_pr_auc = float(average_precision_score(y_dev, proba_dev))

    precision, recall, thresholds = precision_recall_curve(y_dev, proba_dev)
    # precision_recall_curve returns precision/recall of length n but thresholds of
    # length n-1: the final (precision=1, recall=0) point has NO threshold. Drop that
    # trailing point so every F1 value maps to a real cut — otherwise argmax is
    # silently off by one and we lock the wrong threshold.
    p = precision[:-1]
    r = recall[:-1]
    f1_curve = 2 * p * r / (p + r + 1e-9)

    # Operating threshold = the cut that MAXIMISES toxic-class F1 on dev. F1 is the
    # neutral default: it balances catching toxic against wrongly blocking clean.
    best_idx = int(np.argmax(f1_curve))
    operating_threshold = float(thresholds[best_idx])

    # Alternative candidate (reported, NOT operated on): highest recall reachable
    # while dev precision stays >= 0.80. If the floor is unreachable at any cut,
    # record reachable=false and move on rather than crash.
    floor_mask = p >= PRECISION_FLOOR
    if floor_mask.any():
        # Among cuts clearing the floor, take the one with the most recall.
        # where(mask, r, -1) blanks out failing points so argmax ignores them.
        floor_idx = int(np.argmax(np.where(floor_mask, r, -1.0)))
        precision_floor_candidate = {
            "reachable": True,
            "threshold": float(thresholds[floor_idx]),
            "precision": float(p[floor_idx]),
            "recall": float(r[floor_idx]),
        }
    else:
        precision_floor_candidate = {"reachable": False}

    # Threshold is now LOCKED from dev. Only now do we score dev AT it. Using `>=`
    # matches precision_recall_curve's own convention, so these numbers agree with
    # the curve point we marked.
    dev_pred = (proba_dev >= operating_threshold).astype(int)
    dev_scores = toxic_class_scores(y_dev, dev_pred)

    # === TEST: touched EXACTLY ONCE, for reporting only, strictly after the lock.
    # Nothing computed below influences the threshold or the model. ===
    today = date.today().isoformat()
    proba_test = pipe.predict_proba(X_test)[:, 1]
    test_pr_auc = float(average_precision_score(y_test, proba_test))
    test_pred = (proba_test >= operating_threshold).astype(int)
    test_scores = toxic_class_scores(y_test, test_pred)

    # === Majority-class baseline. DummyClassifier(most_frequent) always predicts the
    # majority class (clean), so its toxic-class F1 is 0 by construction — the sanity
    # floor proving the real model learned genuine signal, not the base rate. ===
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    baseline_dev = toxic_class_scores(y_dev, dummy.predict(X_dev))
    baseline_test = toxic_class_scores(y_test, dummy.predict(X_test))

    # --- Figures ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # (1) Dev PR curve with the chosen operating point marked.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4", label=f"dev PR curve (AP={dev_pr_auc:.3f})")
    ax.scatter(
        [r[best_idx]],
        [p[best_idx]],
        color="#d62728",
        zorder=5,
        label=f"operating point (t={operating_threshold:.3g}, F1={dev_scores['f1']:.3f})",
    )
    ax.set_xlabel("Recall (toxic)")
    ax.set_ylabel("Precision (toxic)")
    ax.set_title("Dev precision–recall — threshold chosen on dev only")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    # (2) Test confusion matrix at the locked operating threshold.
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["clean", "toxic"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Test confusion matrix @ t={operating_threshold:.3g}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Persist the operating decision (threshold.json) ---
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    threshold_payload = {
        "operating_threshold": operating_threshold,
        "selection_rule": "max F1 on the toxic class, selected on dev only",
        "date": today,
    }
    THRESHOLD_PATH.write_text(
        json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Persist the full dev-vs-test-vs-baseline record + audit trail (metrics.json) ---
    metrics = {
        "operating_threshold": operating_threshold,
        "threshold_selection_rule": "max F1 on the toxic (positive) class, dev only",
        "precision_floor_candidate": {"floor": PRECISION_FLOOR, **precision_floor_candidate},
        "pr_auc": {"dev": dev_pr_auc, "test": test_pr_auc},
        "model_toxic_class": {"dev": dev_scores, "test": test_scores},
        "majority_baseline_toxic_class": {"dev": baseline_dev, "test": baseline_test},
        "audit": f"test evaluated once on {today} at threshold={operating_threshold:.6g}",
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Human-readable dev-vs-test summary (stdout) ---
    print("\n=== Stage 5 evaluation (frozen pipeline LOADED, never refit) ===")
    print(f"Operating threshold (max dev toxic-F1): {operating_threshold:.4g}")
    if precision_floor_candidate["reachable"]:
        c = precision_floor_candidate
        print(
            f"Precision>={PRECISION_FLOOR:.2f} candidate: t={c['threshold']:.4g} "
            f"(P={c['precision']:.3f}, R={c['recall']:.3f})"
        )
    else:
        print(f"Precision>={PRECISION_FLOOR:.2f} candidate: unreachable on dev")

    print(f"\n{'split':<16}{'PR-AUC':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    print(
        f"{'dev  (model)':<16}{dev_pr_auc:>9.4f}"
        f"{dev_scores['precision']:>11.4f}{dev_scores['recall']:>9.4f}{dev_scores['f1']:>8.4f}"
    )
    print(
        f"{'test (model)':<16}{test_pr_auc:>9.4f}"
        f"{test_scores['precision']:>11.4f}{test_scores['recall']:>9.4f}{test_scores['f1']:>8.4f}"
    )
    print(
        f"{'dev  (baseline)':<16}{'-':>9}"
        f"{baseline_dev['precision']:>11.4f}{baseline_dev['recall']:>9.4f}{baseline_dev['f1']:>8.4f}"
    )
    print(
        f"{'test (baseline)':<16}{'-':>9}"
        f"{baseline_test['precision']:>11.4f}{baseline_test['recall']:>9.4f}{baseline_test['f1']:>8.4f}"
    )

    print(f"\n{metrics['audit']}")
    print(
        f"Saved: {THRESHOLD_PATH.name}, {METRICS_PATH.name}, "
        "figures/pr_curve.png, figures/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()
