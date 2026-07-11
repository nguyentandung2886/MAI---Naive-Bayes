"""Central data-loading module: raw ViHSD CSV -> a clean, canonical DataFrame.

WHY a dedicated module: every later stage (EDA, training, evaluation) needs the
data shaped the *same* way. Centralising the load + column mapping here means the
schema decision (which column is the text, how the label is binarised) is made
once and cannot silently drift between a notebook and the trainer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# When imported as ``src.data`` from the project root, ``src.config`` resolves
# fine. But running this file directly (``python src/data.py``) puts src/ — not
# the project root — on sys.path, so ``import src.config`` would fail. Add the
# project root explicitly to make both entry points work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import TEST_CSV, TRAIN_CSV, DEV_CSV  # noqa: E402

# Map a split name to its canonical CSV path (defined once in config.py). We only
# ever expose the three OFFICIAL ViHSD splits — see the re-split warning below.
_SPLIT_PATHS: dict[str, Path] = {
    "train": TRAIN_CSV,
    "dev": DEV_CSV,
    "test": TEST_CSV,
}

# Raw column names as they actually appear in the ViHSD CSVs. We rename the text
# column to a generic ``text`` so downstream code is agnostic to the source
# dataset's naming.
_RAW_TEXT_COL = "free_text"
_RAW_LABEL_COL = "label_id"

# The only label ids ViHSD uses: 0=CLEAN, 1=OFFENSIVE, 2=HATE.
_VALID_LABEL_IDS = {0, 1, 2}


def load_split(split: str) -> pd.DataFrame:
    """Load one official ViHSD split, mapped to the canonical schema.

    Returns a DataFrame with the original ``free_text``/``label_id`` columns PLUS
    two derived columns:
      - ``text``  : a copy of ``free_text`` under a source-agnostic name.
      - ``label`` : binary toxicity flag, 0=clean / 1=toxic (see below).

    WHY binarise (OFFENSIVE + HATE -> 1): a moderation filter only needs to decide
    BLOCK vs ALLOW, so the fine-grained OFFENSIVE/HATE distinction is not
    actionable here. Merging them also makes the positive (toxic) class larger,
    which softens the heavy class imbalance (~17% toxic) the model must learn from.

    We deliberately do NOT re-split the data: re-splitting would break
    comparability with published ViHSD results and risks train/test leakage.
    Only the three official split files are ever read.
    """
    if split not in _SPLIT_PATHS:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {sorted(_SPLIT_PATHS)}."
        )

    frame = pd.read_csv(_SPLIT_PATHS[split])

    # Fail EARLY and loudly if the on-disk schema is not what every downstream
    # stage assumes — a silent column rename upstream would otherwise surface as a
    # confusing error deep inside model training.
    assert _RAW_TEXT_COL in frame.columns, (
        f"{split}: missing text column {_RAW_TEXT_COL!r}; got {list(frame.columns)}"
    )
    assert _RAW_LABEL_COL in frame.columns, (
        f"{split}: missing label column {_RAW_LABEL_COL!r}; got {list(frame.columns)}"
    )
    assert len(frame) > 0, f"{split}: loaded 0 rows from {_SPLIT_PATHS[split]}"
    unexpected = set(frame[_RAW_LABEL_COL].unique()) - _VALID_LABEL_IDS
    assert not unexpected, (
        f"{split}: {_RAW_LABEL_COL} has unexpected values {unexpected}; "
        f"expected a subset of {sorted(_VALID_LABEL_IDS)}"
    )

    # Source-agnostic text column (a view/copy under a stable name).
    frame["text"] = frame[_RAW_TEXT_COL]
    # Binary toxicity label: any non-clean id (1=OFFENSIVE, 2=HATE) becomes toxic.
    frame["label"] = (frame[_RAW_LABEL_COL] > 0).astype(int)

    return frame


def load_all() -> dict[str, pd.DataFrame]:
    """Load all three official splits, each mapped to the canonical schema.

    Convenience for EDA/evaluation that needs to iterate over every split without
    repeating the split names.
    """
    return {split: load_split(split) for split in _SPLIT_PATHS}


if __name__ == "__main__":
    # Smoke test: running this file directly should load every split and pass the
    # schema asserts, printing the shapes so a human can eyeball them.
    for _name, _df in load_all().items():
        _toxic_ratio = _df["label"].mean()
        print(f"{_name:<6} shape={_df.shape}  toxic_ratio={_toxic_ratio:.3f}")
