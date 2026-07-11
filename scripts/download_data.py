"""Download ViHSD into data/raw/ as plain CSV files.

ViHSD is a *gated* dataset: the canonical copy at ``uitnlp/vihsd`` requires a
HuggingFace login plus a one-time acceptance of the dataset terms. A pending
approval on day one would stall the entire project, so this script automatically
falls back to a public mirror (``sonlam1102/vihsd``) if the canonical source is
unreachable.

Run from the project root (after ``huggingface-cli login``):

    python scripts/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# This script lives in scripts/, which is NOT a package, so `import src.config`
# would fail with the default sys.path. Add the project root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import DatasetDict, load_dataset  # noqa: E402  (import after path fix)

from src.config import DATA_RAW_DIR  # noqa: E402

PRIMARY_SOURCE = "uitnlp/vihsd"       # canonical, gated
MIRROR_SOURCE = "sonlam1102/vihsd"    # public mirror, fallback

# HF sometimes names the validation split "validation"; we standardise on "dev"
# so the rest of the project can rely on train.csv / dev.csv / test.csv.
SPLIT_RENAME = {"validation": "dev", "valid": "dev"}


def _try_load(source: str) -> DatasetDict | None:
    """Load one source, retrying with trust_remote_code for script-based datasets."""
    for kwargs in ({}, {"trust_remote_code": True}):
        try:
            return load_dataset(source, **kwargs)
        except Exception as exc:  # noqa: BLE001 — we intentionally try the next option
            print(f"[warn] load_dataset({source!r}, {kwargs}) failed: {exc}")
    return None


def load_vihsd() -> tuple[DatasetDict, str]:
    for source in (PRIMARY_SOURCE, MIRROR_SOURCE):
        dataset = _try_load(source)
        if dataset is not None:
            return dataset, source
    raise SystemExit(
        "Could not load ViHSD from any source.\n"
        "  1. Run: huggingface-cli login\n"
        "  2. Accept the terms at https://huggingface.co/datasets/uitnlp/vihsd\n"
        "  3. Re-run this script."
    )


def main() -> None:
    dataset, source = load_vihsd()
    print(f"Loaded ViHSD from {source!r}. Splits: {list(dataset.keys())}")

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split in dataset.items():
        out_name = SPLIT_RENAME.get(split_name, split_name)
        frame = split.to_pandas()
        out_path = DATA_RAW_DIR / f"{out_name}.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8")
        # df.shape confirms we actually got rows and columns — not an empty pull.
        print(f"  {out_name:<6} df.shape={frame.shape}  columns={list(frame.columns)}  -> {out_path}")


if __name__ == "__main__":
    main()
