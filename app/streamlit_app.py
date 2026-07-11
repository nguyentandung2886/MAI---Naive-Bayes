"""Minimal Streamlit demo (plan Stage 1): type a Vietnamese comment, get a label.

Skeleton scope only: one chat input, model loaded once, print clean/toxic + score.
Censoring, highlighting and multi-turn history come in Stage 7.
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit runs this file with app/ on sys.path (not the project root), so
# `import src.config` needs the project root added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import PIPELINE_PATH  # noqa: E402

# Skeleton uses a plain 0.5 cut-off. Principled threshold tuning on dev is Stage 5.
TOXIC_THRESHOLD = 0.5


@st.cache_resource
def load_model():
    """Load the pipeline once and keep it across reruns.

    WHY @st.cache_resource (not @st.cache_data): the model is a shared, non-
    serialisable resource. Without this, every keystroke-level rerun would
    reload the .pkl from disk and make the app lag badly.
    """
    return joblib.load(PIPELINE_PATH)


def classify(text: str, model) -> tuple[str, float]:
    """Return (label, probability_toxic) for one comment.

    Kept as a pure function (model passed in) so it is testable without the
    Streamlit runtime.
    """
    prob_toxic = float(model.predict_proba([text])[0][1])
    label = "độc hại" if prob_toxic >= TOXIC_THRESHOLD else "sạch"
    return label, prob_toxic


def render() -> None:
    st.set_page_config(page_title="Bộ lọc bình luận độc hại", page_icon="🛡️")
    st.title("🛡️ Bộ lọc bình luận độc hại (v0.1-skeleton)")
    st.caption("Gõ một bình luận tiếng Việt để phân loại. Đây là bản skeleton, chất lượng còn thô.")

    model = load_model()

    user_text = st.chat_input("Nhập bình luận...")
    if not user_text:
        return

    with st.chat_message("user"):
        st.write(user_text)

    label, prob_toxic = classify(user_text, model)
    with st.chat_message("assistant"):
        if label == "độc hại":
            st.error(f"🚫 **{label}** — xác suất độc hại: {prob_toxic:.1%}")
        else:
            st.success(f"✅ **{label}** — xác suất độc hại: {prob_toxic:.1%}")


if __name__ == "__main__":
    render()
