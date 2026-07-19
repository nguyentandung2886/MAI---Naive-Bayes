"""Giai đoạn 7 — giao diện chat kiểm duyệt (block + censor + explain).

Một khung chat tiếng Việt: mỗi bình luận vẫn được ĐĂNG lên, nhưng nếu mô hình
đánh giá là độc hại thì nội dung hiển thị ở dạng đã che (***) kèm huy hiệu cảnh
báo và một ô "Vì sao?" giải thích. Bình luận sạch đăng nguyên văn.

WHY this file only orchestrates UI: every inference decision — P(toxic), the block
threshold, which words to mask, and the per-word log-odds used for highlighting —
lives in src/explain.py (Stage 6, tested). This module NEVER loads the model, reads
threshold.json, or re-implements masking; it just calls explain()/censor() and paints
the result. One source of truth = the explanation can never contradict the block.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` puts app/ (not the project root) on sys.path,
# so `import src.explain` would fail. Add the project root explicitly BEFORE importing
# it — mirrors the same fix in src/explain.py and src/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

# Colour of the toxic highlight (matplotlib's "tab:red" as RGB). alpha varies per word.
_HIGHLIGHT_RGB = "214, 39, 40"


@st.cache_resource
def load_explainer():
    """Import src.explain once and hand the module back to every rerun.

    WHY @st.cache_resource on top of Python's own import cache: importing src.explain
    triggers its module-level model load (pipeline + threshold + n-gram log-odds map),
    which Python's sys.modules already runs exactly once per process. But Streamlit
    re-executes THIS script top-to-bottom on every interaction, so without caching the
    wrapper (and its one-time log line) would run each rerun. cache_resource memoises
    the returned module — the idiomatic Streamlit way to share a heavy, non-serialisable
    resource — so the load, and the evidence print below, happen a single time.
    """
    import src.explain as explain  # noqa: E402  (deferred so the sys.path fix applies)

    # Stage 7 DoD: this must appear ONCE in the server log no matter how many messages
    # are sent — proof the model is not reloaded per keystroke/rerun.
    print("[Stage 7] src.explain loaded once — model, threshold & log-odds cached.")
    return explain


def _highlight_html(text: str, spans: list[dict]) -> str:
    """Render the ORIGINAL text with each lexicon-flagged (masked) word highlighted.

    A word is tinted iff span["toxic"] — the SAME lexicon signal censor() masks on — so
    the highlight always matches the "***" in the posted body. Highlighting is flat (the
    match is binary), not graded by log-odds. Every character coming from user input is
    html.escape()d first — raw input may contain < > & that would otherwise break the
    layout or inject markup into the unsafe_allow_html render.
    """
    parts: list[str] = []
    last = 0
    for s in spans:
        # Whitespace/gap before this token, kept verbatim (escaped) to preserve layout.
        parts.append(html.escape(text[last : s["start"]]))
        word = html.escape(text[s["start"] : s["end"]])
        if s["toxic"]:
            parts.append(
                f'<span style="background: rgba({_HIGHLIGHT_RGB}, 0.85); '
                f'border-radius: 4px; padding: 1px 3px;">{word}</span>'
            )
        else:
            parts.append(word)
        last = s["end"]
    parts.append(html.escape(text[last:]))  # trailing gap after the last token

    return (
        '<div style="line-height: 2.1; font-size: 1rem; word-break: break-word;">'
        + "".join(parts)
        + "</div>"
    )


def _render_message(msg: dict) -> None:
    """Paint one stored turn. Pure over the stored dict so re-rendering the whole
    history on every rerun is cheap and never re-runs inference."""
    with st.chat_message("user"):
        # Body = censored text when a lexicon word is present, original text otherwise.
        st.write(msg["display"])

        if not msg["flagged"]:
            return  # clean comment: no badge, no explanation panel

        st.badge(f"⚠️ Đã lọc · P(độc hại) = {msg['p_toxic']:.0%}", color="red")

        with st.expander("Vì sao bị lọc?"):
            # (a) Original text with the masked words highlighted in place.
            st.markdown("**Bình luận gốc (tô đậm phần độc hại):**")
            st.markdown(
                _highlight_html(msg["text"], msg["spans"]),
                unsafe_allow_html=True,
            )
            # (b) Table of the masked (lexicon) words + their log-odds, strongest first.
            flagged = sorted(
                (s for s in msg["spans"] if s["toxic"]),
                key=lambda s: s["score"],
                reverse=True,
            )
            if flagged:
                st.markdown("**Từ bị che & điểm log-odds:**")
                st.table(
                    [{"Từ": s["word"], "Log-odds": round(s["score"], 3)} for s in flagged]
                )
            else:
                # Flagged by the classifier's overall score, with no single lexicon word.
                st.caption(
                    "Không có từ tục cụ thể để che — mô hình đánh giá TỔNG THỂ bình luận "
                    f"là độc hại (P = {msg['p_toxic']:.0%})."
                )


def _process(text: str, explainer) -> dict:
    """Run inference ONCE per submitted comment and freeze everything the UI needs.

    explain() gives the classifier verdict + per-word spans; censor() gives the masked
    string (it returns text with no lexicon word unchanged). `flagged` — the moderation
    trigger for the badge/panel — is the UI policy composed here: the comment is flagged
    if the classifier blocks it OR any token matched the lexicon (so "Bạn nguuu thế" is
    flagged even though the classifier alone does not block it). Highlight and mask both
    key off span["toxic"], so the panel never contradicts the posted body.
    """
    result = explainer.explain(text)
    return {
        "text": text,
        "display": explainer.censor(text),
        "flagged": result["blocked"] or any(s["toxic"] for s in result["spans"]),
        "p_toxic": result["p_toxic"],
        "spans": result["spans"],
    }


def main() -> None:
    # set_page_config must be the first Streamlit call.
    st.set_page_config(page_title="Bộ lọc bình luận độc hại", page_icon="🛡️")
    st.title("🛡️ Khung chat có kiểm duyệt")
    st.caption(
        "Gõ bình luận tiếng Việt. Bình luận độc hại vẫn được đăng nhưng bị che (***) "
        "kèm cảnh báo; mở “Vì sao bị lọc?” để xem lời giải thích."
    )

    # Load the model up front so a missing artifact fails with a friendly message
    # instead of a stack trace. NOTE: src.explain raises SystemExit (NOT an Exception
    # subclass) when pipeline.pkl / threshold.json are absent, so we must catch that too.
    try:
        explainer = load_explainer()
    except (SystemExit, Exception):  # noqa: B014 — SystemExit isn't an Exception subclass
        st.error("Model chưa sẵn sàng — chạy Stage 4–5 trước.")
        st.stop()

    # Conversation history survives reruns; we re-render every past turn each run.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        _render_message(msg)

    if prompt := st.chat_input("Nhập bình luận..."):
        msg = _process(prompt, explainer)
        st.session_state.messages.append(msg)
        _render_message(msg)  # show the just-added turn now; history loop covers reruns


if __name__ == "__main__":
    main()
