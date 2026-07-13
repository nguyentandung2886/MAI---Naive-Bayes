"""Tests for src.explain (Stage 6): word-level toxicity attribution + censoring.

These pin the Stage 6 contract, all derived from the frozen model's OWN log-odds:
  * explain() ranks the offensive words of a blocked comment highest, and leaves a
    clean comment un-blocked;
  * censor() masks ONLY when the classifier blocks (P(toxic) >= operating_threshold),
    replaces whole toxic-leaning tokens with "***", and returns clean comments verbatim;
  * the model's top toxic n-grams are intuitively offensive substrings;
  * the artifacts load ONCE (module-level cache), not per call.

Ground-truth phrases below were confirmed against the frozen models/pipeline.pkl.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Mirror tests/test_preprocessing.py: put the project root on sys.path so ``src.*``
# imports cleanly regardless of the pytest entry point / cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import censor, explain, top_toxic_ngrams  # noqa: E402

# The offensive prompt from the plan's Stage 6 Definition of Done.
TOXIC_PHRASE = "đồ ngu vãi"
# A polite comment that must sail through untouched.
CLEAN_PHRASE = "chào bạn nhé"
# Blocked, but MIXES toxic ("thằng", "ngu") with a clean vocative particle ("ơi"),
# so censor's selectivity is observable rather than blanket masking.
MIXED_BLOCKED_PHRASE = "thằng ngu ơi"


def test_explain_returns_the_documented_shape() -> None:
    r = explain(TOXIC_PHRASE)
    assert set(r) == {"p_toxic", "blocked", "spans"}
    assert isinstance(r["p_toxic"], float)
    assert isinstance(r["blocked"], bool)
    for span in r["spans"]:
        assert set(span) == {"word", "start", "end", "score", "toxic"}
        assert isinstance(span["score"], float)
        assert isinstance(span["toxic"], bool)


def test_explain_blocks_toxic_phrase_and_ranks_offensive_word_top() -> None:
    r = explain(TOXIC_PHRASE)
    assert r["blocked"] is True
    # The three words all lean toxic here, but the strongest signal is the insult "ngu".
    top = max(r["spans"], key=lambda s: s["score"])
    assert top["word"] == "ngu"
    # Every offensive word is flagged toxic (score > 0).
    by_word = {s["word"]: s for s in r["spans"]}
    assert by_word["ngu"]["toxic"] is True
    assert by_word["vãi"]["toxic"] is True


def test_explain_does_not_block_clean_phrase() -> None:
    r = explain(CLEAN_PHRASE)
    assert r["blocked"] is False
    # A polite sentence: no word should read as toxic.
    assert all(s["toxic"] is False for s in r["spans"])
    assert all(s["score"] <= 0 for s in r["spans"])


def test_explain_span_offsets_index_the_original_string() -> None:
    # Two spaces between the words: offsets must track the ORIGINAL text (for Stage 7
    # highlighting), not the internally-cleaned form which collapses whitespace.
    text = "đồ  ngu"
    r = explain(text)
    assert [s["word"] for s in r["spans"]] == ["đồ", "ngu"]
    for span in r["spans"]:
        assert text[span["start"] : span["end"]] == span["word"]


def test_censor_masks_every_toxic_word_when_blocked() -> None:
    # All three tokens lean toxic, so all three are masked.
    out = censor(TOXIC_PHRASE)
    assert out == "*** *** ***"
    assert "ngu" not in out and "vãi" not in out


def test_censor_is_selective_keeping_clean_words_in_a_blocked_comment() -> None:
    # "thằng" and "ngu" are masked; the clean particle "ơi" survives intact.
    assert censor(MIXED_BLOCKED_PHRASE) == "*** *** ơi"


def test_censor_returns_clean_comment_unchanged() -> None:
    # Gate on the classifier: not blocked -> returned verbatim, nothing masked.
    assert censor(CLEAN_PHRASE) == CLEAN_PHRASE


def test_censor_preserves_original_whitespace() -> None:
    # Whole tokens are replaced but the original spacing (2 then 3 spaces) is kept,
    # proving the mask is mapped back onto the original string by offset.
    assert censor("đồ  ngu   vãi") == "***  ***   ***"


def test_top_toxic_ngrams_are_intuitively_offensive() -> None:
    tops = top_toxic_ngrams(20)
    assert len(tops) == 20
    # char_wb pads word edges with spaces, so strip before comparing to bare tokens.
    stripped = {ng.strip() for ng in tops}
    # A couple of unambiguous Vietnamese profanities must surface at the very top.
    assert "cặc" in stripped
    assert "đcm" in stripped


def test_empty_string_is_safe_and_not_blocked() -> None:
    r = explain("")
    assert r["blocked"] is False
    assert isinstance(r["p_toxic"], float)
    assert r["spans"] == []
    assert censor("") == ""


def test_none_is_guarded_like_empty() -> None:
    r = explain(None)  # type: ignore[arg-type]
    assert r["blocked"] is False
    assert r["spans"] == []
    assert censor(None) == ""  # type: ignore[arg-type]


def test_emoji_only_string_is_safe_and_not_blocked() -> None:
    # clean_text strips emoji, so the token yields no n-grams -> score 0, not blocked.
    r = explain("😡😡😡")
    assert r["blocked"] is False
    assert all(s["score"] == 0.0 for s in r["spans"])
    assert censor("😡😡😡") == "😡😡😡"  # unchanged: not blocked


def test_artifacts_are_not_reloaded_per_call(monkeypatch) -> None:
    # The pipeline/threshold/log-odds map are built ONCE at import (Streamlit calls
    # these per keystroke). Prove a call never re-hits disk: break joblib.load, then
    # exercise both functions — they must still work purely from the module cache.
    import src.explain as ex

    def _boom(*args, **kwargs):
        raise AssertionError("artifact was reloaded on a call — module cache is broken")

    monkeypatch.setattr(ex.joblib, "load", _boom)
    assert ex.explain(TOXIC_PHRASE)["blocked"] is True
    assert ex.censor(TOXIC_PHRASE) == "*** *** ***"
