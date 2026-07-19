"""Tests for src.explain (Stage 6): word-level toxicity attribution + censoring.

These pin the Stage 6 contract:
  * explain() reports the classifier verdict in `blocked` (P(toxic) >= threshold) and
    still ranks words by the model's log-odds `score`; a span's `toxic` flag is now a
    LEXICON match (src.config.TOXIC_LEXICON), not `score > 0`;
  * censor() masks every token matching the profanity lexicon — regardless of the
    classifier verdict — and returns text with no lexicon word verbatim. This is what
    lets "Bạn nguuu thế" be masked even though the classifier alone does not block it,
    while an innocent high-log-odds word like "mẹ" is left untouched;
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
    # The strongest log-odds signal is still the insult "ngu".
    top = max(r["spans"], key=lambda s: s["score"])
    assert top["word"] == "ngu"
    # `toxic` is now a lexicon match: "ngu" is in TOXIC_LEXICON, "vãi" is deliberately
    # NOT (context-ambiguous slang), so only "ngu" is flagged for masking.
    by_word = {s["word"]: s for s in r["spans"]}
    assert by_word["ngu"]["toxic"] is True
    assert by_word["vãi"]["toxic"] is False


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


def test_censor_masks_only_lexicon_words() -> None:
    # Only "ngu" is in TOXIC_LEXICON; "đồ" and "vãi" are not, so they survive.
    out = censor(TOXIC_PHRASE)
    assert out == "đồ *** vãi"
    assert "ngu" not in out


def test_censor_is_selective_keeping_non_lexicon_words() -> None:
    # "ngu" is masked; "thằng" (rude pronoun, excluded as context-ambiguous) and the
    # clean particle "ơi" survive intact.
    assert censor(MIXED_BLOCKED_PHRASE) == "thằng *** ơi"


def test_censor_masks_toxic_word_when_classifier_does_not_block() -> None:
    # THE reported case: the classifier alone does NOT block this (polite framing pulls
    # P(toxic) under threshold), yet "nguuu" -> clean_text -> "ngu" is in the lexicon, so
    # the word is masked anyway. This is the whole point of lexicon-driven censoring.
    assert explain("Bạn nguuu thế")["blocked"] is False
    assert censor("Bạn nguuu thế") == "Bạn *** thế"


def test_innocent_high_log_odds_word_is_not_masked() -> None:
    # "mẹ" carries a high per-word log-odds (it co-occurs with "địt mẹ" in training) but
    # is perfectly innocent here. It is NOT in the lexicon, so it must never be masked.
    assert censor("Yêu mẹ") == "Yêu mẹ"


def test_censor_returns_clean_comment_unchanged() -> None:
    # No lexicon word present -> returned verbatim, nothing masked.
    assert censor(CLEAN_PHRASE) == CLEAN_PHRASE


def test_censor_preserves_original_whitespace() -> None:
    # The masked token is replaced but the original spacing (2 then 3 spaces) is kept,
    # proving the mask is mapped back onto the original string by offset.
    assert censor("đồ  ngu   vãi") == "đồ  ***   vãi"


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
    assert ex.censor(TOXIC_PHRASE) == "đồ *** vãi"
