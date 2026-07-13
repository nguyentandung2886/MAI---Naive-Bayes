"""Tests for src.explain (Stage 8): adversarial robustness + HONEST limitations.

Three tiers, each with a different promise:

  (1) CONTRACT / robustness — MUST PASS on ANY input. Nothing crashes, explain()
      always returns the documented shape, every span slices the ORIGINAL string back
      to its exact token, and censor() mirrors the block decision. This tier is
      parametrized over every REAL obfuscated row in tests/hard_cases.py plus a set of
      pathological stress strings (empty, None, emoji-only, zero-width/fullwidth
      unicode, a very long paragraph, whitespace-only, embedded newlines, mega-repeat).

  (2) OBVIOUS-SIGNAL — MUST PASS. Bare profanity is blocked and masked; plainly clean
      text is not — including "đồ ăn ngon quá", which shares the token "đồ" with the
      insult "đồ ngu" yet is harmless (guards against context-blind over-blocking).

  (3) KNOWN-LIMITATION — xfail(strict=False). Real toxic rows the char-ngram NB
      baseline is EXPECTED to miss (short teencode initialisms, letter-swaps landing
      just under threshold) and one clean row it over-blocks. We DOCUMENT the gap with
      xfail rather than assert the model is perfect on every adversarial case — that
      would be flaky and dishonest. strict=False so a future model that fixes one of
      these XPASSes without failing the suite.

Mirrors tests/test_explain.py: sys.path insert so `src.*` imports cleanly, and reuses
the tests/hard_cases.py fixtures rather than re-typing adversarial strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Put the project root on sys.path so `src.*` imports cleanly regardless of the pytest
# entry point / cwd (same as tests/test_explain.py, tests/test_preprocessing.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain import censor, explain  # noqa: E402
from tests.hard_cases import HARD_CASES  # noqa: E402


def _fixture(substr: str) -> dict:
    """Pull the single HARD_CASES row whose text contains `substr` (fails loudly on
    0 or >1 matches), so limitation assertions reference the REAL corpus rather than
    re-typed strings."""
    hits = [c for c in HARD_CASES if substr in c["text"]]  # type: ignore[operator]
    assert len(hits) == 1, f"expected exactly one fixture containing {substr!r}, got {len(hits)}"
    return hits[0]


# --- Stress inputs: pathological strings that must never crash the inference API.
# Kept separate from HARD_CASES (which are REAL train.csv rows) — these probe encoding
# and size edges rather than obfuscation.
STRESS_INPUTS = [
    pytest.param("", id="empty"),
    pytest.param(None, id="none"),
    pytest.param("😡😡😡🔥", id="emoji-only"),
    pytest.param("n​g​u", id="zero-width-spaces"),
    pytest.param("ｎｇｕ 𝕟𝕘𝕦 n̸g̸u̸", id="fullwidth-and-combining"),
    pytest.param("đồ ngu vãi " * 500, id="very-long-paragraph"),
    pytest.param("   \t\n   ", id="whitespace-only"),
    pytest.param("dòng một\nđồ ngu vãi\ndòng ba", id="embedded-newlines"),
    pytest.param("chào bạn nhé đồ ngu vãi cảm ơn", id="mixed-clean-and-toxic"),
    pytest.param("a" * 300, id="mega-repeat"),
]

# Every real obfuscated row + every stress string — the full robustness surface.
ALL_INPUTS = [
    pytest.param(c["text"], id=f"hard{i:02d}") for i, c in enumerate(HARD_CASES)
] + STRESS_INPUTS


# ----------------------------------------------------------------------------------
# Tier 1 — CONTRACT / robustness (must pass on ANY input)
# ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ALL_INPUTS)
def test_explain_never_crashes_and_returns_documented_shape(text) -> None:
    r = explain(text)
    assert set(r) == {"p_toxic", "blocked", "spans"}
    assert isinstance(r["p_toxic"], float)
    assert isinstance(r["blocked"], bool)
    assert isinstance(r["spans"], list)
    for span in r["spans"]:
        assert set(span) == {"word", "start", "end", "score", "toxic"}
        assert isinstance(span["word"], str)
        assert isinstance(span["score"], float)
        assert isinstance(span["toxic"], bool)


@pytest.mark.parametrize("text", ALL_INPUTS)
def test_every_span_indexes_the_original_text(text) -> None:
    # start/end must slice the ORIGINAL string back to the exact token (Stage 7 highlights
    # in place, not on the internally-cleaned form) — for weird unicode and long inputs alike.
    src = text or ""
    for span in explain(text)["spans"]:
        assert src[span["start"] : span["end"]] == span["word"]


@pytest.mark.parametrize("text", ALL_INPUTS)
def test_censor_gates_on_blocked_and_always_returns_str(text) -> None:
    # The moderation contract: censor mirrors the block decision. Not blocked -> text is
    # returned verbatim; blocked -> at least one token is masked to "***".
    r = explain(text)
    out = censor(text)
    assert isinstance(out, str)
    if r["blocked"]:
        assert "***" in out
    else:
        assert out == (text or "")


# ----------------------------------------------------------------------------------
# Tier 2 — OBVIOUS-SIGNAL (must pass)
# ----------------------------------------------------------------------------------

# Bare, unambiguous profanity — the plainest possible positive signal. Not from the
# adversarial corpus; the model MUST block and mask these.
OBVIOUS_TOXIC = ["đồ ngu vãi", "thằng ngu", "địt mẹ mày", "cặc", "đcm", "lồn"]

# Plainly clean text — must sail through untouched. "đồ ăn ngon quá" is the key case:
# it shares the token "đồ" with the insult "đồ ngu" yet is harmless.
OBVIOUS_CLEAN = [
    "chào bạn nhé",
    "hôm nay trời đẹp quá",
    "cảm ơn bạn rất nhiều",
    "đồ ăn ngon quá",
    "bài viết hay lắm",
]


@pytest.mark.parametrize("text", OBVIOUS_TOXIC)
def test_bare_profanity_is_blocked_and_masked(text) -> None:
    assert explain(text)["blocked"] is True
    assert "***" in censor(text)


@pytest.mark.parametrize("text", OBVIOUS_CLEAN)
def test_plainly_clean_text_is_not_blocked_or_censored(text) -> None:
    assert explain(text)["blocked"] is False
    assert censor(text) == text


def test_obfuscated_but_clean_row_is_not_over_blocked() -> None:
    # The fixture set's own guard: obfuscation alone is NOT toxicity. This distorted-but-
    # harmless real row ('Đ m nó lại đúng quá anh êi', labeled CLEAN) must not be blocked —
    # proving the model does not simply flag anything weird-looking.
    row = _fixture("nó lại đúng quá")
    assert row["label"] == 0
    assert explain(row["text"])["blocked"] is False


# ----------------------------------------------------------------------------------
# Tier 3 — KNOWN-LIMITATION (xfail: documented, not asserted-correct)
# ----------------------------------------------------------------------------------

# Real toxic rows the char-ngram NB baseline still lets through: short teencode
# initialisms ('vl', 'ccc') carry too little char-signal, and a letter-swap ('dĩ'->'đĩ')
# lands just under the 0.985 threshold. Documented as xfail — NOT asserted-correct.
KNOWN_TOXIC_MISSES = [
    pytest.param(_fixture("Vân Thuỳ")["text"], id="teencode-vl"),
    pytest.param(_fixture("Nguyễn Tài koi")["text"], id="teencode-ccc"),
    pytest.param(_fixture("Con dĩ")["text"], id="letter-swap-di"),
]


@pytest.mark.xfail(reason="NB char-ngram baseline misses this obfuscation", strict=False)
@pytest.mark.parametrize("text", KNOWN_TOXIC_MISSES)
def test_known_toxic_obfuscations_that_baseline_misses(text) -> None:
    assert explain(text)["blocked"] is True


@pytest.mark.xfail(reason="NB over-blocks this clean-but-heated row (precision ~0.58)", strict=False)
def test_known_false_positive_on_a_clean_row() -> None:
    # The flip side of a 0.985 threshold at ~0.58 precision: some clean-but-heated rows
    # trip the filter. This real ViHSD row is labeled CLEAN yet gets blocked.
    row = _fixture("cmt mà đắng")
    assert row["label"] == 0
    assert explain(row["text"])["blocked"] is False
