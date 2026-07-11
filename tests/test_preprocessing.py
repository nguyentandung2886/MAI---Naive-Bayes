"""Tests for src.preprocessing.clean_text.

These pin the Stage 3 contract: clean JUST ENOUGH while preserving the signals that
matter (Vietnamese diacritics, obfuscation punctuation) and dropping only true noise
(URLs, HTML, emoji, elongation, stray whitespace).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Mirror src/data.py: put the project root on sys.path so both ``src.*`` and the
# ``tests`` package import cleanly no matter the pytest entry point / cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import clean_text  # noqa: E402
from tests.hard_cases import HARD_CASES  # noqa: E402


def test_keeps_vietnamese_diacritics() -> None:
    # Diacritics are the distinguishing signal: "má" must NOT become "ma".
    out = clean_text("MÁ")
    assert "á" in out
    assert out != "ma"
    assert clean_text("Cà Phê Sữa Đá") == "cà phê sữa đá"


def test_definition_of_done_case() -> None:
    # The Stage 3 DoD anchor: lowercased, diacritics kept, URL + emoji gone,
    # "nguuuu" -> "ngu", "vãiii" -> "vãi".
    out = clean_text("Đồ NGUuuu vãiii 😡 http://x.co")
    assert out == "đồ ngu vãi"
    assert out == out.lower()          # lowercased
    assert "ồ" in out                  # diacritics preserved
    assert "http" not in out           # URL removed
    assert "x.co" not in out
    assert "😡" not in out             # emoji removed
    assert "nguuuu" not in out and "ngu" in out
    assert "vãiii" not in out and "vãi" in out


def test_collapses_runs_of_three_or_more() -> None:
    assert clean_text("nguuuu") == "ngu"
    assert clean_text("cooool") == "col"      # run of 4 'o' -> 1


def test_double_letters_untouched() -> None:
    # A run of exactly two is a legitimate spelling and must survive intact.
    assert clean_text("cool") == "cool"
    assert clean_text("book") == "book"


def test_empty_string() -> None:
    assert clean_text("") == ""


def test_emoji_only_string() -> None:
    # An emoji-only comment must reduce to "" without raising.
    assert clean_text("😡😡😡") == ""


def test_nfc_equivalence() -> None:
    # Precomposed "à" (U+00E0) vs base 'a' + combining grave (U+0300) must clean
    # to the SAME output after NFC normalisation.
    precomposed = "à"
    decomposed = "à"
    assert precomposed != decomposed  # different byte sequences going in
    assert clean_text(precomposed) == clean_text(decomposed)


def test_idempotent() -> None:
    # Running clean twice must equal running it once (safe to re-apply).
    samples = [
        "Đồ NGUuuu vãiii 😡 http://x.co",
        "Cay vãi c***",
        "  spaced   OUT  text  ",
        "",
    ]
    for s in samples:
        once = clean_text(s)
        assert clean_text(once) == once


def test_hard_cases_do_not_raise() -> None:
    # Sanity on REAL obfuscated rows: clean_text must never raise and always
    # return a str for the ground-truth seed set.
    for case in HARD_CASES:
        out = clean_text(case["text"])  # type: ignore[arg-type]
        assert isinstance(out, str)
