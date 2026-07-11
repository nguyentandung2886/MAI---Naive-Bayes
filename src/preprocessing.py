"""Text cleaning for the toxic-comment filter: a single, reusable ``clean_text``.

WHY a dedicated module: the exact same normalisation must run at TRAIN time and at
PREDICT time. If the two ever diverge, the model sees a different surface form than
it was fitted on and silently degrades. Centralising the steps here — pure and
deterministic — makes that impossible.

Design intent (see plan Stage 3): clean JUST ENOUGH to canonicalise noise while
PRESERVING the signals that distinguish toxic from clean text. Concretely we keep
Vietnamese diacritics (dropping them collapses "má"/"ma" and destroys meaning) and
keep punctuation such as ".", "*", "@" (these ARE the obfuscation signal, e.g.
"đ.m", "c***"); stripping them would erase the very features Stage 4's char n-grams
learn from. No word segmentation either — char_wb n-grams recover sub-word cues
without a fragile tokenizer that mis-splits teencode.
"""
from __future__ import annotations

import re
import unicodedata

# --- Precompiled patterns (compiled once at import; clean_text stays hot & pure) ---

# URLs first so their punctuation ("://", ".") never leaks into later steps. Match
# both scheme-qualified links and bare "www." hosts up to the next whitespace.
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Minimal HTML tag stripper: scraped comments occasionally carry markup like
# "<br>". We only remove the tags, keeping any inner text.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Emoji / pictographic characters, by Unicode block. WHY explicit ranges instead of
# the `emoji` package: no extra dependency, fully deterministic, and easy to audit.
# Covered: emoticons, misc symbols & pictographs, transport, supplemental symbols,
# symbols & pictographs extended-A, misc symbols, dingbats, regional-indicator
# flags, plus the zero-width joiner and variation selectors that glue emoji together.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons (e.g. 😡)
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flags)
    "\U0000200D"             # zero-width joiner (emoji sequences)
    "\U0000FE00-\U0000FE0F"  # variation selectors (emoji vs text presentation)
    "]",
    flags=re.UNICODE,
)

# Collapse any run of the SAME character repeated 3+ times down to a single one.
# WHY threshold 3 (not 2): a run of exactly two is often a legitimate Vietnamese or
# loanword spelling ("cool", "book", double letters), so we must not touch it; runs
# of 3+ are elongation used to dodge exact-match filters ("nguuuu", "vãiii").
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.DOTALL)

# Whitespace collapser: any run of whitespace becomes a single ASCII space.
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise a raw comment into its canonical cleaned form.

    Pure and deterministic. Empty input — and input that reduces to nothing (e.g. an
    emoji-only string) — returns "" without raising. STEP ORDER MATTERS and each
    step below explains why it comes where it does.
    """
    # Guard: empty (or falsy) input has nothing to clean. Returning "" here also
    # keeps the function safe on None, though callers are expected to pass str.
    if not text:
        return ""

    # (a) Unicode NFC FIRST. Vietnamese accented letters can exist either as a single
    # precomposed code point ("à") or as base + combining mark ("a" + U+0300). Left
    # unmerged, those two byte sequences would tokenize as different characters and
    # split what should be one feature. Normalising up front makes every later step
    # operate on one canonical representation.
    text = unicodedata.normalize("NFC", text)

    # (b) Lowercase. str.lower() is Unicode-aware and safe for Vietnamese, folding
    # "NGUuuu"/"nguuuu" together so casing is not a spurious distinction.
    text = text.lower()

    # (c) Remove URLs and HTML tags. Done BEFORE emoji/repeat handling so their inner
    # punctuation and repeated characters (e.g. "//", "..") cannot survive as noise.
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)

    # (d) Remove emoji / pictographs. They carry no lexical signal for this filter and
    # would otherwise inflate the feature space. Deleting them here (before repeat
    # collapse) also means an emoji-only comment cleanly reduces toward "".
    text = _EMOJI_RE.sub(" ", text)

    # (e) Collapse 3+ character elongation to a single character. Applied AFTER emoji
    # removal (so we never collapse emoji runs) and BEFORE whitespace normalisation.
    # This also collapses repeated punctuation ("c***" -> "c*", "=))))" -> "=)"),
    # which still preserves a single marker of the obfuscation for the n-grams.
    text = _REPEAT_RE.sub(r"\1", text)

    # (f) Normalise whitespace last: every substitution above injected spaces, so we
    # squeeze runs to one space and strip the ends only once, at the end.
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
