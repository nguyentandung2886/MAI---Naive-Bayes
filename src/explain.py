"""Stage 6 — explainability + censoring, driven by the model's OWN log-odds.

WHY one shared source of truth: the same signal that decides toxic-vs-clean also
decides which words to highlight and mask, so the explanation never contradicts the
block. That signal is the per-n-gram TOXIC LOG-ODDS of the frozen ComplementNB:

    log_odds(ng) = log P(ng | toxic) - log P(ng | clean)   # feature_log_prob_[1] - [0]

A value > 0 means the char n-gram leans toxic. We score a whole word by summing the
log-odds of the char n-grams the model's vectorizer extracts from it.

WHY word-level attribution is clean here: the vectorizer uses analyzer="char_wb",
whose n-grams NEVER cross word boundaries. So running the pipeline's own analyzer on
a single word reproduces exactly that word's n-grams — the same ones the model saw at
fit time (the analyzer also applies clean_text, so teencode like "NGUuuu" is folded to
"ngu" before scoring). No cross-word n-grams to untangle.

WHY censor uses a profanity LEXICON, not the log-odds score: the per-word toxic log-odds
measures "this word CO-OCCURS with toxicity", not "this word IS toxic" — innocent words
like "mẹ", "bò", "dân" score very high because they appear next to profanity in training,
so masking by score censors clean text badly. censor() instead masks a token iff it
matches src.config.TOXIC_LEXICON exactly (after clean_text folds "nguuu" -> "ngu"). This
is independent of the classifier: "Bạn nguuu thế" is masked even though its whole-comment
P(toxic) sits under the block threshold, while "Yêu mẹ" is left untouched. The classifier
verdict still lives in explain()'s `blocked`; the app flags a comment when EITHER fires.

Streamlit calls explain()/censor() per keystroke, so the pipeline, the threshold, and
the {n-gram: log-odds} map are all loaded/built ONCE at import and cached at module
level — rebuilding a 50k-vocab map per call would lag the demo badly.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Running/importing from src/ puts src/ (not the project root) on sys.path, so
# `import src.config` would fail. Add the project root explicitly — mirrors train.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from src.config import PIPELINE_PATH, THRESHOLD_PATH, TOXIC_LEXICON  # noqa: E402
from src.preprocessing import clean_text  # noqa: E402

# --- LAST-RESORT fallback only. The frozen classifier is a Naive Bayes variant and
# ALWAYS exposes feature_log_prob_, so the primary path below is what actually runs.
# This tiny blacklist exists purely so the module still censors *something* if a future,
# non-probabilistic classifier were ever frozen in — it is NOT the feature's basis.
_FALLBACK_BLACKLIST: tuple[str, ...] = ("ngu", "cặc", "đcm", "clm", "cmm", "địt", "lồn")


def _load_pipeline():
    """Load the frozen pipeline. A missing file means Stages 4–5 have not run; fail
    loudly (mirrors src/evaluate.py) rather than fabricate an untrained model."""
    if not PIPELINE_PATH.exists():
        raise SystemExit(
            f"Frozen pipeline not found at {PIPELINE_PATH}. Run Stage 4 (src/train.py) first."
        )
    return joblib.load(PIPELINE_PATH)


def _load_threshold() -> float:
    """Load the dev-tuned operating threshold. Same file src/evaluate.py wrote, so a
    block decision here is IDENTICAL to the one reported in metrics.json."""
    if not THRESHOLD_PATH.exists():
        raise SystemExit(
            f"Threshold not found at {THRESHOLD_PATH}. Run Stage 5 (src/evaluate.py) first."
        )
    return float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["operating_threshold"])


def _build_ngram_log_odds(pipeline) -> dict[str, float]:
    """Map every vocabulary char n-gram to its toxic log-odds (built ONCE).

    Primary path: read the model's feature_log_prob_. Fallback (last resort): if the
    classifier is not probabilistic, seed positive scores from _FALLBACK_BLACKLIST via
    the SAME analyzer, so _score_word's summation logic stays identical either way.
    """
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_log_prob_"):
        log_odds = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        vocab = vec.get_feature_names_out()
        return {str(ng): float(log_odds[i]) for i, ng in enumerate(vocab)}

    analyzer = vec.build_analyzer()
    fallback: dict[str, float] = {}
    for word in _FALLBACK_BLACKLIST:
        for ng in analyzer(word):
            fallback[str(ng)] = 1.0
    return fallback


# --- Module-level cache: built once at import, reused by every call ---
_PIPELINE = _load_pipeline()
_THRESHOLD = _load_threshold()
# The pipeline's own analyzer = clean_text + char_wb extraction, i.e. the EXACT
# tokenization the model was fitted on. Reusing it is what keeps per-word scoring
# aligned with the trained vocabulary.
_ANALYZER = _PIPELINE.named_steps["vec"].build_analyzer()
_NGRAM_LOG_ODDS = _build_ngram_log_odds(_PIPELINE)


def _p_toxic(text: str) -> float:
    """P(toxic) from the frozen pipeline. List-wrap because predict_proba expects an
    iterable of documents; the guard keeps None/"" from crashing the vectorizer."""
    return float(_PIPELINE.predict_proba([text or ""])[:, 1][0])


def _score_word(word: str) -> float:
    """Toxic score of one token = summed toxic log-odds of its char n-grams.

    _ANALYZER applies clean_text + char_wb extraction, so obfuscation ("NGUuuu") is
    normalised exactly as at fit time. char_wb never crosses word boundaries, so a
    single word yields only its own n-grams. Unknown n-grams contribute 0.
    """
    return float(sum(_NGRAM_LOG_ODDS.get(ng, 0.0) for ng in _ANALYZER(word)))


# Everything that is NOT a Vietnamese letter or digit. Stripping these from a cleaned
# token folds trailing punctuation ("ngu!" -> "ngu") and internal obfuscation dots/stars
# ("đ.m" -> "đm", "c*c" -> "cc") onto the bare form we match against TOXIC_LEXICON.
_NON_WORD = re.compile(
    r"[^0-9a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]"
)


def _token_is_profane(word: str) -> bool:
    """True iff an original whitespace token matches the profanity lexicon.

    clean_text normalises the token exactly as at fit time (lowercase, "nguuu" -> "ngu"),
    then we strip every non-letter/digit char and test EXACT membership in TOXIC_LEXICON.
    Matching whole tokens (not substrings) is what keeps "Nguyễn"/"ngủ"/"nguy" clear of
    "ngu". A token that clean_text splits (e.g. around a stripped URL) is profane if ANY
    resulting piece matches.
    """
    return any(_NON_WORD.sub("", sub) in TOXIC_LEXICON for sub in clean_text(word).split())


def explain(text: str) -> dict:
    """Attribute toxicity over the ORIGINAL text.

    Returns p_toxic (P of the toxic class), blocked (p_toxic >= operating_threshold,
    the classifier verdict, same rule as src/evaluate.py), and spans: one dict per
    whitespace token with its original-string offsets, summed-log-odds score (kept for
    the explanation table), and a toxic flag that is a LEXICON match — the same signal
    censor() masks on. start/end index the ORIGINAL string so Stage 7 highlights in place.
    """
    text = text or ""
    p = _p_toxic(text)
    spans = []
    for match in re.finditer(r"\S+", text):
        word = match.group()
        score = _score_word(word)
        spans.append(
            {
                "word": word,
                "start": match.start(),
                "end": match.end(),
                "score": score,
                "toxic": _token_is_profane(word),
            }
        )
    return {"p_toxic": p, "blocked": p >= _THRESHOLD, "spans": spans}


def censor(text: str) -> str:
    """Mask every token that matches the profanity lexicon; leave the rest untouched.

    Independent of the classifier: a token is masked iff _token_is_profane, so a mild
    insult slips no matter how the whole-comment probability lands ("Bạn nguuu thế" ->
    "Bạn *** thế"). If NO token matches the lexicon the text is returned verbatim — even
    when the classifier flags it, we never mask a word we cannot name (the app's badge
    carries that warning instead). Masks are mapped back by offset so the original
    whitespace and non-masked tokens are preserved, and teencode stays a single "***"
    even though clean_text collapses it internally.
    """
    text = text or ""
    matches = list(re.finditer(r"\S+", text))
    mask = [_token_is_profane(m.group()) for m in matches]
    if not any(mask):
        return text

    out: list[str] = []
    last = 0
    for match, flag in zip(matches, mask):
        out.append(text[last : match.start()])  # inter-token whitespace, kept verbatim
        out.append("***" if flag else match.group())
        last = match.end()
    out.append(text[last:])
    return "".join(out)


def top_toxic_ngrams(n: int = 20) -> list[str]:
    """The n char n-grams the model weights most toward toxic (highest log-odds).

    Used by the Stage 6 sanity check and available to Stage 7 for highlighting the
    model's most offensive learned substrings. Returned high-to-low.
    """
    ranked = sorted(_NGRAM_LOG_ODDS.items(), key=lambda kv: kv[1], reverse=True)
    return [ng for ng, _ in ranked[:n]]
