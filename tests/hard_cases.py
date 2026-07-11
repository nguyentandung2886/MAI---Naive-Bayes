"""Seed test cases: REAL character-obfuscation examples pulled verbatim from
data/raw/train.csv.

WHY this file exists (and why it is only DATA, not assertions yet):
Vietnamese users routinely dodge naive keyword filters by mangling the surface
form of a word — spacing letters ("Đ m"), dotting ("đ.m"), teencode ("vcl",
"loz"), elongation ("lồnnn", "êiiiii"), star-censoring ("c***"), or swapping
digits for letters ("2l", "3K"). A model that only memorised clean spellings
will miss these. These strings are the raw material for the "hard" evaluation in
a later stage; here we merely collect them so that work starts from ground truth,
not invented text.

Contract for every entry:
  - ``text``  : an EXACT substring/row taken from train.csv (never fabricated).
  - ``label`` : the binary ground truth, 0=clean / 1=toxic, i.e. (label_id > 0)
                using the same mapping as src.data.
  - ``note``  : which obfuscation technique makes the example hard.

Note the deliberate clean-but-obfuscated rows at the end: obfuscation alone is
NOT proof of toxicity, so the seed set must include distorted text that is still
harmless, to guard against a model that simply flags anything weird-looking.
"""
from __future__ import annotations

# Each dict: {"text": <verbatim row from train.csv>, "label": 0|1, "note": str}
HARD_CASES: list[dict[str, object]] = [
    # --- Dotted / spaced profanity ("đm" written as separated letters) ---
    {"text": "Nói ý thức như đb thì tội cho đb quá. Đ m", "label": 1,
     "note": "spaced profanity 'Đ m' (=đm) + teencode 'đb'"},
    {"text": "d m hải quan k tham nhũng mang đầu tao đi mà chặt", "label": 1,
     "note": "spaced profanity 'd m' (=đm)"},
    {"text": "Đ m bọn phản động này", "label": 1,
     "note": "spaced profanity 'Đ m' (=đm)"},

    # --- Character elongation (repeated letters to dodge exact-match filters) ---
    {"text": "lolzzzz", "label": 1,
     "note": "elongation of 'loz' (=lồn) with repeated 'z'"},
    {"text": "Xaolonnnnnnnnn=)))", "label": 1,
     "note": "elongated 'xạo lồn' run together, repeated 'n'"},
    {"text": "Nguyễn Hoàng Sâm lồnnn Sâm", "label": 1,
     "note": "elongated profanity 'lồnnn'"},
    {"text": "nhắn thêm chữ địt chị mày nữa chất vlon a êiiiii", "label": 1,
     "note": "teencode 'vlon' + vowel elongation 'êiiiii'"},
    {"text": "Đm abe giống bóng vcccc", "label": 1,
     "note": "teencode 'Đm' + elongated 'vcccc' (=vãi cả...)"},

    # --- Teencode abbreviations (initialisms for profanity) ---
    {"text": "Minh mẫn vcl", "label": 1,
     "note": "teencode 'vcl' (=vãi cả lồn)"},
    {"text": "Thầy địt mẹ giả tạo vl =))", "label": 1,
     "note": "teencode 'vl' (=vãi lồn)"},
    {"text": "Đm mặc sịp vào chim to vcl ạ =))))", "label": 1,
     "note": "teencode 'Đm' + 'vcl'"},
    {"text": "Vân Thuỳ xàm vl :))", "label": 1,
     "note": "teencode 'vl'"},
    {"text": "Nguyễn Tài koi ccc bn", "label": 1,
     "note": "teencode initialism 'ccc' + 'bn' (=bạn)"},

    # --- Star-censoring / letter substitution to hide the slur ---
    {"text": "Cay vãi c***", "label": 1,
     "note": "star-censored profanity 'c***'"},
    {"text": "đó là sự tinh tế kèm theo một chút vãi loz.", "label": 1,
     "note": "teencode 'loz' (=lồn) via letter swap"},
    {"text": "Con dĩ nào bên có gì chị ba có đó :)) đẳng ngang với di kkk", "label": 1,
     "note": "'dĩ'/'di' spelling swap for 'đĩ' + teencode laugh 'kkk'"},

    # --- Digit-for-letter / mixed number substitution ---
    {"text": "Tủn 3K nghe nói xạo lồn quá tr", "label": 1,
     "note": "digit substitution '3K' + profanity 'xạo lồn'"},
    {"text": "dm ông nói phét à 🙂 cái vòng cổ đã quá 2l cụ rồi", "label": 1,
     "note": "teencode 'dm' + digit-letter '2l'"},
    {"text": "Dcm fb 1tr fl mà cũng cho bay màu dc,giỏi đấy", "label": 1,
     "note": "teencode initialism 'Dcm' (=địt cụ mày) + digits '1tr'"},

    # --- Clean-but-obfuscated (hard NEGATIVES: distortion != toxicity) ---
    {"text": "Đ m nó lại đúng quá anh êi", "label": 0,
     "note": "distorted 'Đ m' but labeled CLEAN — obfuscation without toxic intent"},
    {"text": "Nhìn chúng nó cmt mà đắng bạn à =)))))))", "label": 0,
     "note": "teencode 'cmt' + elongated emoticon, non-toxic"},
]
