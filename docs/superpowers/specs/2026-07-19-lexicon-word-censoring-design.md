# Thiết kế: Che từ tục theo lexicon (word-level censoring)

**Ngày:** 2026-07-19
**Trạng thái:** Đã duyệt thiết kế, chờ duyệt spec trước khi lập plan.

## 1. Vấn đề

Người dùng quan sát: câu khá độc như "Bạn nguuu thế" vẫn **không bị chặn**. Điều tra
(systematic-debugging) cho thấy:

- **Không phải lỗi tiền xử lý:** `clean_text` gộp "nguuu" → "ngu" đúng như thiết kế.
- **Ngưỡng 0.9854 không đặt bừa** — nó là điểm **tối đa hóa F1** trên tập dev. Naive
  Bayes coi đặc trưng độc lập nên xác suất bị dồn về hai cực (0/1); vì thế "chính giữa"
  hiệu dụng lại nằm ở ~0.98. Theo thiết kế, ngưỡng này để lọt ~37% câu độc (recall 0.63).
- **"Bạn nguuu thế" đạt P(toxic)=0.914** — dưới ngưỡng vì model học được "bạn"/"thế"
  nghiêng **sạch**, làm loãng tín hiệu độc của từ "ngu".

Yêu cầu thật của người dùng (làm rõ qua brainstorming): **có từ tục trong câu thì che
(`***`) đúng từ đó, giữ nguyên các từ còn lại; có từ bị che thì hiện cảnh báo.**

## 2. Vì sao KHÔNG dùng ngưỡng-điểm-model theo từ

Đã thử nghiệm trên dev: che mọi từ có log-odds ≥ cutoff. Kết quả **che nát câu sạch**
vì log-odds theo từ đo "từ **đi cùng** ngữ cảnh độc", không phải "từ **là** độc":

| Cutoff | % từ sạch bị che | % câu sạch bị chạm | % câu độc bắt được |
|--------|------------------|--------------------|--------------------|
| 4.0    | 5.12%            | 31.1%              | 87.8%              |
| 5.0    | 2.86%            | 19.6%              | 78.6%              |

Ví dụ che nhầm ở cutoff 5.0: "Yêu **mẹ**"(+7.8), "chăn **bò**"(+10.5), "trường **tồn**"(+6.1),
"**dân** ta"(+5.5). Không cutoff nào tách được vì từ độc thật (+5) và từ thường (+6..+13)
trộn lẫn. → **Loại bỏ hướng này.**

## 3. Giải pháp: che theo lexicon + cờ hybrid

### 3.1 Cơ chế khớp

Che khi một token khớp **đúng** (exact, không phải chuỗi con) với `TOXIC_LEXICON` sau khi
chuẩn hóa bằng `clean_text`:

1. Chạy `clean_text(token)` (gộp "nguuu"→"ngu", hạ thường, gộp lặp) → có thể ra ≥1 sub-token.
2. Với mỗi sub-token: bỏ dấu câu bao quanh (giữ chữ cái tiếng Việt + số) và tạo thêm một
   biến thể bỏ `.`/`*` nội bộ (bắt "đ.m"→"đm").
3. Token là "tục" nếu **bất kỳ** biến thể nào thuộc `TOXIC_LEXICON`.

Khớp nguyên token nên **"Nguyễn"/"ngủ"/"nguy" KHÔNG dính "ngu"**. Đã kiểm chứng: false-mask
chỉ **0.30%** số từ trong câu sạch (so với 5.12% của ngưỡng model — ít hơn ~17 lần).

### 3.2 Cờ kiểm duyệt hybrid

- **Che** (`censor`) chỉ dựa trên lexicon — chính xác, không che nhầm.
- **Gắn cờ câu** (badge + panel) = `blocked (classifier) OR có ≥1 từ lexicon`.
- Classifier vẫn bắt các câu độc-theo-ngữ-cảnh (không chứa từ tục tường minh); lexicon bắt
  các câu classifier bỏ sót (như "Bạn nguuu thế"). **Không đụng tới ngưỡng 0.9854.**

Số liệu dev (kiểm chứng):

| Cấu hình | Recall (câu độc) | Tỉ lệ gắn cờ câu sạch |
|----------|------------------|-----------------------|
| Classifier đơn thuần (hiện tại) | 0.633 | 0.095 |
| Lexicon đơn thuần | 0.251 | 0.014 |
| **Hybrid (OR)** | **0.680** | **0.103** |

Thêm lexicon: recall +4.7đđ (bắt thêm 23 câu độc bị sót), chi phí gắn cờ nhầm chỉ +0.8đđ.

## 4. Thay đổi theo file

### 4.1 `src/config.py` — thêm hằng `TOXIC_LEXICON`

Đặt tại đây theo đúng triết lý "mọi hằng số ở một chỗ" của file. Seed từ tục **rõ nghĩa**
(dạng mà `clean_text` sinh ra):

```
địt, đụ, đéo, đéch, đách, cặc, buồi, lồn,
đm, đcm, dcm, đcmm, clm, cmm, cml, vcl, vkl, cứt,
ngu, ngốc, đần, khốn, đĩ, điếm
```

**Cố ý loại** từ đa nghĩa theo ngữ cảnh: `chó, mày, thằng, vãi, mẹ, óc, lol, cc, vl` — để
không che nhầm "con chó dễ thương", "mày ơi", "vãi hàng". **Đây là danh sách người bản ngữ
sẽ tinh chỉnh**; seed chỉ là điểm khởi đầu.

`_FALLBACK_BLACKLIST` trong explain.py **giữ nguyên** — nó phục vụ đường dẫn khác (seed
log-odds map khi classifier không phải mô hình xác suất) và không liên quan tới việc che.
`TOXIC_LEXICON` là hằng số mới, độc lập.

### 4.2 `src/explain.py`

- **Thêm** `_token_is_profane(token: str) -> bool` (cơ chế mục 3.1). Dùng `clean_text` từ
  `src.preprocessing` + `TOXIC_LEXICON` từ `src.config`.
- `explain()`:
  - `blocked` **giữ nguyên nghĩa** = classifier `p_toxic >= _THRESHOLD` (nhất quán metrics.json).
  - Mỗi span: `toxic` = `_token_is_profane(word)` (thay cho `score > 0`).
  - Giữ `p_toxic`, `score` (log-odds, để hiển thị bảng). Shape trả về **không đổi**:
    `{p_toxic, blocked, spans}`, span `{word, start, end, score, toxic}`.
- `censor()`:
  - Che mọi token có `_token_is_profane` = True. Nếu không có token nào → trả **nguyên văn**
    (kể cả khi classifier chặn — badge ở app truyền tải cảnh báo). **Độc lập với `blocked`.**
  - **Bỏ** cơ chế cũ "câu bị chặn phải che ≥1 từ" (nó có thể che nhầm từ vô hại).
- Giữ `_score_word`, `_NGRAM_LOG_ODDS`, `top_toxic_ngrams` (dùng cho bảng log-odds + test).

### 4.3 `app/streamlit_app.py`

- `_process`: thêm `"flagged": result["blocked"] or any(s["toxic"] for s in result["spans"])`.
- `_render_message`: dùng `msg["flagged"]` (thay `msg["blocked"]`) để quyết định hiện badge +
  panel; `display = censor(text)` như cũ.
- `_highlight_html` + bảng "từ bị gắn cờ": dùng `s["toxic"]` (lexicon) thay `s["score"] > 0`,
  để tô/che nhất quán. Tint có thể vẫn theo `score` cho các từ `toxic`.
- Chỉnh nhẹ chữ badge nếu cần (P(độc hại) có thể thấp khi chỉ lexicon gắn cờ).

### 4.4 Tests

- `tests/test_explain.py` — cập nhật hợp đồng:
  - `censor("đồ ngu vãi")` → `"đồ *** vãi"` ("vãi" bị loại khỏi lexicon).
  - `censor("thằng ngu ơi")` → `"thằng *** ơi"` ("thằng" bị loại).
  - `test_censor_preserves_original_whitespace`: "đồ  ngu   vãi" → "đồ  ***   vãi".
  - `by_word["vãi"]["toxic"]` giờ là `False`; `by_word["ngu"]["toxic"]` vẫn `True`.
  - `test_censor_returns_clean_comment_unchanged` ("chào bạn nhé") vẫn nguyên văn.
  - **Thêm** `test_censor_masks_toxic_word_when_classifier_does_not_block`:
    `censor("Bạn nguuu thế") == "Bạn *** thế"` và `explain(...)["blocked"] is False`
    (chốt đúng case người dùng).
  - **Thêm** `test_innocent_high_scoring_word_is_not_masked`:
    `censor("Yêu mẹ") == "Yêu mẹ"` (dù "mẹ" log-odds cao).
- `tests/test_hard_cases.py`:
  - Viết lại `test_censor_gates_on_blocked_and_always_returns_str` sang hợp đồng mới:
    che gắn với **span.toxic**, không phải `blocked`:
    ```
    toxic_spans = [s for s in r["spans"] if s["toxic"]]
    if toxic_spans: assert "***" in out
    else:           assert out == (text or "")
    ```
  - Tier-2 `OBVIOUS_TOXIC`/`OBVIOUS_CLEAN`: kiểm tra lại — tất cả OBVIOUS_TOXIC đều chứa
    từ lexicon (ngu/địt/cặc/đcm/lồn) nên vẫn masked; OBVIOUS_CLEAN không có từ lexicon.
  - Tier-3 xfail giữ nguyên (letter-swap "dĩ"≠"đĩ", teencode "vl"/"ccc" không trong lexicon
    → vẫn xfail; strict=False nên an toàn).

## 5. Tiêu chí hoàn thành (verify)

1. `censor("Bạn nguuu thế") == "Bạn *** thế"`; `censor("Yêu mẹ") == "Yêu mẹ"`.
2. Toàn bộ `pytest` xanh (test cũ đã cập nhật + test mới).
3. Chạy app: "Bạn nguuu thế" hiện "Bạn `***` thế" + badge cảnh báo; "Yêu mẹ" nguyên văn.
4. Không đụng `models/pipeline.pkl`, `models/threshold.json`, `reports/metrics.json`.

## 6. Ngoài phạm vi

- Không đổi ngưỡng classifier 0.9854, không train lại model.
- Không tự động sinh lexicon từ dữ liệu — danh sách do người bản ngữ curate.
- Không xử lý obfuscation ngoài khả năng của `clean_text` (ví dụ letter-swap đ↔d).
