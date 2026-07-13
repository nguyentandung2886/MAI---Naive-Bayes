# Toxic Comment Filter — Bộ lọc bình luận độc hại tiếng Việt

Trong tập **ViHSD**, cứ khoảng **6 bình luận mới có 1 câu độc hại** — lớp độc hại chỉ chiếm
**~17%**. Hệ quả trực tiếp: một model lười biếng chỉ việc đoán "sạch" cho *mọi* câu đã đạt
~83% accuracy mà không bắt được một câu độc nào. Vì thế ở bài toán này **accuracy là con số vô
dụng** — nó thưởng cho đúng cái hành vi ta muốn tránh. Toàn bộ dự án được đo bằng
**precision / recall / F1 trên riêng lớp độc hại** và **PR-AUC**, luôn đặt cạnh baseline "luôn
đoán sạch" (F1 = 0).

Sản phẩm là một khung chat kiểm duyệt real-time: nhập bình luận → model chặn câu độc, **che đúng
từ** thành `***`, và mở được bảng "Vì sao" chỉ ra model đang bắt tín hiệu vào chữ nào. Lõi là
**Naïve Bayes (ComplementNB)** trên **char n-gram**, demo bằng **Streamlit**.

> **Demo trực tuyến:** 🔗 _TODO — điền link Streamlit Community Cloud tại đây sau khi deploy._
> Trong lúc chờ, mục [Cách chạy (local)](#cách-chạy-local) dựng demo trên máy trong vài phút.

---

## Bài toán và vì sao nó khó

Kiểm duyệt bình luận tiếng Việt vấp hai khó khăn cộng dồn.

**Thứ nhất, dữ liệu lệch nặng.** 82.7% câu trong ViHSD là sạch. Một classifier chỉ cần "đoán lớp
đa số" là trông có vẻ giỏi trên accuracy, nhưng vô dụng vì mục tiêu của ta nằm ở đúng cái 17%
thiểu số. Đây là lý do mọi con số bên dưới đều tính riêng cho lớp độc hại.

**Thứ hai, người dùng chủ động né bộ lọc.** Không ai gõ thẳng từ cấm khi biết có filter — họ làm
méo bề mặt của chữ: tách chữ (`đm` → `"Đ m"`), kéo dài (`lồn` → `"lồnnn"`), viết tắt teencode
(`vcl`, `loz`, `ccc`), che sao (`c***`), hay thay số cho chữ (`2l`, `3K`). Một model chỉ thuộc
lòng cách viết chuẩn sẽ trượt sạch những biến thể này. Các ví dụ trên là **hàng thật lấy nguyên
văn từ `train.csv`** (xem `tests/hard_cases.py`), không phải bịa ra.

![Phân bố lớp trong ViHSD](reports/figures/class_dist.png)

*Phân bố lớp: phần độc hại (~17%) nhỏ hơn hẳn phần sạch — nền tảng cho mọi quyết định đo lường phía dưới.*

---

## Cách tiếp cận và vì sao chọn từng thứ

Mỗi lựa chọn dưới đây đều xuất phát từ hai khó khăn trên, không phải theo mặc định.

**Đặc trưng char n-gram (`char_wb`, n=2–4), không phải từ.** Người dùng né chữ ở mức *ký tự*, nên
ta cũng học ở mức ký tự: `"lồnnn"` và `"loz"` vẫn chia sẻ các n-gram con với `"lồn"`, nên biến
dạng không xoá sạch tín hiệu. `char_wb` còn né được cạm bẫy **tách từ tiếng Việt** — vốn hay
chặt sai teencode — vì n-gram không cần biên từ. Chuẩn hoá (`src/preprocessing.py`) cố tình *giữ*
dấu tiếng Việt (bỏ dấu là gộp nhầm "má"/"ma") và *giữ* dấu chấm/sao (`đ.m`, `c***` — chính là tín
hiệu né chữ), chỉ dọn nhiễu thật (URL, HTML, emoji, kéo dài 3+ ký tự).

**ComplementNB, không phải MultinomialNB.** MultinomialNB ước lượng tham số từ chính mỗi lớp nên
bị lớp đa số "kéo"; ComplementNB ước lượng từ *phần bù* của mỗi lớp nên ít thiên vị lớp đa số hơn
— đúng thứ ta cần cho dữ liệu lệch. (Ở mốc chọn model, hai biến thể PR-AUC gần như hoà; ta chốt
ComplementNB vì lý do thiết kế hợp dữ liệu lệch — ghi trong `reports/model_comparison_dev.json`.)

**Đo bằng PR-AUC và F1, bỏ accuracy.** Đã giải thích ở đầu: accuracy bị lớp sạch thống trị.
PR-AUC tóm gọn đánh đổi precision–recall trên lớp dương mà không phụ thuộc một ngưỡng cụ thể; F1
đo tại ngưỡng vận hành.

**Ngưỡng chọn trên dev, test chỉ chạm một lần.** Ngưỡng chặn được dò để **tối đa F1 lớp độc hại
trên tập dev**, rồi *đóng băng*; tập test chỉ được đánh giá đúng **một lần** ở ngưỡng đã chốt
(`reports/metrics.json` ghi lại mốc audit này). Nhờ vậy con số test là ước lượng trung thực, không
bị dò-ngưỡng làm rò rỉ. Ngưỡng vận hành rơi vào **~0.985** — cao bất thường — vì xác suất của NB
bão hoà sát 0/1 (đặc trưng độc lập giả định làm log-odds cộng dồn rất mạnh); chi tiết ở
[Hạn chế đã biết](#hạn-chế-đã-biết).

**Giải thích và che từ dùng chung một nguồn.** Cùng cái log-odds quyết định độc-hay-sạch cũng
quyết định che chữ nào: điểm mỗi từ = tổng toxic log-odds các char n-gram của nó, nên lời giải
thích không bao giờ mâu thuẫn với quyết định chặn. `censor()` chỉ che khi câu thực sự bị chặn, và
chỉ che token có điểm dương — câu sạch trả về nguyên văn.

---

## Kết quả

Mọi số dưới đây lấy nguyên từ `reports/metrics.json` (không làm tròn ở nguồn), đặt cạnh baseline
"luôn đoán sạch". Ngưỡng vận hành = **0.985**.

| Chỉ số (lớp độc hại) | Baseline "luôn đoán sạch" | ComplementNB — dev | ComplementNB — test |
|---|---|---|---|
| Precision | 0.000 | 0.595 | 0.582 |
| Recall    | 0.000 | 0.633 | 0.633 |
| **F1**    | **0.000** | **0.613** | **0.607** |
| PR-AUC    | — | 0.610 | 0.617 |

Đọc bảng: baseline đạt accuracy ~83% nhưng **F1 = 0** trên lớp độc hại — nó không bắt được câu độc
nào. Model đưa F1 từ 0 lên **~0.61**, và điều quan trọng là **dev và test gần như trùng khớp**
(F1 0.613 vs 0.607; PR-AUC 0.610 vs 0.617), tức không có dấu hiệu overfit lên dev.

![Đường Precision–Recall](reports/figures/pr_curve.png)

*Đường PR trên test (PR-AUC ≈ 0.617). Mốc so sánh "no-skill" là đường ngang ≈ 0.17 — đúng bằng tỷ
lệ lớp độc hại — nên diện tích 0.617 là tín hiệu thật, không phải may rủi.*

![Ma trận nhầm lẫn](reports/figures/confusion_matrix.png)

*Ma trận nhầm lẫn tại ngưỡng 0.985: recall ~0.63 (bắt được khoảng 2/3 câu độc) đổi lấy một lượng
chặn oan thấy rõ ở hàng false-positive — chính là hạn chế precision bàn ngay dưới.*

---

## Hạn chế đã biết

Đây là baseline trung thực, không phải hệ thống hoàn hảo — nêu rõ giới hạn để người đọc (và người
phỏng vấn) khỏi phải tự đoán.

- **Precision ~0.58 → chặn oan khoảng 40%.** Ở ngưỡng vận hành, cứ ~10 câu bị chặn thì ~4 câu thực
  ra sạch. Với một khung chat, chặn oan là phiền toái chứ chưa nguy hiểm, nên ta ưu tiên recall;
  nhưng đây là chỗ yếu rõ nhất và là mục tiêu cải thiện đầu tiên.
- **Xác suất NB bão hoà → phải đẩy ngưỡng lên ~0.985.** Vì giả định độc lập, log-odds cộng dồn dồn
  `predict_proba` sát 0 hoặc 1; ở ngưỡng 0.5 mặc định model chặn quá tay. Ngưỡng ~0.985 là cách
  bù lại — hiệu chỉnh xác suất kém là bản chất của NB, không phải lỗi cấu hình.
- **Teencode ngắn vẫn lọt.** Viết tắt kiểu `vl`, `ccc` mang quá ít tín hiệu char-ngram, và vài
  kiểu thay chữ (`dĩ`→`đĩ`) rơi *ngay dưới* ngưỡng. `tests/test_hard_cases.py` **không giấu** điều
  này: các ca đó được ghi bằng `xfail` (đánh dấu "biết trước là trượt") thay vì ép model phải đúng
  — chạy `pytest` sẽ thấy 4 `xfailed`, đó là kỳ vọng, không phải lỗi.

Bù lại, model **đã** hoá giải nhiều kiểu né chữ khác một cách chắc chắn: tách chữ (`Đ m`), kéo dài
(`lồnnn`), che sao (`c***`), và thay số cho chữ (`3K`, `2l`) đều bị chặn — kiểm chứng trong bộ test.

---

## Cách chạy (local)

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt   # bộ đầy đủ cho train / EDA / test

# Lưu ý: app deploy chỉ cài requirements.txt (runtime = load .pkl + phục vụ).
# requirements-dev.txt thêm các gói chỉ dùng khi train/EDA (datasets, matplotlib, pytest).

huggingface-cli login                 # ViHSD là dataset gated — cần đăng nhập + đồng ý điều khoản
python scripts/download_data.py       # tải ViHSD về data/raw/
python src/train.py                   # fit pipeline, xuất models/pipeline.pkl
streamlit run app/streamlit_app.py    # mở demo local
```

Có sẵn `Makefile` gói các lệnh trên: `make data`, `make train`, `make app`, `make test`.

Chạy test: `pytest` (hoặc `make test`). Suite gồm test cho preprocessing, explain/censor, và bộ
input hiểm — **kết quả kỳ vọng là toàn bộ pass, cộng 4 `xfailed`** (các hạn chế teencode/chặn oan
đã ghi ở trên, cố ý đánh dấu để không giả vờ model hoàn hảo).

## Cấu trúc thư mục

| Thư mục | Trách nhiệm |
|---------|-------------|
| `src/` | Lõi thuần Python, test được (preprocessing, train, evaluate, explain, config) |
| `app/` | Chỉ lo giao diện Streamlit — tách hẳn khỏi lõi model |
| `models/` | `pipeline.pkl` (vectorizer + classifier) và `threshold.json` (ngưỡng đã chốt) |
| `reports/` | `metrics.json` + `figures/` — bằng chứng tái lập, nguồn duy nhất cho mọi con số |
| `tests/` | Test hiểm cho preprocessing / inference / explain + fixtures né chữ (`hard_cases.py`) |
| `scripts/`, `notebooks/`, `data/` | Tải dữ liệu, EDA, và dữ liệu thô/phái sinh (không commit) |

Chi tiết đầy đủ từng giai đoạn: xem [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) ở thư mục cha.
