# Toxic Comment Filter — Khung chat lọc bình luận độc hại tiếng Việt

Bộ lọc real-time cho bình luận tiếng Việt: chặn tin độc hại, che từ nhạy cảm thành `*`,
và chỉ ra model đang "bắt lỗi" vào chữ nào. Model chính là **Naïve Bayes (ComplementNB)**
trên **char n-gram**, dữ liệu **ViHSD**, demo bằng **Streamlit**.

> Trạng thái: `v0.1-skeleton` — walking skeleton chạy end-to-end (data → model → `.pkl` → Streamlit).
> Chất lượng model ở mốc này còn thô; các giai đoạn sau sẽ thay từng miếng để cải thiện.

## Kiến trúc thư mục (tóm tắt)

| Thư mục | Trách nhiệm |
|---------|-------------|
| `src/` | Logic thuần Python, test được, tái dùng được (config, train, ...) |
| `app/` | Chỉ lo giao diện Streamlit — tách hẳn khỏi lõi model |
| `scripts/` | Tiện ích chạy tay, ví dụ tải dữ liệu |
| `data/` | `raw/` (nguồn bất khả xâm phạm) và `processed/` (phái sinh) — **không commit** |
| `models/` | `pipeline.pkl`: một file gói cả vectorizer + classifier |
| `reports/` | Hình và `metrics.json` — bằng chứng tái lập |
| `tests/` | Test hiểm cho preprocessing / inference / explain |
| `notebooks/` | Khám phá (EDA, error analysis) |

## Cách chạy (local)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

huggingface-cli login                 # ViHSD là dataset gated — cần đăng nhập + đồng ý điều khoản
python scripts/download_data.py       # tải ViHSD về data/raw/
python src/train.py                   # fit pipeline, xuất models/pipeline.pkl
streamlit run app/streamlit_app.py    # mở demo local
```

## Nguyên tắc thiết kế

- **Skeleton trước, tính năng sau** — luôn có bản end-to-end chạy được.
- **Chống data leakage** — vectorizer nằm trong Pipeline, chỉ `fit` trên train.
- **Không nhìn accuracy** — dữ liệu lệch ~82.7% CLEAN; dùng PR-AUC + F1 lớp độc hại.
- **Ký tự, không phải từ** — `analyzer="char_wb"`, giữ nguyên dấu tiếng Việt.

Chi tiết đầy đủ: xem [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) ở thư mục cha.
