# Báo cáo Giai đoạn 0 + Giai đoạn 1 — Walking Skeleton

> Mục tiêu của 2 giai đoạn này: từ repo trống → một sợi dây chạy suốt end-to-end
> (**data → model → `pipeline.pkl` → Streamlit**), sẵn sàng deploy thành link sống.
> Chất lượng model ở mốc này còn thô **là chủ đích** — ta ưu tiên "chạy được" trước, "chạy tốt" sau.

---

## 1. Đã làm những gì

### Giai đoạn 0 — Khởi tạo & môi trường

| File / thư mục | Vai trò |
|---|---|
| Cây thư mục (`src/`, `app/`, `data/`, `models/`, `reports/`, `tests/`, `scripts/`, `notebooks/`) | Khung chứa mã theo đúng nguyên tắc *separation of concerns* — mỗi thư mục một trách nhiệm |
| `.gitignore` | Chặn commit dữ liệu gated, cache, `.venv`, model trung gian; **giữ lại** đúng `pipeline.pkl` để deploy |
| `requirements.txt` | Khóa (pin) version chính xác 10 package để tái lập được |
| `README.md` | Khung mô tả project + cách chạy local |
| `Makefile` | Gom các lệnh chuẩn (`make data/train/app/test`) về một nguồn sự thật |
| `src/__init__.py` | Đánh dấu `src/` là package để `import src.config` chạy được |
| `src/config.py` | Gom mọi hằng số: `SEED=42`, `NGRAM_RANGE=(2,4)`, `ALPHA=1.0`, `ANALYZER="char_wb"`, và mọi đường dẫn |
| `scripts/download_data.py` | Tải ViHSD (dataset gated) với fallback tự động sang mirror; in `df.shape` để xác nhận |
| Commit đầu `chore: init project skeleton` | Mốc git đầu tiên |

### Giai đoạn 1 — Walking skeleton

| File | Vai trò |
|---|---|
| `src/train.py` | Lấy 2.000 dòng train, gộp nhãn nhị phân thô, fit `Pipeline(CountVectorizer(char_wb) → ComplementNB)` **chỉ trên train**, lưu `models/pipeline.pkl` bằng joblib |
| `models/pipeline.pkl` | **Một** file gói cả vectorizer + classifier (928 KB) — commit vào git để app deploy chạy ngay |
| `app/streamlit_app.py` | UI tối giản: `st.chat_input`, load model qua `@st.cache_resource`, in nhãn `sạch/độc hại` + xác suất |

**Kết quả kiểm thử thực tế:**

- Dữ liệu tải về: `train (24048, 2)`, `dev (2672, 2)`, `test (6680, 2)`; cột thật là `free_text` + `label_id` (0=CLEAN, 1=OFFENSIVE, 2=HATE). Tỉ lệ CLEAN train = 82.7% — khớp mô tả dataset.
- `train.py` → `pipeline.pkl` (vocab 18.804 char n-gram).
- Smoke test hàm `classify`: `"đồ ngu"` → **độc hại** (p=0.999); `"chào bạn"` → **sạch** (p=0.000); ô rỗng / emoji / khoảng trắng → **không crash**.
- `streamlit run` boot thành công (server listening, không lỗi stderr).

---

## 2. Tác dụng — mỗi thành phần giải quyết vấn đề gì

- **Cây thư mục tách bạch:** sửa một phần không đụng phần khác. Sau này đổi UI (`app/`) không cần chạm lõi model (`src/`); đem model sang API khác cũng vậy.
- **`.gitignore`:** tránh 2 tai nạn kinh điển — (a) commit nhầm dữ liệu gated (vi phạm license), (b) repo phình to vì data/model rác. Nhưng vẫn cho `pipeline.pkl` đi theo để link deploy sống được mà không cần tải lại data.
- **`requirements.txt` pin version:** đảm bảo máy bạn, git, và Streamlit Cloud dùng **cùng một** bộ version → `.pkl` load được ở mọi nơi.
- **`config.py`:** một chỗ duy nhất chứa hằng số → chạy lại luôn ra kết quả cũ (reproducibility), và muốn tinh chỉnh chỉ sửa một nơi.
- **`download_data.py`:** biến "dataset gated khó tải" thành "một lệnh"; ai clone repo về cũng dựng lại được data mà không cần bạn gửi file.
- **`train.py`:** chứng minh mạch `data → model → file` thông suốt; là chỗ để các giai đoạn sau *thay từng miếng* (preprocessing thật, tune, threshold).
- **`pipeline.pkl` một file:** inference chỉ cần load 1 object → không lệch tiền xử lý giữa lúc train và lúc phục vụ (*train–serve skew*).
- **`streamlit_app.py`:** đầu ra người dùng chạm vào — bằng chứng "hệ thống thật sự phân loại được một câu tiếng Việt".

---

## 3. Tại sao cần làm — lý do kỹ thuật của từng quyết định

**Vì sao gói vào MỘT `Pipeline` và lưu MỘT `.pkl`?**
Vectorizer và classifier phải đi cùng nhau: cùng một vocabulary, cùng cách biến chữ → số. Nếu lưu rời hai file, rất dễ ở lúc serve dùng vectorizer "lệch pha" với model → kết quả sai âm thầm. Gói chung → inference chỉ `pipeline.predict_proba([text])`, không thể lệch.

**Vì sao `analyzer="char_wb"` (char n-gram) chứ không phải word n-gram?**
Mạng xã hội tiếng Việt đầy né chữ: `"đ.m"`, `"nguuuu"`, `"v@i"`. Word n-gram coi mỗi biến thể là một từ mới lạ → trượt. Char n-gram bắt pattern ở cấp **ký tự con** nên bền với biến dạng, và **không cần word segmentation** (underthesea/pyvi) — vốn là rào cản lớn nhất của NLP tiếng Việt và hay sai chính trên teencode. `_wb` = không vượt ranh giới từ, tránh học những cụm ký tự vô nghĩa bắc cầu qua khoảng trắng.

**Vì sao `ComplementNB` chứ không `MultinomialNB`?**
Dữ liệu lệch nặng (82.7% CLEAN). MultinomialNB ước lượng tham số từ chính mỗi lớp nên bị lớp đa số "kéo". ComplementNB ước lượng từ **phần bù** của mỗi lớp → ít thiên vị lớp đa số hơn → hợp dữ liệu mất cân bằng như ViHSD.

**Vì sao `alpha=1.0`?**
Laplace smoothing: một n-gram chỉ xuất hiện ở lớp này mà không ở lớp kia sẽ cho xác suất 0, làm hỏng phép nhân xác suất của Naïve Bayes. `alpha=1.0` cộng thêm một lượng nhỏ để không bao giờ có xác suất 0.

**Vì sao pin version trong `requirements.txt`?**
File `.pkl` là pickle của các object scikit-learn. scikit-learn **không** đảm bảo pickle lưu ở version này load được ở version khác. Nếu lúc train là 1.7.2 mà Streamlit Cloud cài 1.9.0, app có thể vỡ ngay khi `joblib.load`. Pin `==` khóa toàn bộ chuỗi về một version đã kiểm chứng.

**Vì sao `@st.cache_resource` (không phải `@st.cache_data`)?**
Streamlit chạy lại **toàn bộ** script mỗi lần người dùng tương tác. Không cache thì mỗi lần gõ phím sẽ `joblib.load` lại file `.pkl` → lag nặng. `cache_resource` giữ model **một lần duy nhất** qua mọi rerun. Dùng `cache_resource` (không phải `cache_data`) vì model là **tài nguyên dùng chung, không serialize lại mỗi lần** — đúng ngữ nghĩa của cache resource.

**Vì sao fit vectorizer BÊN TRONG Pipeline?**
Để chống *data leakage*. Vì vectorizer nằm trong Pipeline, `.fit()` chỉ học vocabulary từ **train**. Dev/test về sau chỉ `.transform()` — không bao giờ "nhìn trộm" dữ liệu chưa được phép thấy. Nếu fit vectorizer trên toàn bộ data trước khi chia, số liệu sẽ đẹp giả và sụp khi lên thật.

**Vì sao tách hàm `classify(text, model)` thuần khỏi UI?**
Để **test được** mà không cần bật server Streamlit. Truyền `model` vào như tham số → smoke test chỉ cần `joblib.load` rồi gọi hàm, bỏ qua toàn bộ runtime của Streamlit.

---

## 4. Kiến thức thực chiến rút ra — để LẦN SAU TỰ LÀM được

Đây là phần quan trọng nhất. Không phải chép lệnh, mà là **nguyên tắc tư duy**:

**① Walking skeleton — dựng khung chạy suốt trước, đẹp sau.**
Sai lầm phổ biến của người mới: làm tuần tự kiểu waterfall (EDA thật kỹ → preprocessing thật xịn → model thật tốt → *cuối cùng mới ráp UI*). Rủi ro dồn hết vào ngày cuối: "ráp mọi thứ lại" luôn lòi ra hàng loạt lỗi tích hợp khi không còn thời gian. Cách đúng: làm một **lát cắt dọc** mỏng nhất chạy được từ đầu đến cuối trước (dù model chỉ 2000 dòng, dù nhãn gộp thô), rồi **thay từng miếng** trong khung đã sống. Luôn có một bản "chạy được" để lùi về.

**② Deploy sớm để lộ lỗi môi trường sớm.**
Lỗi tệ nhất là lỗi chỉ xuất hiện trên môi trường deploy (thiếu package, sai version, sai đường dẫn). Nếu để deploy đến cuối, bạn gặp chúng đúng lúc áp lực nhất. Deploy ngay từ skeleton → phát hiện khi còn rẻ để sửa. (Đây là lý do `pipeline.pkl` được commit: link deploy chạy được **không phụ thuộc** việc Cloud có tải được data gated hay không.)

**③ Separation of concerns — mỗi thứ một chỗ, một trách nhiệm.**
`src/` = logic thuần Python test được; `app/` = chỉ giao diện; `data/raw` = nguồn bất khả xâm phạm; `data/processed` = phái sinh có thể sinh lại; `notebooks/` = nơi khám phá lộn xộn, **code chín thì chuyển vào `src/`** chứ không để logic quan trọng mắc kẹt trong notebook. Ranh giới rõ → sửa một phần không sợ vỡ phần khác.

**④ Reproducibility — chạy lại phải ra đúng kết quả cũ.**
Ba trụ cột: (a) **seed cố định** ở một nơi (`config.SEED`) import đi khắp nơi; (b) **pin version** để môi trường không trôi; (c) **không commit data/model rác**, thay bằng script dựng lại. Người chấm (recruiter) tin con số của bạn chỉ khi họ có thể chạy lại ra đúng con số đó.

**⑤ Đọc dữ liệu THẬT trước khi hardcode.**
Ta đã đoán đúng cột là `free_text`/`label_id` — nhưng vẫn `print(df.columns)` để **xác nhận** thay vì tin. Rất nhiều bug âm thầm đến từ hardcode `text`/`label` trong khi thật ra tên khác. Luôn nhìn `df.columns` + `df.head()` trước.

**⑥ Test cái lõi, không test cái khung.**
Ta tách `classify()` ra để test được nó mà không cần Streamlit. Nguyên tắc: đẩy logic quan trọng ra khỏi framework để nó **thuần** và **test được** — framework (UI) chỉ là lớp vỏ mỏng gọi vào lõi.

---

## 5. Bước tiếp theo — Giai đoạn 2

GĐ2 (EDA & chốt schema, Ngày 3–4) sẽ:

- Tải & dùng đúng 3 split chính thức, **không tự chia lại** (giữ tính so sánh + tránh leakage).
- Viết `src/data.py`: một hàm tập trung load + map cột + gộp nhãn nhị phân (thay cho code gộp thô đang nằm trong `train.py`).
- `notebooks/01_eda.ipynb`: phân bố nhãn, phân bố độ dài câu, top char n-gram theo lớp.
- Lọc **≥ 15 ví dụ biến dạng ký tự thật** trong data (teencode, gõ sai, né chữ) → làm seed cho test hiểm ở `tests/`.

Sau GĐ2 mới tới preprocessing thật (GĐ3), pipeline chính thức + tune (GĐ4), threshold + đánh giá test 1 lần (GĐ5).

---

*Ghi chú môi trường: các package đã cài đúng version pin nhưng vào **python global** thay vì `.venv` (lúc `pip install` chưa activate venv). Không ảnh hưởng chức năng vì version khớp; nếu muốn sạch, có thể activate `.venv` rồi cài lại `requirements.txt`.*
