# Kịch bản demo — Toxic Comment Filter

Sáu câu, xếp theo độ khó tăng dần. Ý đồ là dẫn người xem đi từ "model hoạt động" đến "model
hoạt động *đúng cách*": bắt được câu độc kể cả khi bị né chữ, che đúng từ mà không đụng từ sạch,
và — quan trọng nhất — **không chặn oan** một câu sạch chỉ vì nó "trông giống" câu độc.

Tất cả câu dưới đây đã chạy thật trên `pipeline.pkl` đang đóng băng, nên kết quả kỳ vọng là
những gì demo sẽ hiển thị (ngưỡng vận hành = 0.985). Gõ từng câu vào ô chat, `Enter`, rồi nói
đúng một câu "muốn cho thấy gì" trước khi sang câu tiếp theo.

---

### 1. Câu sạch thường — thiết lập mốc "bình thường"

> **Câu:** `hôm nay trời đẹp quá, cảm ơn mọi người`
>
> **Kỳ vọng:** đăng bình thường, hiện nguyên văn, **không** badge, **không** che.
>
> **Muốn cho thấy:** bộ lọc không phải cái máy chặn bừa — câu sạch đi thẳng, không ma sát. Đây là
> mốc để mọi phản ứng phía sau có nghĩa.

### 2. Chửi thẳng — chặn và che

> **Câu:** `đồ ngu vãi`
>
> **Kỳ vọng:** bị chặn, hiện badge độc hại, nội dung che thành `*** *** ***`.
>
> **Muốn cho thấy:** tín hiệu rõ thì model bắt dứt khoát, và **che chứ không xoá** — người đọc
> biết có gì đó bị kiểm duyệt mà không phải đọc lại từ bẩn.

### 3. Né chữ (che sao) — vẫn bắt

> **Câu:** `Cay vãi c***`
>
> **Kỳ vọng:** vẫn bị chặn; che thành `Cay *** ***` — giữ `Cay`, che `vãi` và `c***`.
>
> **Muốn cho thấy:** đây là lúc "khoe" char n-gram. `c***` là chiêu né filter kinh điển, nhưng vì
> ta học ở mức *ký tự* chứ không so khớp nguyên từ, dấu sao không cứu được câu độc. Nói thêm: model
> còn hoá giải tách chữ (`Đ m`), kéo dài (`lồnnn`), thay số (`3K`, `2l`) — đều có trong bộ test.

### 4. Trộn sạch + độc — che đúng chỗ

> **Câu:** `cảm ơn thằng ngu`
>
> **Kỳ vọng:** bị chặn; che thành `cảm ơn *** ***` — `cảm ơn` **còn nguyên**, chỉ `thằng ngu` bị che.
>
> **Muốn cho thấy:** che có chọn lọc, không bôi đen cả câu. Cùng một nguồn log-odds vừa quyết định
> chặn vừa quyết định *chữ nào* đáng che, nên phần sạch của câu được giữ lại.

### 5. Câu sạch nhạy cảm ngữ cảnh — KHÔNG chặn oan  ⭐

> **Câu:** `đồ ăn ngon quá`
>
> **Kỳ vọng:** **không** bị chặn, hiện nguyên văn.
>
> **Muốn cho thấy:** đây là câu "gài" và là điểm nhấn của demo. Nó mở đầu bằng `đồ` — đúng chữ có
> trong câu chửi `đồ ngu` ở ca 2 — nhưng model *không* chặn, vì nó cân cả câu chứ không giật mình vì
> một chữ. Một filter theo danh sách từ cấm sẽ chặn oan câu này; của ta thì không.

### 6. Mở "Vì sao" — khoe explainability

> **Thao tác:** quay lại ca 2 (`đồ ngu vãi`) và **mở bảng "Vì sao"**.
>
> **Kỳ vọng:** bảng chỉ ra `ngu` là từ có điểm toxic log-odds **cao nhất**, các từ độc được tô sáng.
>
> **Muốn cho thấy:** quyết định chặn không phải hộp đen. Model chỉ thẳng nó bắt tín hiệu vào chữ
> nào — cùng con số đó vừa dùng để chặn, vừa để che, vừa để giải thích, nên lời giải thích **không
> bao giờ mâu thuẫn** với hành động. Đây là chốt: kiểm duyệt kèm lý do là kiểm duyệt tin được.

---

## Nếu muốn nói thẳng về hạn chế (khi bị hỏi)

Trung thực ăn điểm hơn giấu. Nếu người xem hỏi "có bao giờ sai không?", trả lời gọn: precision
~0.58 nên còn **chặn oan ~40%**; teencode viết tắt cực ngắn như `vl`, `ccc` vẫn lọt. Bộ test ghi
rõ các ca lọt này bằng `xfail` thay vì giả vờ hoàn hảo — chi tiết ở mục "Hạn chế đã biết" trong
`README.md`.

## Phương án dự phòng (chuẩn bị TRƯỚC khi demo)

Mạng và deploy là thứ hay hỏng đúng lúc đông người xem. Chuẩn bị sẵn, theo thứ tự ưu tiên:

- [ ] **Quay sẵn 1 clip/GIF** chạy trọn 6 ca ở trên — nếu link sống chết, chiếu clip là xong.
- [ ] **Chạy được local** phòng khi mất mạng: `streamlit run app/streamlit_app.py` (model đã đóng
      băng trong `models/pipeline.pkl`, inference **không cần mạng**).
- [ ] **Ảnh chụp** trạng thái chặn/che/"Vì sao" của vài ca then chốt (ca 2, 4, 6), để dán nếu cả
      clip lẫn local đều trục trặc.
- [ ] **Mở sẵn tab** demo trước khi lên nói, đã "làm nóng" bằng một câu để pipeline nạp xong —
      lần chạy đầu tốn vài giây load `.pkl`, đừng để khoảng lặng đó rơi vào lúc đang diễn.
- [ ] **Copy sẵn 6 câu mẫu** ra một chỗ dán nhanh, khỏi gõ tay (gõ sai dấu tiếng Việt trước đám
      đông là mất nhịp).
