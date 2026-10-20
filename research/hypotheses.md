# Câu hỏi nghiên cứu và giả thuyết

## Câu hỏi nghiên cứu lõi

### RQ1: Chính sách dừng sớm nào cho đường cong EX–chi phí tốt nhất với SLM?

So sánh fixed-k, Beta, margin/entropy và learned stopper trên cùng candidate cache. Đánh giá theo đơn vị câu hỏi và database-disjoint calibration; không dùng gold trong feature lúc inference.

### RQ2: Có thể ước lượng khả năng đáp án đúng nằm trong top-2 để chọn dừng, lấy thêm mẫu hay gọi CSC không?

Tách rõ `top2_correct` khỏi `sc_correct`. Router được fit trên calibration DB, còn EX của `CALL_CSC` phải đến từ merger chạy thật.

### RQ3: LoRA trên CSC merger của tác giả có cải thiện lựa chọn top-2 dưới cùng ngân sách không?

So sánh author merger và LoRA merger với cùng generator cache, top-2, số merge samples, seed và scorer.

## Câu hỏi phụ

- RQ4: Bỏ hoặc rút gọn CoT làm EX, token và thời gian thay đổi thế nào?
- RQ5: Lọc lỗi/rỗng, logprob, prompt diversity hay temperature adaptation có ích ở cùng ngân sách không?
- RQ6: Chính sách có giữ lợi ích trên Spider, biến thể robustness, tiếng Việt và dữ liệu business không?
- RQ7: Độ đồng thuận có thể dùng cho selective prediction/từ chối khi chưa chắc không?

## Giả thuyết và tiêu chí

### H1 — Adaptive sampling giảm chi phí với non-inferiority

Ở cùng khoảng EX đã khai báo, adaptive dùng ít mẫu trung bình hơn fixed-16. Tiêu chí chính: chênh lệch EX `adaptive - fixed` có cận dưới của bootstrap CI không thấp hơn **−0.5 điểm phần trăm**, đồng thời giảm trung bình ít nhất 40% số mẫu; báo cáo thêm McNemar exact và paired CI. McNemar `p > 0.05` chỉ có nghĩa là chưa bác bỏ khác biệt, không phải bằng chứng hai chính sách tương đương.

Kết quả âm vẫn công bố được bằng đường cong EX–sample–time và các nhóm câu mà Beta/learned stopper dừng sai.

### H2 — Router ba hành động có lợi khi tính cả chi phí merge

Router dừng/lấy thêm mẫu/gọi CSC chỉ được coi là tốt hơn nếu EX và tổng chi phí generation + merge được báo trên cùng bộ câu. So sánh với fixed SC và CSC gốc; không dùng `sc_proxy_when_csc_requested` làm EX.

Kết quả âm vẫn có giá trị nếu cho thấy top-2 coverage không đủ để dự báo khi merger sửa đúng hoặc chi phí merge xóa lợi ích.

### H3 — LoRA đạt phần lớn lợi ích của một baseline SFT được ghép cặp

Tỷ lệ `(LoRA - base) / (matched SFT - base)` chỉ được tính khi base, prompt, dữ liệu, seed và scorer phù hợp. Mốc **1.5B SLM-SQL 64.84%** là SFT+RL+CSC ở một cấu hình bài báo, không được dùng làm denominator cho `LoRA SFT` đơn thuần. Nếu không có matched full-SFT reference, chỉ báo cáo delta LoRA-vs-author-merger và không gán nhãn H3 đã kiểm chứng.

Kết quả âm 70–80% vẫn là kết quả trade-off về thời gian, VRAM, adapter size và EX.

## Quy ước báo cáo

- `BIRD dev` của 09–11 là evaluation chính; train/validation BIRD chỉ phục vụ cache/pair/adaptation. Nhánh 00–08 dùng held-out dev theo database khi calibration, nên không gọi đó là full-dev paper score.
- Không dùng output GPU chưa chạy để điền số. Mọi kết quả phải có model revision, prompt version, seed, k, scorer, source hash và runtime.
