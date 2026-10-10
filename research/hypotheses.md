# Câu hỏi nghiên cứu và Giả thuyết — Task 0.2

## Câu hỏi nghiên cứu lõi

### RQ1: Chính sách dừng sớm nào cho đường cong EX-số mẫu tốt nhất với SLM?
- Tín hiệu thực thi (kích thước nhóm dẫn đầu, entropy, tỉ lệ lỗi/rỗng, logprob) có đủ không?
- **Cách kiểm chứng**: Sinh N=32 ứng viên một lần, phát lại offline ≥100 hoán vị; so fixed-k, Beta-stopping, margin, entropy, stopper học.

### RQ2: Có thể ước lượng khả năng 'đáp án đúng nằm trong top-2' để chọn hành động (dừng / thêm mẫu / CSC) không?
- **Cách kiểm chứng**: Đo top-2 coverage@k; huấn luyện bộ định tuyến 3 hành động; so với quy tắc gốc CSC-SQL ở cùng ngân sách.

### RQ3: LoRA với rank/module/lượng dữ liệu nào đạt gần SFT đầy đủ trên T4?
- **Cách kiểm chứng**: Lưới nhỏ r∈{16,64}, attn-only vs all-linear, 10K/30K/50K mẫu; so với số liệu công bố.

## Câu hỏi nghiên cứu phụ

### RQ4: Bỏ hoặc rút gọn CoT làm EX giảm bao nhiêu và tiết kiệm bao nhiêu token/thời gian?
### RQ5: Lọc bằng thực thi, trọng số logprob, đa dạng hóa prompt có nâng EX ở cùng số mẫu không?
### RQ6: Chính sách có giữ được lợi ích trên Spider, câu hỏi tiếng Việt và miền dữ liệu thực tiễn?
### RQ7 (mở rộng): Độ đồng thuận có hiệu chuẩn tốt để 'từ chối khi không chắc' không?

---

## Giả thuyết

### H1 — Adaptive sampling giảm chi phí
Với cùng EX, chính sách adaptive dùng ít mẫu trung bình hơn đáng kể so với fixed-k.
- **Mục tiêu**: giảm ≥40% số mẫu ở EX không thấp hơn fixed-16 quá 0.5 điểm
- **Tiêu chí thành công**: p < 0.05 (McNemar), bootstrap CI không chồng lấp
- **Kịch bản kết quả âm**: pass@k thấp → dừng sớm không giúp; công bố phân tích top-2 coverage

### H2 — Routing 3 hành động tốt hơn quy tắc cố định
Quyết định 'lấy thêm mẫu' hay 'gọi CSC' theo độ đồng thuận tốt hơn quy tắc cố định 'chỉ gọi CSC khi phiếu không nhất quán'.
- **Tiêu chí thành công**: EX cao hơn CSC-SQL gốc ở cùng ngân sách mẫu
- **Kịch bản kết quả âm**: routing phức tạp hơn nhưng không đáng kể → công bố phân tích khi nào CSC giúp

### H3 — LoRA đủ tốt trên phần cứng yếu
LoRA (r 16-64, all-linear) trên tập con 30-50K mẫu đạt ≥90% mức tăng EX của SFT đầy đủ so với mô hình gốc.
- **Tiêu chí thành công**: (LoRA_EX - base_EX) / (published_SFT_EX - base_EX) ≥ 0.90
- **Kịch bản kết quả âm**: LoRA chỉ đạt 70-80% → vẫn có giá trị thực nghiệm, nêu rõ trade-off

---

## Giả định lập kế hoạch
- Thời lượng 16 tuần; 1 người làm
- Mô hình: Qwen2.5-Coder-0.5B và 1.5B
- N tối đa 32 ứng viên/câu
- Đánh giá chính trên BIRD dev, OOD trên Spider
- Kaggle GPU (T4 16GB hoặc P100 16GB)
