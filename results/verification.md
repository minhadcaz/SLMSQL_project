# Kiểm chứng AI local — 12/09/2026

- Windows, Python 3.12.5, NVIDIA GeForce RTX 2050 4 GB, driver 528.20.
- PyTorch 2.6.0+cu118: CUDA khả dụng, phép tính tensor trên GPU thành công.
- Transformers 4.51.3; `pip check` không có dependency lỗi.
- Model mặc định: cycloneboy/SLM-SQL-1.5B, weights local tại `models/SLM-SQL-1.5B`, chế độ greedy, batch 1, device auto, inference offline.
- KV cache bật khi generation. Schema trình bày bằng CREATE TABLE; prompt nhắc chọn đúng entity/attribute và giữ nguyên chữ hoa/thường của giá trị dữ liệu.
- Smoke test model thật: câu “Có bao nhiêu sinh viên?” sinh `SELECT COUNT(*) AS total_students FROM students;`, trả 16; khoảng 17.9 giây, chưa gồm load model.
- Năm câu đầu trong `data/evaluation_questions.json`: đúng kết quả 5/5, SQL hợp lệ 5/5, exact match 4/5; trung bình 18.0 giây/câu, load model 15.9 giây. Báo cáo: `results/real-1.5b-20260912/`.
- Bản 0.5B với prompt trước thay đổi: đúng 3/5, còn nhầm tên bảng và điều kiện. Báo cáo: `results/real-baseline-20260912/`. Hai lượt khác cả model và prompt, không dùng để kết luận riêng hiệu quả của từng thay đổi.
- Unit/UI tests không model: 66 passed. Ruff: passed.
- Streamlit AppTest với model thật và cấu hình mặc định: nhập câu hỏi, bấm Generate SQL, SQL trả 16, không có lỗi UI. Trace: `results/real-ui-20260912.json`.

Năm câu là kiểm tra tích hợp nhỏ, đã được xem trong quá trình cấu hình nên không phải tập đánh giá mù cuối cùng. Không suy ra độ chính xác 100% trên câu mới. Gold SQL chỉ dùng ở evaluator để đối chiếu, không đưa vào prompt model.

## Chạy

Mở `scripts/run_demo.bat`, truy cập http://127.0.0.1:8501, giữ backend huggingface và greedy, nhập câu hỏi rồi bấm Generate SQL. Lần đầu phải chờ nạp model; model được cache trong phiên Streamlit. Có thể chọn adaptive_sc để thử nhiều ứng viên nhưng sẽ chậm hơn. Model có thể sinh SQL chạy được nhưng sai ý nghĩa.
