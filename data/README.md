# Bộ 200 câu hỏi Text-to-SQL tiếng Việt

Database: `databases/student_management.db`. Dữ liệu sinh viên giả lập, 5 bảng; không thay đổi số bản ghi khi mở rộng câu hỏi.

| Tập | Số câu | Dễ | Trung bình | Khó | Mục đích |
| --- | ---: | ---: | ---: | ---: | --- |
| `demo_questions.json` | 100 | 33 | 41 | 26 | Dropdown giao diện; fixture của mock |
| `evaluation_questions.json` | 100 | 26 | 39 | 35 | Đối chiếu SQL do model thật sinh |
| Tổng | 200 | 59 | 80 | 61 | |

Giữ nguyên nội dung và ID của 30 câu demo và 20 câu đánh giá ban đầu. Bổ sung `demo_031`–`demo_100` và `eval_021`–`eval_100`. Mỗi bản ghi có `id`, `question`, `gold_sql`, `difficulty`, `tags`, `language`, `source`, `db_id`.

Nội dung gồm SELECT, lọc, sắp xếp, phân trang, JOIN/LEFT JOIN, GROUP BY/HAVING, NULL, tổng hợp có điều kiện, truy vấn con tương quan, EXISTS/NOT EXISTS, CTE, phép tập hợp và hàm cửa sổ. Các câu lấy top có quy tắc xử lý hòa; câu giữ mọi đồng hạng nói rõ điều đó.

## Quy ước đáp án

- Một lượt đăng ký là một dòng `enrollments`; một môn khác nhau chỉ tính một lần khi câu hỏi yêu cầu rõ.
- Điểm NULL là chưa có điểm, không phải 0. Điểm đạt là từ 5 trở lên khi được nêu trong câu hỏi.
- Trung bình các lượt và trung bình các trung bình sinh viên là hai phép tính khác nhau, được mô tả trong từng câu.
- Trung bình có trọng số tín chỉ chỉ dùng các lượt có điểm ở cả tử và mẫu.
- So sánh tên dùng giá trị lưu trong SQLite, giữ dấu và chữ hoa/thường.
- Không yêu cầu thứ tự thì so sánh multiset; không tự loại dòng trùng nếu câu hỏi không yêu cầu.
- Trên database hiện tại, `demo_082`, `eval_074`, `eval_079`, `eval_098` trả rỗng hợp lệ. Các câu này lần lượt tìm người đăng ký nhưng chưa có điểm nào, môn có đăng ký và mọi lượt đều có điểm, người chỉ đăng ký kỳ 2024-1, người có nhiều lượt hơn sinh viên mã 1.

## Kiểm chứng và giới hạn

Đã kiểm tra 200 ID/câu hỏi duy nhất, không trùng SQL chuẩn hóa và toàn bộ SQL qua validator/read-only executor. Tests bổ sung kiểm tra đăng ký lại khác học kỳ, điểm NULL và LEFT JOIN môn chưa đăng ký. Chạy từ thư mục project:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe databases/create_student_db.py
```

Đánh giá AI thật trên 100 câu (có thể mất nhiều thời gian hơn các câu dễ đã chạy):

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --backend huggingface --mode greedy --dataset data/evaluation_questions.json
```

Đây là dữ liệu tự soạn trong một miền và một schema, chưa được người gán nhãn độc lập rà soát. Hai tập không trùng câu chữ hoặc SQL chuẩn hóa nhưng có cấu trúc/ngữ nghĩa gần nhau; không phải phép chia chống rò rỉ theo họ truy vấn. Năm câu đánh giá đầu đã được dùng khi cấu hình model. Mở rộng câu hỏi không huấn luyện lại hay tự tăng năng lực AI. Chưa chạy model thật trên toàn bộ 100 câu đánh giá mới; kết quả 5/5 trước đó chỉ áp dụng lượt kiểm tra cũ.
