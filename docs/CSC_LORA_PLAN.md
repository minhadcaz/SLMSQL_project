# Kế hoạch: cải tiến CSC trên SLM-SQL của tác giả

Mục tiêu: giữ generator SLM-SQL đã SFT + GRPO, thích nghi checkpoint CSC đã SFT +
GRPO bằng LoRA cho bất đồng JOIN/tổng hợp. So sánh với CSC gốc trên cùng ứng viên.
Không tuyên bố tái lập 67,08% khi prompt/schema/hạ tầng chưa trùng thiết lập tác giả.

Các mục dưới đây đánh dấu phần **đã triển khai trong source**, không phải kết quả
huấn luyện/EX trên model thật.

1. [x] Baseline dùng hai checkpoint tác giả, lấy mẫu generation và merge tách biệt.
2. [x] Chia train/validation theo database, BIRD dev chỉ dành cho đánh giá cuối.
3. [x] Tạo cặp top-2 từ train; chỉ chọn cặp có đáp án thực thi đúng và bất đồng cấu trúc.
4. [x] Huấn luyện LoRA trên merger, completion-only loss, chọn checkpoint bằng validation.
5. [x] So sánh CSC gốc/LoRA cùng cache, prompt, seed và ngân sách; EX theo BIRD set semantics.
6. [x] Lưu artifact theo câu, resume; kiểm tra config/data/model/protocol trước khi tái sử dụng.
7. [x] Notebook/CLI mới chạy từng stage, có preflight và dự toán số mẫu trước khi dùng GPU.
8. [x] Kiểm thử CPU bằng SQLite và backend giả, chặn leakage và cache không khớp.

Không huấn luyện lại Qwen, không cần SynSQL, không chạy lại RL trong hướng này.
Không chạy GPU khi artifact phù hợp đã hoàn tất. Dữ liệu và weights không có sẵn
trong source; các kết quả EX chỉ được công bố khi có artifact đánh giá thật.

## Kết quả kiểm tra triển khai ban đầu (lịch sử)

- `python -m pytest -q --tb=short`: **118 passed**, gồm 19 trường hợp trong
  `tests/test_slm_sql_workflow.py`; SQLite thật và backend GPU giả.
- Ruff lint và format check: đạt. `git diff --check`: đạt.
- JSON của toàn bộ notebooks hợp lệ; Python cells của 09–12 parse thành công.
- CLI `--help`, `init --pilot`, `plan`: chạy được trên CPU, không download model.
  CLI `baseline` đã kiểm thử chạy lại trên cache hoàn tất mà không cần vLLM/GPU.
- Đã kiểm thử dev-only baseline rồi thêm train/validation mà không sinh lại dev;
  từ chối predictions từ adapter cũ và runtime inference không khớp; từ chối
  concurrent writers trước khi train/prepare/build-pairs.

Môi trường kiểm thử local dùng Python 3.14.8 và sqlglot 26.16.4. Đây là kiểm thử
logic/CLI và regression, chưa phải xác nhận stack CUDA của hai dependency profile.

## Mở rộng sampling ngày 20/10/2026

Đã mở rộng 09–12 để đo greedy riêng, fixed-k/pass@k/SC/author CSC/LoRA CSC và
adaptive online với author/LoRA CSC trên cùng stopped candidates. Thiết kế,
checklist triển khai và phần GPU còn lại trong
[SLM_SQL_SAMPLING_PLAN.md](SLM_SQL_SAMPLING_PLAN.md). Kết quả kiểm thử mới nhất
ở [verification.md](../results/verification.md); các con số ở mục trước là lịch sử.

## Thí nghiệm cần chạy trên dữ liệu/model thật

- [ ] Pilot baseline với generator và merger của tác giả trên BIRD dev.
- [ ] Tạo pairs đủ dùng từ BIRD train và huấn luyện adapter trên CUDA.
- [ ] Khóa adapter rồi so sánh EX author/LoRA trên full dev cùng ngân sách.
- [ ] Lặp inference theo seed đã định; báo cáo cả cải thiện và regression.

Phiên local không có BIRD/weights và `nvidia-smi` không kết nối được driver CUDA.
Không khởi chạy download/train thay thế hoặc điền kết quả EX từ backend giả.
