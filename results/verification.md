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

## Audit phương pháp ngày 19/10/2026

- Python kiểm thử: `/tmp/slm-sql-audit-py312/bin/python` (3.12).
- `139 passed`; Ruff: `All checks passed`. Có 2 cảnh báo deprecation từ multiprocessing fork, không làm fail test.
- Đã compile tất cả code cell trong 12 notebook `00`–`11`; không có SyntaxError.
- Đã kiểm tra parser SynsQL-Think với record JSONL và hỗ trợ JSON-array public export; notebook 08 có guard không gộp hai report cùng variant/seed khác config.
- Chưa chạy: generation/SFT/merge/EX thật trên full BIRD, profiling GPU T4, 3 seed final, Spider OOD, human error review và upload public.
- Kết luận của audit này là cải thiện độ tin cậy phương pháp và khả năng tái lập; chưa phải bằng chứng cải thiện chất lượng SLM-SQL.


## Mở rộng sampling 09–11 ngày 20/10/2026

- Python 3.12.15, môi trường CPU `/tmp/slm-sampling-py312`.
- `python -m pytest -q --tb=short`: **148 passed**, 2 cảnh báo deprecation
  multiprocessing fork từ test legacy; không có test fail. Riêng
  `tests/test_slm_sql_workflow.py`: **31 passed**.
- `ruff check research training evaluation scripts tests`: `All checks passed!`.
- `git diff --check`: đạt.
- nbformat validate và AST parse/compile **12 notebook, 84 code cell**. Các notebook
  legacy 06/07/08 có cảnh báo thiếu cell ID; notebook 09–11 giữ cell ID hợp lệ.
  Đây là kiểm tra source, chưa chạy notebook CUDA end-to-end.
- CPU CLI smoke: `init --pilot`, `plan`, `--help` và help cho `suite`, `merge`,
  `sampling-baseline`, `sampling-final`, `report-suite`; không tải model.
- Integration dùng SQLite thật và backend giả kiểm tra:
  - Greedy là request riêng với temperature=0, khác stochastic candidate đầu;
    seed ghi trong cache khớp request thực tế.
  - Beta và margin đọc đúng nested execution votes: với fixture budget 8,
    unanimous dừng ở 4; tie/invalid dùng 8; tổng sinh 40 thay vì 48 completion.
    Đây là số trong test, không phải benchmark tiết kiệm trên BIRD.
  - Adaptive hoạt động khi chưa có fixed cache, không phát lại candidates;
    resume giữ completed questions và từ chối đổi policy/execution/packages.
  - Author/LoRA fixed cho k=1,2,4 và adaptive dùng candidates chung; k=1 không
    load merger. Tính output tokens gồm generator và merger; prefix k<n để
    generation timing null. Missing artifact hoặc sai package provenance bị chặn.
  - CLI `sampling-baseline` và `sampling-final` chạy lại trên cache hoàn tất mà
    không có vLLM. Freeze khóa adaptive settings/seeds/k; báo cáo hai seed vẫn
    bootstrap trên 6 câu, không gộp thành 12 quan sát độc lập.
- Đã cập nhật config, CLI, notebook 09–11 và tài liệu. Thiết kế/checklist:
  [SLM_SQL_SAMPLING_PLAN.md](../docs/SLM_SQL_SAMPLING_PLAN.md).
- Chưa chạy: model inference/LoRA CUDA, full BIRD matrix qua 42/43/44, đo latency
  cùng workload hoặc xác nhận EX tăng. Không dùng điểm từ backend giả làm kết quả
  chất lượng SLM-SQL.


## Kiểm tra tiếp tục ngày 20/10/2026

- `.venv-csc/bin/pytest -q --tb=short`: **157 passed** trong 22.93 giây; không có test fail. Ba regression test mới khóa semantics max-budget/CSC, offline consensus stop và shape/digest của stage records.
- `ruff check .`: `All checks passed!`; `pyproject.toml` loại riêng notebook legacy 00–08 khỏi lint vì chúng dùng state xuyên cell và không thuộc pipeline hiện hành. Notebook 09–12, source và test vẫn được kiểm tra.
- `git diff --check`: đạt. `python -m py_compile research/slm_sql/__main__.py` và help cho `adaptive-sweep`, `adaptive-csc-eval`, `adaptive-merge`: đạt.
- AST parse/JSON validation: **13 notebook, 90 code cell**, gồm notebook 12; không có `SyntaxError`.
- Tài liệu đã nối pipeline thành `09 → 10 → 11 → 12`: policy được chọn chỉ trên validation; `adaptive-merge --variant author|lora` xử lý online adaptive cache; `adaptive-csc-eval --split evaluation` ghi replay `offline_prefix_exact`; bảng online đầy đủ nằm trong `sampling.comparison`.
- Chưa chạy: model inference/LoRA CUDA, full BIRD matrix qua 42/43/44, latency thực tế hoặc xác nhận EX tăng. Không dùng test backend giả làm kết quả chất lượng SLM-SQL.
