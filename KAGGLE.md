# Quy trình Kaggle: cải tiến CSC của SLM-SQL bằng LoRA

Luồng nghiên cứu chính hiện tại: **09 → 10 → 11 → 12**.

1. `09_slm_sql_baseline.ipynb`: đo greedy riêng, fixed-k (pass@k, SC@k, author
   CSC@k) và adaptive online với generator **SLM-SQL của tác giả**; lưu fixed và
   adaptive candidates, cùng train/validation cache để tái sử dụng.
2. `10_csc_lora.ipynb`: tạo cặp bất đồng JOIN/tổng hợp từ **train**, huấn luyện LoRA
   trên checkpoint CSC đã SFT + GRPO; chọn checkpoint bằng validation.
3. `11_csc_evaluation.ipynb`: khóa adapter/config/seeds/k/adaptive settings rồi
   so sánh CSC gốc và CSC + LoRA trên cả fixed và adaptive candidates; báo cáo
   toàn sampling matrix qua các seed. Không gọi generator lại.
4. `12_adaptive_sampling.ipynb`: chọn policy chỉ trên validation, chạy adaptive
   author/LoRA merge và in báo cáo online/offline adaptive cùng bảng sampling.

Không cần chạy notebook 02 Qwen baseline, 03 Qwen SFT, hoặc 04–08 adaptive/demo cũ.
Không cần SynSQL và không huấn luyện lại RL. Notebook 00 cũ cài stack nghiên cứu
rộng hơn cần thiết; bốn notebook 09–12 có bootstrap/profile riêng.

Đọc **[hướng dẫn đầy đủ và CLI](docs/CSC_LORA.md)** trước khi chạy GPU.
Để biết **từng notebook 00–12 lấy input từ đâu, tạo file gì, cách chuyển session
và upload/restore LoRA CSC trên Hugging Face**, xem
[hướng dẫn notebook 00–12](docs/NOTEBOOKS_00_11.md).

## Chuẩn bị

Đưa **source mới** lên Kaggle; clone remote chỉ dùng được sau khi remote đã có
module `research/slm_sql`. Bật GPU và Internet để tải checkpoint khi cần.
Attach BIRD **train** và **dev** để làm toàn bộ thí nghiệm. Nếu chỉ có dev, đặt
`BUILD_CSC_TRAIN_CACHE=0` để notebook 09 dùng `prepare --evaluation-only` và đo
baseline model gốc. Khi có train, đặt lại `BUILD_CSC_TRAIN_CACHE=1` rồi chạy lại
09: code thêm train/validation và tái sử dụng evaluation cache, không sinh lại.

Đặt trong notebook trước bootstrap:

```python
import os
os.environ['CSC_TRAIN_DIR'] = '/kaggle/input/your-bird-train/train'
os.environ['CSC_EVAL_DIR'] = '/kaggle/input/your-bird-dev/dev'
os.environ['CSC_CONFIG'] = '/kaggle/working/SLM-PRJ/results/csc_run_config.json'
os.environ['CSC_EXPERIMENT_DIR'] = '/kaggle/working/SLM-PRJ/results/csc_slm_sql_pilot'
os.environ['CSC_PILOT'] = '1'
os.environ['CSC_EVAL_SEEDS'] = '42'  # full run: '42,43,44'
os.environ['RUN_CSC_GPU'] = '1'
os.environ['RUN_ADAPTIVE_POLICY_SWEEP'] = '0'  # notebook 12: chọn policy trên validation
os.environ['RUN_ADAPTIVE_MERGE'] = '0'  # notebook 12: chạy adaptive author/LoRA merge
os.environ['INSTALL_CSC_DEPS'] = '1'  # chỉ khi chưa có profile phù hợp
```

Train root chứa `train.json`, `train_databases/`; eval root chứa `dev.json`,
`dev_databases/`. Điều chỉnh theo dataset đã attach. CPU preflight dừng rõ ràng nếu
thiếu DB hoặc gold không compile được trước khi khởi tạo model.

Mặc định pilot 80/30/20 câu, 4 generation samples và 2 merge samples. Pilot kiểm tra
luồng; không dùng điểm đó thay full BIRD dev. Full protocol dùng config/experiment
mới, đặt `CSC_PILOT=0` trước khi tạo config, 64 evaluation samples và 8 merge samples.
Fixed k: `[1,2,4]` pilot / `[1,2,4,8,16,32,64]` full. Để `CSC_K` trống ở notebook 11
để chạy toàn sweep; chỉ đặt khi chủ đích chạy một k. Đặt cùng `CSC_EVAL_SEEDS` ở
09 và 11/12; 10 dùng `CSC_SEED` để huấn luyện một adapter chung.

Adaptive mặc định Beta threshold=0.95, batch=4, min=4, max=64 (pilot: batch=2,
min=2,max=4). Có thêm margin/entropy qua config. Cấu hình trước thí nghiệm cuối;
không tune threshold bằng final dev EX. Adaptive tạo draws riêng theo vòng, không
phát lại full cache. Vì vậy 09 có thêm chi phí greedy/adaptive và author CSC cho
mỗi k; xem `plan` trước khi chạy. Pilot có thể không tiết kiệm mẫu do cap nhỏ.

Trong §4.3.2, `SLM-SQL@k` là kết quả **có CSC**, còn SC vote là
`self_consistency@k`. `pass@k` là oracle coverage; không đưa vào xếp hạng bộ chọn
SQL triển khai. Các curve dùng checkpoint SFT + GRPO đã công bố, không huấn luyện
một SFT-only baseline mới. [Plan sampling](docs/SLM_SQL_SAMPLING_PLAN.md) ghi rõ
thiết kế và những phép đo GPU còn cần chạy.

## Tách môi trường và lưu artifacts

09/11/12 dùng `requirements-csc-inference.txt`. 10 dùng `requirements-csc-training.txt`
trong session riêng để tránh conflict vLLM–torch–transformers. Không cài tất cả các
profile cùng lúc chỉ để chạy một stage.

Preserve **toàn bộ experiment directory cùng JSON config**, không chỉ adapter.
Restore ở cùng đường dẫn giữa session. Dùng Kaggle output/dataset hoặc phương thức
lưu artifact của bạn; code không tự upload hay gửi dữ liệu ra ngoài.

Mỗi stage ghi JSONL theo câu và sidecar manifest. Cache hoàn tất, đúng config/hash
thì không load model lại. Ngắt giữa chừng thì resume; config/data không khớp thì
báo lỗi thay vì tự tính lại tốn GPU. CSC được bỏ qua nếu chỉ có một nhóm executable.

Không dùng gold dev để huấn luyện CSC. Legacy `RUN_CSC_TRAINING` trong notebook 05
đã được thay bằng hướng dẫn tới 09–12; API training CSC cũ cũng bị chặn.
Cell cuối 10 chỉ upload adapter lên Hugging Face **private** khi
`PUBLISH_CSC_ADAPTER=1`; cell đầu 11 chỉ restore khi `RESTORE_CSC_ADAPTER=1`.
Hub không thay thế việc lưu toàn bộ cache/config qua các session Kaggle.

## Báo cáo sau notebook 11–12

Trong `<experiment_dir>/reports/`:

- `evaluation-seed42.sampling.baseline.json` và `.csv`: ma trận author baseline từ 09.
- `evaluation-seed42.sampling.comparison.json` và `.csv`: greedy, fixed SC/CSC,
  adaptive SC/CSC author và LoRA cùng paired changes.
- `evaluation-seed42.adaptive-csc-eval.json`: replay policy đã chọn trên validation
  theo `offline_prefix_exact`; report này giữ riêng với adaptive online trong suite.
- `validation-seed42.adaptive-sweep.json` và `adaptive/selected_policy.json`:
  sweep/provenance của policy. Chọn policy chỉ trên validation, tuyệt đối không tune
  threshold bằng EX của evaluation.
- `evaluation.sampling.comparison.multi-seed.json` và `.csv`: EX mean/std, chi phí,
  CI theo câu đã lấy trung bình seed. Một `--k`/`CSC_K` thêm `-k<K>` vào tên report.

Số samples và output tokens gồm generator lẫn merger. Latency prefix k<n được
để null; thời gian adaptive tuần tự và fixed batch không dùng để suy ra speedup
trực tiếp. Thiếu artifacts hoặc sai hash/runtime phải hoàn tất stage tương ứng,
không tự điền điểm. Chưa có benchmark GPU/BIRD thật cho phần mở rộng này.
