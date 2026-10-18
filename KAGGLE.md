# Quy trình Kaggle: cải tiến CSC của SLM-SQL bằng LoRA

Luồng nghiên cứu chính hiện tại: **09 → 10 → 11**.

1. `09_slm_sql_baseline.ipynb`: đo baseline bằng generator **và** CSC của tác giả;
   lưu các ứng viên evaluation, train và validation để tái sử dụng.
2. `10_csc_lora.ipynb`: tạo cặp bất đồng JOIN/tổng hợp từ **train**, huấn luyện LoRA
   trên checkpoint CSC đã SFT + GRPO; chọn checkpoint bằng validation.
3. `11_csc_evaluation.ipynb`: dùng cùng evaluation candidates để so sánh CSC gốc
   và CSC + LoRA. Không gọi generator lại.

Không cần chạy notebook 02 Qwen baseline, 03 Qwen SFT, hoặc 04–08 adaptive/demo cũ.
Không cần SynSQL và không huấn luyện lại RL. Notebook 00 cũ cài stack nghiên cứu
rộng hơn cần thiết; ba notebook mới có bootstrap/profile riêng.

Đọc **[hướng dẫn đầy đủ và CLI](docs/CSC_LORA.md)** trước khi chạy GPU.
Để biết **từng notebook 00–11 lấy input từ đâu, tạo file gì, cách chuyển session
và upload/restore LoRA CSC trên Hugging Face**, xem
[hướng dẫn notebook 00–11](docs/NOTEBOOKS_00_11.md).

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
os.environ['RUN_CSC_GPU'] = '1'
os.environ['INSTALL_CSC_DEPS'] = '1'  # chỉ khi chưa có profile phù hợp
```

Train root chứa `train.json`, `train_databases/`; eval root chứa `dev.json`,
`dev_databases/`. Điều chỉnh theo dataset đã attach. CPU preflight dừng rõ ràng nếu
thiếu DB hoặc gold không compile được trước khi khởi tạo model.

Mặc định pilot 80/30/20 câu, 4 generation samples và 2 merge samples. Pilot kiểm tra
luồng; không dùng điểm đó thay full BIRD dev. Full protocol dùng config/experiment
mới, đặt `CSC_PILOT=0` trước khi tạo config, 64 evaluation samples và 8 merge samples.

## Tách môi trường và lưu artifacts

09/11 dùng `requirements-csc-inference.txt`. 10 dùng `requirements-csc-training.txt`
trong session riêng để tránh conflict vLLM–torch–transformers. Không cài tất cả các
profile cùng lúc chỉ để chạy một stage.

Preserve **toàn bộ experiment directory cùng JSON config**, không chỉ adapter.
Restore ở cùng đường dẫn giữa session. Dùng Kaggle output/dataset hoặc phương thức
lưu artifact của bạn; code không tự upload hay gửi dữ liệu ra ngoài.

Mỗi stage ghi JSONL theo câu và sidecar manifest. Cache hoàn tất, đúng config/hash
thì không load model lại. Ngắt giữa chừng thì resume; config/data không khớp thì
báo lỗi thay vì tự tính lại tốn GPU. CSC được bỏ qua nếu chỉ có một nhóm executable.

Không dùng gold dev để huấn luyện CSC. Legacy `RUN_CSC_TRAINING` trong notebook 05
đã được thay bằng hướng dẫn tới 09–11; API training CSC cũ cũng bị chặn.
Cell cuối 10 chỉ upload adapter lên Hugging Face **private** khi
`PUBLISH_CSC_ADAPTER=1`; cell đầu 11 chỉ restore khi `RESTORE_CSC_ADAPTER=1`.
Hub không thay thế việc lưu toàn bộ cache/config qua các session Kaggle.
