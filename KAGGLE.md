# Chạy thí nghiệm trên Kaggle

Các notebook được tách theo artifact, nhưng mỗi notebook đều tự tìm/clone repo và
tự dựng lại biến đường dẫn. Vì vậy có thể chạy nối tiếp trong một session hoặc mở
từng notebook ở session riêng.

## 1. Luồng khuyến nghị

Chạy theo chuỗi sau để tránh sinh lại artifact đắt tiền:

1. `00_setup.ipynb`: cài dependency tối thiểu, clone repo, chuẩn bị **BIRD dev đầy đủ**.
2. `01_data_prep.ipynb`: kiểm tra BIRD; nếu có file synthetic training thì lọc và tạo
   `data/sft_subsets/sft_*.jsonl`.
3. `02_baseline.ipynb`: greedy và SC@16 trên model nền.
4. `03_lora_sft.ipynb`: LoRA SFT; mặc định chỉ chạy một cấu hình để phù hợp giới hạn
   GPU, sau đó chọn cấu hình tốt và mới chạy rộng hơn.
5. `04_candidates.ipynb`: sinh ứng viên một lần và lưu `results/candidates/`.
6. `05_adaptive.ipynb`: replay các policy trên cùng ứng viên, không sinh lại model.

Nếu chỉ muốn kiểm tra nhanh, đặt `BIRD_LIMIT=20` và `N_CANDIDATES=4`; khi chạy số liệu
chính thức bỏ hai biến này (`BIRD_LIMIT=0`, `N_CANDIDATES=32`) để dùng toàn bộ dev.

## 2. Dữ liệu

`00_setup.ipynb` tìm dataset BIRD đã attach ở các thư mục phổ biến:

- `/kaggle/input/bird-benchmark`
- `/kaggle/input/bird-dev`
- `/kaggle/input/bird-full-dev`

Dataset phải có `dev.json` và `dev_databases/`. Nếu không có, notebook tải archive
development vào `data/downloads/` bằng URL cấu hình `BIRD_DEV_URL`. Có thể bật lại tải
đầu vào bằng `FORCE_REFRESH_DATA=1`; mặc định chỉ dùng thư mục hiện có sau khi đã kiểm
tra từng database tham chiếu trong `dev.json`.

SynSQL/Syn Think là **nguồn train tùy chọn**, không phải benchmark chính. Notebook không
tự tải toàn bộ SynSQL vì có thể quá nặng và dễ làm đầy đĩa/session. Hãy attach một
JSONL đã chọn hoặc đặt `SYNSQL_RAW=/kaggle/input/.../raw.jsonl`, rồi chạy notebook 01.
Nếu không có file này, 01 sẽ bỏ qua SFT subset và 03 sẽ dừng với thông báo rõ ràng.

## 3. Biến cấu hình hữu ích

| Biến | Mặc định | Dùng cho |
|---|---:|---|
| `SLM_PRJ_REPO_URL` | repo hiện tại | đổi nguồn clone |
| `BIRD_INPUT_DIR` | tự dò `/kaggle/input` | chỉ rõ dataset BIRD |
| `BIRD_DEV_URL` | archive dev | đổi nguồn archive khi cần |
| `BIRD_LIMIT` | `0` | giới hạn câu để pilot; `0` là full dev |
| `INSTALL_VLLM` | `0` | đặt `1` trước 02/04 nếu Kaggle chưa có vLLM |
| `SYNSQL_RAW` | không có | file train JSONL cho SFT |
| `SFT_SIZES` | `10000,30000,50000` | kích thước subset |
| `MAX_EXPERIMENTS` | `1` | số cấu hình LoRA chạy pilot |
| `RUN_FULL_GRID` | `0` | đặt `1` để chạy toàn bộ grid |
| `MODEL_PATH` | adapter merged mẫu | checkpoint dùng ở 04 |
| `N_CANDIDATES` | `32` | số ứng viên/câu |
| `GEN_BATCH_SIZE` | `8` | số câu đưa vào vLLM mỗi batch |
| `N_PERMUTATIONS` | `100` | số hoán vị replay ở 05 |
| `RUN_CSC_TRAINING` | `0` | bật CSC LoRA tốn GPU một cách chủ động |

## 4. Checkpoint lên Hugging Face

Trong Kaggle tạo hai Secrets, không viết token vào notebook:

- `HF_TOKEN`: token có quyền ghi model repo.
- `HF_REPO_ID`: ví dụ `your-account/slm-sql-bird-dev`.

Cell cuối của notebook 03 gọi `HfApi.create_repo(..., private=True)` rồi upload thư mục
`merged` (hoặc đường dẫn trong `CHECKPOINT_DIR`). Không có đủ hai secret thì cell chỉ
in lý do bỏ qua, không upload gì. Giữa các session, checkpoint nằm trong Hugging Face
repo hoặc Kaggle Dataset; `/kaggle/working` không nên được xem là nơi lưu lâu dài.

Để lấy checkpoint trong session 04 mới, thêm cell trước bước sinh ứng viên:

```python
from scripts.kaggle_utils import load_kaggle_secrets, download_checkpoint
load_kaggle_secrets()
MODEL_PATH = download_checkpoint()
import os
os.environ["MODEL_PATH"] = str(MODEL_PATH)
```

Nếu không muốn dùng Hugging Face, attach Kaggle Dataset chứa checkpoint và gán
`MODEL_PATH` sang đường dẫn mount của dataset. Notebook 05 cần file
`results/candidates/exec_dev.jsonl` từ notebook 04; attach artifact tương tự và đặt
`EXEC_FILE` khi chạy khác session.

## 5. Syn Think / Syn Merge

Repo hiện không hardcode một repo ID CycloneBoy cho hai biến thể này. Lý do là nguồn và
quyền truy cập có thể thay đổi, trong khi SFT không cần chúng để kiểm tra pipeline BIRD.
Nếu muốn dùng, attach đúng file JSONL đã chọn rồi dùng `SYNSQL_RAW`; cần xác nhận riêng
chỉ khi bạn muốn mình khóa một URL/repo cụ thể hoặc tự tải dữ liệu bên ngoài.
