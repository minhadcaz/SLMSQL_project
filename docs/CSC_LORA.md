# CSC LoRA trên SLM-SQL đã SFT + GRPO

Luồng nghiên cứu chính là **09 → 10 → 11**. Các notebook 02–08 là thí nghiệm
Qwen/LoRA/adaptive cũ hoặc demo tùy chọn; không cần chạy chúng cho hướng này.
Không cần SynSQL, không SFT Qwen lại từ đầu, không chạy GRPO lại.

## Thành phần và phép so sánh

- Generator giữ nguyên: `cycloneboy/SLM-SQL-1.5B`.
- Merger baseline: `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct`.
- Cải tiến: LoRA/QLoRA trên chính merger đã SFT + GRPO của tác giả.
- Task: lựa chọn SQL đúng giữa hai nhóm kết quả bất đồng về JOIN/tổng hợp.
- Baseline và adapter dùng **cùng generator cache, top-2, prompt, seed, k và số mẫu merge**.
- CSC chỉ chạy khi có ít nhất hai nhóm kết quả SQL thực thi được. Merge outputs
  cũng vote theo kết quả thực thi. Tất cả merge outputs lỗi thì tính thất bại;
  không âm thầm dùng lại đáp án SC.

Tác giả liệt kê cả generator và merger là SFT + GRPO trong repository. Chọn
revision/commit cụ thể trong config nếu đã biết. Với `main`, commit thực tế được
resolve ở lần đầu tải model và ghi vào manifest; các lần resume dùng đúng commit
đó. Resume và so sánh author/LoRA yêu cầu cùng phiên bản runtime inference;
manifest ghi phiên bản vLLM/transformers/torch để khôi phục môi trường. LoRA luôn
dùng đúng base merger commit của baseline. Local checkpoint được
hash weights/config để phát hiện thay đổi.

Đây là **baseline dùng model và cơ chế SLM-SQL**, không tuyên bố đã tái lập EX
67,08%. Prompt là bản dựng lại theo ý nghĩa Phụ lục E, dùng full DDL, ba mẫu giá trị
schema, result preview có giới hạn và tie-break theo mẫu xuất hiện trước. Cần giữ
các khác biệt này trong báo cáo. Không chép số công bố vào kết quả chạy thật.

## Dữ liệu và chống leakage

Để huấn luyện adapter cần hai nguồn độc lập. Nếu chỉ đo baseline của tác giả,
chỉ cần nguồn BIRD dev và `prepare --evaluation-only`:

```text
/path/to/bird-train/
  train.json
  train_databases/<db_id>/<db_id>.sqlite
/path/to/bird-dev/
  dev.json
  dev_databases/<db_id>/<db_id>.sqlite
```

`prepare` chia các database của **BIRD train** thành train và validation trước
khi áp dụng giới hạn số câu. BIRD dev là evaluation cuối; kiểm tra ID database,
trùng nội dung SQLite, source split và hash dữ liệu. Gold SQL được compile trước
khi dùng GPU. Hash SQLite chỉ đọc một lần ở prepare; các stage tiếp theo kiểm tra
size/mtime cùng hash của prepared records/schema. Không sửa database sau prepare.

Validation chỉ được giữ riêng đối với **adapter của dự án**. Model của tác giả
đã được học từ BIRD train; không gọi validation này là dữ liệu chưa từng xuất hiện
trong quá trình huấn luyện upstream.

Pair builder chỉ đọc train/validation artifacts. Cặp được giữ khi:

1. Hai nhóm executable khác kết quả, nằm ở top-2 theo số phiếu.
2. Một nhóm khớp gold execution của tập tương ứng.
3. SQL của hai nhóm khác cấu trúc JOIN hoặc aggregate/GROUP BY/HAVING/DISTINCT.

Đây là bộ lọc cấu trúc, không phải nhãn nguyên nhân lỗi ngữ nghĩa đã được con người
xác nhận. Giữ cả trường hợp leader đúng và leader sai; train có hoán đổi A/B để
tránh học vị trí. Target là SQL draft đã được kiểm chứng trong `<answer>`;
không tự bịa reasoning labels. Loss chỉ tính trên assistant completion.
Không có cặp phù hợp thì dừng rõ ràng; tăng train limit/số mẫu trong experiment mới,
không lấy gold evaluation để bù dữ liệu.

Checkpoint chọn bằng validation loss. Freeze hyperparameters và adapter trước
khi xem so sánh cuối trên dev; không chọn adapter theo EX của tập này.

## Môi trường

Khuyến nghị Python 3.11/3.12, CUDA. Tách inference và training thành hai Kaggle
session để tránh xung đột torch/transformers do vLLM lựa chọn dependency riêng.
Không cần cài `requirements-research.txt` hoặc Unsloth cho quy trình mới.

Inference:

```bash
python -m pip install -r requirements-csc-inference.txt
```

Training:

```bash
python -m pip install -r requirements-csc-training.txt
```

4-bit NF4 là tùy chọn training mặc định; rank 16, LR 5e-5, batch 1, gradient
accumulation 16. Đây là điểm khởi đầu, chưa phải siêu tham số đã chứng minh tối ưu.
Kiểm tra context và VRAM trên GPU thực tế; không có benchmark GPU trong source.

## CLI: chạy pilot trước

Chạy từ thư mục repository. Các lệnh init/plan/prepare/pairs/execute/score không
khởi tạo model GPU. Không có tự động download BIRD hoặc train khi thiếu artifact.

```bash
python -m research.slm_sql init --pilot --output configs/csc_pilot.json \
  --train-dir /path/to/bird-train --evaluation-dir /path/to/bird-dev
python -m research.slm_sql --config configs/csc_pilot.json plan
python -m research.slm_sql --config configs/csc_pilot.json prepare
python -m research.slm_sql --config configs/csc_pilot.json plan
```

Pilot: 80/30/20 câu train/validation/evaluation, 4 generation samples, 2 merger
samples. Điểm pilot không đại diện full BIRD dev.

Nếu hiện chỉ có BIRD dev, thay lệnh `prepare` ở trên bằng:

```bash
python -m research.slm_sql --config configs/csc_pilot.json prepare --evaluation-only
```

Sau đó vẫn chạy được `baseline`. Khi có train, chạy `prepare` bình thường với cùng
config/experiment để bổ sung train và validation; evaluation cache được giữ nguyên.
Không đổi evaluation settings, dev JSON hay database giữa hai lần chuẩn bị.

Đo baseline trước khi huấn luyện:

```bash
python -m research.slm_sql --config configs/csc_pilot.json baseline --seed 42
```

Lệnh này chạy generation → execution → author merger → baseline score trong các
process riêng để nhả VRAM giữa hai model. Nếu chỉ cần đánh giá model gốc, dừng ở đây.

Chuẩn bị train/validation cache trong inference session, chỉ một lần:

```bash
python -m research.slm_sql --config configs/csc_pilot.json generate --split train
python -m research.slm_sql --config configs/csc_pilot.json execute --split train
python -m research.slm_sql --config configs/csc_pilot.json generate --split validation
python -m research.slm_sql --config configs/csc_pilot.json execute --split validation
python -m research.slm_sql --config configs/csc_pilot.json pairs
```

Trong training session, dùng cùng source/config/experiment folder:

```bash
python -m research.slm_sql --config configs/csc_pilot.json train
```

Adapter được lưu tại `<experiment_dir>/adapter/final`. Huấn luyện ngắt sẽ resume
checkpoint Trainer cuối; job đã xong và khớp hash không train lại. Ví dụ quá context
bị bỏ kèm danh sách ID trong training manifest, không cắt lén SQL target. Nếu toàn
bộ train hoặc validation quá dài thì dừng trước khi tải model weights.

Trong inference session cuối, chỉ gọi merger + LoRA; generator không chạy lại:

```bash
python -m research.slm_sql --config configs/csc_pilot.json merge --variant lora
python -m research.slm_sql --config configs/csc_pilot.json score
```

Muốn dùng full protocol: tạo config mới **không có `--pilot`**, output config mới,
và experiment directory mới. Default chính: train limit 1200, validation 200,
**full dev**, 16 train/validation samples, 64 evaluation samples, 8 merger samples.
Không thay config của pilot rồi trộn caches cũ.

Có thể dùng prefix k=16 từ cache 64 mà không sinh lại generator:

```bash
python -m research.slm_sql --config configs/csc_full.json merge --variant author --k 16
python -m research.slm_sql --config configs/csc_full.json merge --variant lora --k 16
python -m research.slm_sql --config configs/csc_full.json score --k 16
```

Sau khi khóa adapter, đánh giá thêm seed 43/44 bằng `baseline --seed 43`,
`merge --variant lora --seed 43`, `score --seed 43` và tương tự seed 44. Mỗi seed
cần generator draws riêng; **không train lại adapter** và không dựng pairs lại.
Không gộp những lần lặp cùng câu thành các quan sát độc lập để thu hẹp CI.

## Notebook Kaggle

Xem thêm [hướng dẫn chi tiết từng notebook 00–11](NOTEBOOKS_00_11.md): nguồn
input/output, thứ tự các nhánh, cách lưu/khôi phục cache và upload adapter lên
repo Hugging Face private theo cờ opt-in.

Các notebook tự tìm repo, không cần chạy 00–08 hoặc cài toàn bộ dependencies cũ.
Đồng bộ source mới này lên Kaggle trước; một bản clone remote chưa có thay đổi mới
sẽ không có module `research.slm_sql`.

- `09_slm_sql_baseline.ipynb`: prepare, baseline và cache train/validation cần cho LoRA.
- `10_csc_lora.ipynb`: pair builder CPU rồi training/resume merger adapter.
- `11_csc_evaluation.ipynb`: author/LoRA merge trên cùng cache, score và xem lỗi.

Notebook 10 chỉ upload **adapter, không phải base weights hay BIRD**, khi đã
train xong và đặt `PUBLISH_CSC_ADAPTER=1`, secrets `HF_TOKEN` (write) và
`HF_REPO_ID` (`username/repo`). Từ chối repo public. Notebook 11 khôi phục
adapter từ repo private khi `RESTORE_CSC_ADAPTER=1`, sau khi restore toàn bộ
experiment/cache/config qua Kaggle output; model repo riêng không chứa cache.

Biến cấu hình đặt **trước cell bootstrap**:

| Biến | Ý nghĩa |
|---|---|
| `SLM_PRJ_PATH` | thư mục source đã cập nhật |
| `CSC_CONFIG` | JSON config, mặc định `results/csc_run_config.json` |
| `CSC_TRAIN_DIR` / `CSC_EVAL_DIR` | hai nguồn dữ liệu như cấu trúc trên |
| `CSC_PILOT` | `1` mặc định khi tạo config lần đầu; `0` cho full |
| `CSC_EXPERIMENT_DIR` | nơi lưu toàn bộ artifacts, nên cố định giữa session |
| `RUN_CSC_GPU` | `1` để chạy các stage GPU; mặc định `0` chỉ kiểm tra CPU |
| `INSTALL_CSC_DEPS` | `1` cài profile tương ứng; chỉ bật khi cần |
| `BUILD_CSC_TRAIN_CACHE` | `0` nếu notebook 09 chỉ đo baseline |
| `CSC_SEED` | seed inference, mặc định 42 |
| `CSC_K` | prefix k trong notebook 11, mặc định toàn bộ cache |

Giữ toàn bộ experiment folder **và config** giữa session: prepared records,
schemas/manifest, generation/execution/merge JSONL và sidecars, SQL execution cache,
pairs, adapter weights/training manifest, `models/` chứa commit đã khóa, reports. Notebook 11 không cần mount lại
raw train databases, nhưng vẫn cần prepared metadata, pair files và raw evaluation DB.
Đường dẫn trong config phải khớp vị trí restore; không trộn artifact từ run khác.

## Đọc kết quả và giới hạn

Bộ chấm mới mô phỏng BIRD EX: so **set** của complete result rows, bỏ qua thứ tự và
row duplicates, giữ đúng so sánh numeric của Python/SQLite. Không dùng scorer strict
của demo app. SQL timeout, format lỗi, result-limit tính sai; gold lỗi dừng hẳn.
Resource cap là 1 triệu dòng mặc định; tăng trong experiment mới nếu cần.

`reports/*.baseline.json` có SC và author CSC. `*.comparison.json` có thêm LoRA:

- EX toàn bộ và theo difficulty/JOIN/aggregate; SQL-feature slices có thể trùng nhau.
- EX trong tập được gọi CSC; wrong→correct và correct→wrong so với SC.
- Paired author-vs-LoRA delta EX, bootstrap 95% CI theo câu và McNemar exact test.
- Số merge calls, output tokens, output bị cắt và format errors.
- Thời gian generation/merge là amortized batch time; không phải interactive latency.
  Không suy diễn latency k=16 từ generation k=64.
- Có cả SQL/flags từng câu, commit generator/merger và hash adapter để kiểm tra cải thiện và regression.
- Chấm điểm từ chối predictions của adapter cũ nếu weights/manifest đã thay đổi;
  khóa ghi ngăn hai tiến trình cùng chuẩn bị dữ liệu, dựng pairs hoặc huấn luyện trùng.

Không tự điền điểm khi thiếu model/artifact. Kiểm thử backend giả chỉ xác nhận code
và protocol; không phải số đo chất lượng SLM-SQL. Chạy GPU trên BIRD thật mới xác định
được hướng LoRA này có cải thiện EX hay không.

## Nguồn phương pháp

- Paper: https://aclanthology.org/2025.findings-ijcnlp.92.pdf
- Author checkpoints: https://github.com/CycloneBoy/slm_sql
- CSC reference inference: https://github.com/CycloneBoy/csc_sql
- BIRD scorer: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/bird/llm/src/evaluation.py
- PEFT LoRA: https://huggingface.co/docs/peft/developer_guides/lora

## Kiểm thử local không GPU

```bash
python -m pip install pytest sqlglot==26.16.4 PyYAML
python -m pytest tests/test_slm_sql_workflow.py -q
```

Dùng SQLite thật với backend giả: kiểm tra split isolation, numerical/set semantics,
resume, cache rejection, completion masking, invalid merge handling, paired metrics
và chặn đường CSC cũ dùng dev gold. App/demo tests dùng requirements.txt riêng. Kết quả kiểm tra triển khai và các
thí nghiệm GPU chưa chạy được ghi trong [kế hoạch](CSC_LORA_PLAN.md).
