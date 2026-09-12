# Hướng dẫn setup và chạy Adaptive SLM SQL sau khi clone

Ứng dụng chuyển câu hỏi tiếng Việt/tiếng Anh thành SQL và chạy trên SQLite. Backend AI mặc định là **SLM-SQL-1.5B**, chạy trên máy local qua Transformers; giao diện dùng Streamlit.

Hướng dẫn chính dành cho **Windows PowerShell**. Linux/macOS xem mục 9.

## 1. Chuẩn bị

- Cài Git và Python **3.11 hoặc 3.12**; môi trường đã kiểm chứng dùng Python 3.12.5. Không dùng Python 3.10 vì executor cần API SQLite có từ Python 3.11.
- Có Internet để cài thư viện và tải model lần đầu. Sau khi tải đủ, cấu hình mặc định chạy model offline.
- Có dung lượng đĩa trống cho môi trường Python và weights, nên dành ít nhất khoảng 15 GB cho setup GPU. Đây là mức dự phòng, không phải kích thước tải chính xác.
- Có thể chạy CPU. GPU NVIDIA giúp giảm thời gian sinh SQL. Máy đã kiểm chứng: RTX 2050 4 GB, driver 528.20, PyTorch 2.6.0+cu118, greedy và batch 1. Database/prompt lớn hơn có thể cần nhiều bộ nhớ hơn.

Các thư mục `.venv/`, `models/`, `.cache/` bị Git bỏ qua, **không đi kèm khi clone**. Người dùng mới phải tạo môi trường và tải model theo các bước dưới đây. Không sao chép `.venv` từ máy khác.

## 2. Clone và vào thư mục ứng dụng

Thay `URL_REPO_CUA_BAN` bằng URL Git thật của repo:

```powershell
git clone URL_REPO_CUA_BAN SLM-PRJ
cd .\SLM-PRJ\adaptive-slm-sql
```

Nếu repo của bạn đặt `app.py` ngay ở thư mục gốc, dùng `cd .\SLM-PRJ` thay cho dòng `cd` trên. Từ đây, mọi lệnh phải chạy trong thư mục có **`app.py`, `config.yaml`, `requirements.txt`**.

Kiểm tra Python và tạo môi trường:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu dùng Python 3.11, thay `py -3.12` bằng `py -3.11`. Các lệnh bên dưới gọi Python trực tiếp trong `.venv`, không cần activate hay thay execution policy.

## 3. Cài thư viện AI — chọn một nhánh

### A. Có GPU NVIDIA

Kiểm tra driver:

```powershell
nvidia-smi
```

Cài PyTorch CUDA 11.8 và các thư viện model:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU chua kha dung')"
```

Kỳ vọng có `CUDA: True` và tên GPU. Đây là bộ phiên bản đã chạy trên máy kiểm chứng; GPU/driver khác cần wheel tương thích. Các wheel CPU/CUDA cho 2.6.0 được liệt kê tại [tài liệu PyTorch chính thức](https://pytorch.org/get-started/previous-versions/#v260). Project chỉ cần `torch`, không cần `torchvision` hoặc `torchaudio`.

### B. Chạy bằng CPU

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
```

CPU dùng FP32 và có thể chậm hơn nhiều. Khi chạy giao diện, chọn thiết bị `cpu`, phương pháp `greedy`, batch 1. Có thể tăng **Ngân sách thời gian (giây)** nếu sinh chưa xong trong giới hạn mặc định 120 giây.

Không chạy cả nhánh A và B. File `requirements-gpu.txt` dành cho quantization tùy chọn; **không cần cài** cho luồng mặc định `Quantization = none`.

## 4. Tải model và chuẩn bị database

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe databases/create_student_db.py
.\.venv\Scripts\python.exe -m pip check
```

Script tải mặc định `cycloneboy/SLM-SQL-1.5B` vào:

```text
models/
  SLM-SQL-1.5B/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    ...
```

Giữ nguyên tên thư mục mặc định để backend tự tìm model. Lệnh tải có thể chạy lại nếu lần trước gián đoạn. Không cần đặt `HF_HOME` hoặc tạo `.env` cho luồng này; chương trình không tự đọc `.env`.

Lệnh tạo database giữ nguyên file có sẵn; nếu chưa có thì tạo dữ liệu mẫu. Kết quả mong đợi có `checked 200 gold queries`. `pip check` cần báo không có dependency lỗi.

`config.yaml` mặc định:

```yaml
model:
  backend: huggingface
  model_id: cycloneboy/SLM-SQL-1.5B
  device: auto
  dtype: auto
  quantization: none
  local_files_only: true
```

Đây là phần `model` trong file hiện có, không phải toàn bộ file để ghi đè. `local_files_only: true` có nghĩa là phải tải model trước khi sinh SQL; khi chạy inference không tự tải weights còn thiếu.

## 5. Mở giao diện và chạy AI

```powershell
.\scripts\run_demo.bat
```

Hoặc dùng lệnh tương đương:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Mở [http://127.0.0.1:8501](http://127.0.0.1:8501). Giữ terminal đang chạy.

| Cấu hình giao diện | Giá trị để bắt đầu |
| --- | --- |
| Backend | `huggingface` |
| Model ID | `cycloneboy/SLM-SQL-1.5B` |
| Thiết bị | `auto`; hoặc chọn rõ `cuda`/`cpu` |
| Quantization | `none` |
| Phương pháp | `greedy` |
| Batch size | `1` |

1. Chọn **Dữ liệu sinh viên mẫu**.
2. Nhập **Có bao nhiêu sinh viên?**, hoặc chọn câu mẫu rồi bấm **Dùng câu hỏi mẫu**.
3. Bấm **Generate SQL**.
4. Xem SQL được chọn và bảng kết quả; câu đếm trên dữ liệu mẫu phải trả **16**.

Không cần chạy model server riêng: lần đầu bấm Generate SQL, app nạp model vào bộ nhớ. Các câu tiếp theo dùng model được cache trong phiên Streamlit. Trên máy kiểm chứng, 5 câu dễ mất trung bình khoảng 18 giây/câu, chưa gồm nạp model; đây không phải thời gian bảo đảm trên máy khác.

Có thể chọn `adaptive_sc` để sinh nhiều ứng viên hoặc tải file SQLite riêng, tối đa 20 MB. SQL thực thi thành công và độ đồng thuận cao không bảo đảm trả đúng ý câu hỏi.

**Tắt app:** nhấn `Ctrl+C` ở terminal. **Chạy lại lần sau:** vào thư mục ứng dụng và chạy `scripts\run_demo.bat`; không cần cài lại thư viện hoặc tải lại model.

## 6. Kiểm tra bằng terminal

### Kiểm thử phần mềm, không cần tải model

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/smoke_test.py --backend mock
```

Tại thời điểm viết hướng dẫn, bộ test có 67 bài đạt. Mock chỉ dùng đáp án fixture, không đo chất lượng AI.

### Kiểm tra AI thật

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test.py --backend huggingface --device auto --mode greedy
```

Kỳ vọng `PASS backend=huggingface` và SQL đếm bảng `students`. Script kiểm tra kết quả bằng 16, lưu trace tại `results/smoke_huggingface.json`. Nếu nhận lỗi assertion thì model trả khác đáp án; không xem đó là một lượt kiểm tra đạt.

### Đánh giá bộ câu hỏi

Chạy thử 5 câu:

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --backend huggingface --device auto --mode greedy --dataset data/evaluation_questions.json --limit 5
```

Chạy cả 100 câu đánh giá:

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluate --backend huggingface --device auto --mode greedy --dataset data/evaluation_questions.json
```

Báo cáo tự tạo trong thư mục có timestamp dưới `results/`, gồm `summary.json`, `summary.csv`, `predictions.jsonl`, `config.json`. Lượt 100 câu có thể chạy lâu. Nếu chạy CPU và chạm giới hạn thời gian, tăng `sampling.time_budget_seconds` trong `config.yaml` cho CLI.

Dataset tổng cộng **200 câu**: 100 demo trong `data/demo_questions.json` và 100 đánh giá trong `data/evaluation_questions.json`. Chi tiết tại [data/README.md](data/README.md). Đây là dữ liệu truy vấn/đánh giá, không phải pipeline huấn luyện; chạy evaluation không cập nhật weights.

## 7. Dùng model nhỏ hơn hoặc thử mock trước

### Model 0.5B

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py --model-id cycloneboy/SLM-SQL-0.5B
```

Trong UI, đổi **Model ID** thành `cycloneboy/SLM-SQL-0.5B`. Để CLI và các lần mở app sau cũng dùng model này, sửa `model.model_id` trong `config.yaml`. Bản nhỏ hơn có thể giảm yêu cầu bộ nhớ nhưng chất lượng cần kiểm tra lại; trong kiểm tra ban đầu của project, 0.5B còn nhầm một số câu đơn giản.

### Mock không cần weights/PyTorch

Chỉ làm bước clone, tạo venv và cài `requirements.txt`; tạo database rồi mở app như mục 5. Trước khi bấm Generate SQL, chọn **Backend = mock**, dùng câu từ dropdown. Mock chỉ hỗ trợ 100 câu demo, không hỗ trợ câu tự nhập ngoài fixture.

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
| --- | --- |
| Không tìm thấy `py` | Kiểm tra cài Python và mở terminal mới. Có thể dùng `python -m venv .venv` sau khi xác nhận `python --version` là 3.11/3.12. |
| Không tìm thấy `app.py` hoặc `requirements.txt` | Chuyển vào thư mục `adaptive-slm-sql` chứa các file đó. |
| PowerShell chặn activate | Gọi `.\.venv\Scripts\python.exe` như hướng dẫn, không cần activate. |
| `No module named ...` | Cài requirements bằng đúng Python trong `.venv`. Backend thật còn cần PyTorch và `requirements-model.txt`. |
| Không tải được model / thiếu file local | Chạy lại `scripts/download_model.py` khi có mạng; kiểm tra Model ID khớp thư mục `models/SLM-SQL-1.5B`. |
| `CUDA: False` | Kiểm tra `nvidia-smi`, driver, wheel PyTorch; có thể chọn CPU để chạy. Nếu đã cài wheel CPU, cài lại torch với `--force-reinstall` và đúng index CUDA tương thích máy. |
| CUDA hết bộ nhớ | Giữ greedy, batch 1; giảm Max new tokens/số dòng mẫu, dùng model 0.5B hoặc CPU. |
| Sinh chưa xong / không có SQL hoàn chỉnh | Tăng ngân sách thời gian; nếu output bị giới hạn token, tăng Max new tokens trong khả năng bộ nhớ. |
| Port 8501 đã dùng | Mở app đang chạy hoặc thêm `--server.port 8502` vào lệnh Streamlit và truy cập `http://127.0.0.1:8502`. |
| Terminal lỗi ký tự tiếng Việt | Chạy `$env:PYTHONUTF8='1'` trước lệnh Python. |
| Model chạy nhưng trả sai | Đối chiếu SQL với schema và đáp án; đây là lỗi chất lượng model, không được tính là đúng chỉ vì có bảng kết quả. |

## 9. Linux/macOS

Sau khi clone, vào thư mục chứa `app.py`. Ví dụ với Python 3.12 đã cài:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Linux CPU:**

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Linux NVIDIA:** thay lệnh CPU bằng lệnh CUDA 11.8 nếu phù hợp GPU/driver:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

**macOS:**

```bash
python -m pip install torch==2.6.0
```

Sau đó, với cả ba trường hợp:

```bash
python -m pip install -r requirements-model.txt
python scripts/download_model.py
python databases/create_student_db.py
python -m pytest -q
python -m streamlit run app.py --server.address 127.0.0.1
```

Project hiện hỗ trợ CPU/CUDA, chưa có backend Apple MPS. Hướng dẫn Linux/macOS chưa được chạy kiểm chứng trong phiên triển khai Windows này. Nếu Python/kiến trúc máy không có wheel tương ứng, kiểm tra nền tảng hỗ trợ của phiên bản PyTorch thay vì dùng package khác phiên bản một cách tự động.
