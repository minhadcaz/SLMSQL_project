# Mở rộng sampling cho SLM-SQL — kế hoạch và triển khai 20/10/2026

Mục tiêu: bổ sung phép so sánh generator và cách chọn ứng viên vào luồng 09–12,
đồng thời đo riêng tác động LoRA merger và tác động adaptive inference. Giữ
checkpoint SLM-SQL đã SFT + GRPO của tác giả; adapter tiếp tục huấn luyện trên
merger. Phần mở rộng không đòi hỏi chạy lại notebook 00–08.

## Ma trận thực nghiệm

| Phương pháp | Generator candidates | Bộ chọn cuối | Câu hỏi nghiên cứu |
|---|---|---|---|
| Greedy | Một request riêng, temperature=0 | SQL duy nhất | Mốc inference ít mẫu |
| pass@k | Prefix k của stochastic cache | Oracle dùng gold khi chấm | Generator có sinh đáp án đúng không? |
| self_consistency@k | Cùng prefix k | Nhóm execution đông nhất | SC chọn được bao nhiêu đáp án? |
| SLM-SQL@k / author CSC@k | Cùng prefix k | Author merger trên top-2 nếu bất đồng | Mốc đầy đủ có CSC |
| LoRA CSC@k | Cùng prefix k | LoRA merger, cùng seed/budget | Tác động riêng của adapter |
| Adaptive SC | Draws độc lập, sinh theo vòng | Execution vote khi dừng | Chất lượng so với số mẫu đã dùng |
| Adaptive + author CSC | Cùng adaptive cache | Author merger trên top-2 | Mốc để tách tác động sampling |
| Adaptive + LoRA CSC | Cùng adaptive cache | LoRA merger, cùng seed/budget | Hiệu quả kết hợp |

Theo §4.3.2 bài SLM-SQL, self_consistency@k là vote và SLM-SQL@k có CSC.
Không dùng tên SLM-SQL@k cho SC đơn thuần. SFT là giai đoạn huấn luyện của
checkpoint, không phải một tên khác của SC; ở đây không có ablation SFT-only mới.
Prompt dựng lại theo ý nghĩa Appendix E, nên chưa tuyên bố tái lập điểm công bố.

## Kế hoạch đã thực hiện

1. [x] **Config và preflight.** Thêm sweep `sampling_k_values` full
   `[1,2,4,8,16,32,64]`, pilot `[1,2,4]`; validate giới hạn và thống nhất CLI/notebook.
   `plan` hiển thị greedy/adaptive và ngân sách CSC trước khi dùng GPU.
2. [x] **Greedy riêng.** Sinh với temperature=0, top_p=1, n=1; cache/seed riêng.
   Không gọi stochastic candidate đầu là greedy.
3. [x] **Fixed-k.** Sinh n mẫu stochastic một lần mỗi seed; SC và hai CSC dùng
   cùng prefix k, top-2 và ngân sách merge. k=1 không load merger.
4. [x] **Adaptive online.** Tái sử dụng Beta hoặc margin/entropy từ nhánh trước
   trên generator SLM-SQL. Policy chỉ nhận observed execution votes, không gold.
   Chỉ sinh vòng tiếp khi chưa dừng. Lưu round seeds, mẫu đã dùng và stop reason.
   Sửa chuyển fingerprint lồng trong execution sang format policy đang đọc.
5. [x] **Adaptive CSC.** Chạy author và LoRA merger trên cùng candidate set đã
   dừng. Cache riêng fixed/adaptive; tách process generator/merger để nhả VRAM.
6. [x] **Scoring CPU.** EX, pass@k prefix coverage, unbiased estimated_pass@k,
   top-2 coverage, slices, paired delta/CI/McNemar; JSON/CSV và từng câu.
   Chi phí gồm generator và merger. Không suy ra thời gian k nhỏ từ full-n batch.
7. [x] **Freeze và nhiều seed.** Khóa source, data, adapter, config gồm policy,
   toàn bộ k và seed trước final LoRA predictions. Bootstrap theo câu đã average
   các seed; mỗi seed vẫn có kiểm định cặp riêng. Một adapter dùng chung các seed.
8. [x] **Nối notebook.** 09 tạo đầy đủ baselines/caches, 10 train merger adapter,
   11 freeze/merge final và chấm toàn matrix, 12 chọn policy validation và đọc
   adaptive online/offline report; final stage không gọi generator lại.
9. [x] **Kiểm thử và hướng dẫn.** CPU integration dùng SQLite thật và backend
   giả; kiểm tra early stopping, ties/errors, greedy độc lập, resume, provenance,
   k=1, paired author/LoRA, CLI từ cache và multi-seed. Xem số test và kiểm tra
   notebook thực tế trong [verification.md](../results/verification.md).

## Cách chạy

Tạo config pilot và prepare theo [CSC_LORA.md](CSC_LORA.md), rồi:

```bash
# Inference session 09: generator, author CSC và baseline suite
python -m research.slm_sql --config configs/csc_pilot.json sampling-baseline --seed 42
# Cùng session: generate/execute train và validation; sau đó pairs
# Training session 10:
python -m research.slm_sql --config configs/csc_pilot.json train
# Sau training, tạo validation fixed merge checkpoints cho từng adaptive k.
for k in 2 4; do
  python -m research.slm_sql --config configs/csc_pilot.json merge --split validation --seed 42 --variant author --k "$k"
  python -m research.slm_sql --config configs/csc_pilot.json merge --split validation --seed 42 --variant lora --k "$k"
done
# Validation policy sau khi có cả author/LoRA checkpoints:
python -m research.slm_sql --config configs/csc_pilot.json adaptive-sweep --split validation --seed 42
# Inference session 11/12, trước final LoRA predictions:
python -m research.slm_sql --config configs/csc_pilot.json freeze --seeds 42 --ks 1 2 4
python -m research.slm_sql --config configs/csc_pilot.json sampling-final --split evaluation --seed 42
python -m research.slm_sql --config configs/csc_pilot.json adaptive-merge --split evaluation --seed 42 --variant author
python -m research.slm_sql --config configs/csc_pilot.json adaptive-merge --split evaluation --seed 42 --variant lora
python -m research.slm_sql --config configs/csc_pilot.json adaptive-csc-eval --split evaluation --seed 42
python -m research.slm_sql --config configs/csc_pilot.json report-suite --seeds 42
```

Full run tạo config/experiment mới, không `--pilot`; chạy baseline cho 42/43/44,
freeze cả ba seed rồi final từng seed và aggregate. Notebook 09–12 dùng `CSC_EVAL_SEEDS`
(danh sách dấu phẩy), mặc định full `42,43,44`, pilot `CSC_SEED`. Để trống `CSC_K`
để chấm mọi k; có thể giới hạn một prefix nếu đã khai báo trong freeze.

`sampling-baseline` luôn lưu full-n author CSC cho training, kể cả sweep không
chứa n. `sampling-final` chỉ gọi merger và scoring. Thiếu generator artifacts thì
hoàn tất 09; không sinh thêm trong 11 để lấp khoảng trống. Cache hoàn tất và hợp
protocol được dùng lại; thiếu/hỏng/khác config, model hoặc packages phải báo lỗi.
Cache adaptive từ phiên bản logic cũ bị từ chối để tránh dùng kết quả sai.

## Cách đọc kết quả

- `reports/evaluation-seed42.sampling.baseline.{json,csv}`: author baselines.
- `reports/evaluation-seed42.sampling.comparison.{json,csv}`: full comparison.
- `reports/evaluation.sampling.comparison.multi-seed.{json,csv}`: EX mean/std,
  cost và question-level CI qua các seed. Baseline có file multi-seed tương tự.
- `summary.methods`: các phương pháp thực sự chọn một SQL, kèm costs/slices.
  `summary.fixed[k]`: oracle, SC, author/LoRA CSC và alias SLM-SQL@k đúng nghĩa.
- `paired_comparisons`: LoRA-vs-author, CSC-vs-SC, adaptive-vs-fixed và SC-vs-greedy.
  `records`: từng câu, SQL, correctness, oracle, costs và adaptive rounds.

pass@k prefix trả lời “prefix đang xét có đáp án đúng không?”. Ước lượng tổ hợp
`1-C(n-c,k)/C(n,k)` dùng toàn bộ n được báo riêng, không thay thế prefix coverage.
Oracle dùng gold chỉ khi scoring và không được xếp hạng như bộ chọn SQL triển khai.

Đối chiếu LoRA với author trên cùng fixed k để đo adapter; đối chiếu adaptive SC
với fixed SC để đo sampling; adaptive LoRA với adaptive author tách tác động
adapter trong điều kiện adaptive. So sánh EX cùng mean samples/tokens; không tự
nhận “cùng chi phí” chỉ vì có cùng max_k. Các kiểm định được báo riêng từng phép
so sánh; chưa điều chỉnh multiple comparisons, cần định trước phép so sánh chính.

Fixed-prefix k<n không có generation latency được đo riêng, nên report để null.
Adaptive lưu thời gian sequential generate/execution/vote (không gồm load model,
đọc schema/gold hoặc CSC); execution có thể trúng cache. Fixed batch timing và
adaptive sequential timing là workload khác nhau. Cần đo cùng batch, trạng thái
cache và phần cứng trước khi kết luận speedup end-to-end. Tokens hiện báo output
tokens; prompt_tokens vẫn có trong candidates nhưng chưa tổng hợp thành chi phí.

## Phần cần thực nghiệm trên GPU/BIRD thật

- [ ] Chạy pilot matrix, kiểm tra context/VRAM và khả năng chạy hai dependency profiles.
- [ ] Tạo đủ train/validation pairs và huấn luyện merger adapter trên CUDA.
- [ ] Chọn/fix policy trên train/validation, không dùng final dev để tune ngưỡng.
- [ ] Chạy full matrix đã freeze qua 42/43/44; xem EX, mean samples/tokens và regression.
- [ ] Đo latency cùng workload và review lỗi JOIN/aggregate để diễn giải cơ chế.

Các test CPU xác nhận logic và hợp đồng dữ liệu. Chưa có EX hay latency thực nghiệm
của phần mở rộng này. Code cho phép kiểm chứng hai hướng cải tiến SLM-SQL; kết quả
âm (không tăng EX, dừng quá sớm hoặc overhead làm chậm) vẫn phải được báo đầy đủ.
