# Ma trận khảo sát tài liệu — Task 0.1

> Cập nhật lần cuối: ____  
> Cần xác nhận lại ở task 0.1 (tìm kiếm tháng 9/2026)

| Công trình | Mô hình | Dữ liệu huấn luyện | LoRA/RL | Chọn ứng viên | Adaptive | BIRD dev EX% | Ghi chú |
|---|---|---|---|---|---|---|---|
| **SLM-SQL** (Sheng & Xu, IJCNLP-AACL 2025) | Qwen2.5-Coder 0.5B/1.5B | SynSQL-Think-916K (SFT) + BIRD train (RL) | SFT đầy đủ + GRPO | SC + CSC (top-2 merge) | Không (fixed 16/64) | 0.5B: 56.87 (64), 53.22 (16); 1.5B: 67.08 (64), 64.84 (16) | Baseline chính; pass@64=82.64%; CSC chỉ khi phiếu không nhất quán |
| **CSC-SQL** | — | — | — | Hợp nhất-hiệu chỉnh top-2 | Không | — | Kỹ thuật CSC được SLM-SQL kế thừa |
| **SQL-R1** | — | — | RL | — | — | — | RL cho Text-to-SQL |
| **Think2SQL** | — | — | — | — | — | — | CoT reasoning cho SQL |
| **Adaptive-Consistency** (Aggarwal 2023) | GPT-3.5/4 (LLM lớn) | — | Không | SC + Beta stopping | Có (Beta posterior) | — | Adaptive cho suy luận nói chung, không chuyên Text-to-SQL |
| **Knowing When to Stop** (arXiv 2607.03991) | GPT-4.1 | — | Không | SC | Có (XGBoost/LR/TCN) | 83 câu BIRD finance | Adaptive cho Text-to-SQL nhưng dùng LLM lớn, 83 câu, không SLM fine-tune, không CSC |
| **FINER-SQL** | — | — | — | — | — | — | — |
| **SQuaD-SQL** | — | — | — | — | — | — | — |

## Khoảng trống chính

1. **G1**: Chưa có adaptive sampling cho SLM fine-tuned (chỉ có cho LLM lớn)
2. **G2**: Chưa có routing 3 hành động (dừng/thêm mẫu/gọi CSC)
3. **G3**: Chưa có phân tích LoRA vs full SFT cho Text-to-SQL SLM
4. **G4-G7**: CoT trade-off, chất lượng ứng viên, khái quát hóa, từ chối khi không chắc

> **TODO**: Điền đầy đủ sau khi đọc kỹ từng bài (task 0.1 thực tế)
