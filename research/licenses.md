# Bảng giấy phép dữ liệu, mô hình và thư viện

Đã kiểm tra các thẻ nguồn công khai vào **19/10/2026**. Giấy phép áp dụng cho đúng tài nguyên được tải; giấy phép của repository không tự động thay thế giấy phép của weights hoặc dataset trên Hub.

| Tài nguyên | Giấy phép đã thấy | Điều kiện cần giữ trong project | Trích dẫn / provenance |
|---|---|---|---|
| BIRD benchmark | CC BY-SA 4.0 trên trang benchmark | Giữ attribution, chia sẻ lại theo điều khoản khi phân phối dữ liệu phái sinh; không đóng gói DB vào repo | BIRD official homepage |
| Spider dataset | CC BY-SA 4.0 | Giữ attribution; phân biệt dataset license với license của code scorer | Spider official homepage |
| `seeklhy/SynSQL-2.5M` | Apache-2.0 theo dataset card đã kiểm tra | Giữ NOTICE/LICENSE và provenance; đây là nguồn SynSQL, không gọi là dataset Gretel | Dataset card |
| `cycloneboy/SynsQL-Think-916k` | CC-BY-NC-4.0 | Chỉ dùng trong phạm vi cho phép phi thương mại; không mặc định phát hành cho mục đích thương mại | Dataset card của tác giả SLM-SQL |
| `cycloneboy/SynsQL-Merge-Think-310k` | CC-BY-NC-4.0 | Như trên; dataset merge khác dataset generator | Dataset card của tác giả SLM-SQL |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Apache-2.0 theo model card | Giữ attribution/license của Qwen; kiểm tra điều khoản thay đổi nếu đổi revision | Model card |
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Apache-2.0 theo model card | Như trên | Model card |
| `cycloneboy/SLM-SQL-1.5B` | Theo model card/revision cụ thể đã tải | Lưu model ID, revision và card cùng cache; không suy ra license từ Qwen base | Model card tác giả |
| `cycloneboy/CscSQL-Merge-Qwen2.5-Coder-1.5B-Instruct` | CC-BY-NC-4.0 theo model card | Đây là ràng buộc quan trọng của luồng 09–11; adapter dự án kế thừa yêu cầu kiểm tra quyền phân phối của base | Model card merger |
| Unsloth | Apache-2.0 | Giữ license của thư viện khi phân phối script | PyPI/repository |
| vLLM | Apache-2.0 | Giữ license của thư viện | Repository |
| TRL | Apache-2.0 | Giữ license của thư viện | Repository |

Các file BIRD/Spider, weights upstream và raw SynSQL không được commit vào repo. Script đóng gói adapter phải nhận license từ người dùng và không tự tuyên bố rằng model là thương mại tự do. Trước khi upload public adapter, cần kiểm tra lại license của **đúng revision** generator, merger, dataset train và model base.

Nguồn chính được dùng khi audit:

- SLM-SQL: `https://arxiv.org/abs/2507.22478`
- CSC-SQL: `https://arxiv.org/abs/2505.13271`
- SynsQL-Think: `https://huggingface.co/datasets/cycloneboy/SynsQL-Think-916k`
- SynsQL-Merge-Think: `https://huggingface.co/datasets/cycloneboy/SynsQL-Merge-Think-310k`
- SynSQL-2.5M: `https://huggingface.co/datasets/seeklhy/SynSQL-2.5M`
