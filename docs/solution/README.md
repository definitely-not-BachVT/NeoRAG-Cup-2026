# Giải pháp RAG cho NeoRAG Cup 2026

Pipeline trả lời câu hỏi về CLB Lập Trình PTIT, CLB Multimedia PTIT và CLB Nhà Sáng Tạo Game PTIT, tối ưu theo bộ chấm của cuộc thi (nửa retrieval, nửa generation). Mã nguồn nằm ở `src/my_solution/`, chatbot ở `app/chat.py`. Nhật ký đầy đủ mọi quyết định và số liệu: [nhat-ky-cai-tien.md](nhat-ky-cai-tien.md).

**Kết quả Public Test tốt nhất: raw 90.14** (retrieval 95.81, generation 84.47). Cùng cấu hình chạy lại cho 86.04 và 90.14: các thành phần LLM (xếp lại, đọc corpus, sinh câu trả lời) không tất định và `claude-haiku-5-5` không cho đặt `temperature`, nên **kỳ vọng thực tế khoảng 86–90**.

## Chạy nhanh

```bash
# 1. Cài đặt (có reranker và embedding chạy trên máy; PyTorch CUDA nếu có GPU NVIDIA)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python scripts/install.py --local

# 2. Điền key
cp .env.example .env                                       # Windows: copy .env.example .env
#   mở .env, điền OPENAI_API_KEY

# 3. Chạy pipeline và tạo file nộp
neorag predict public_test --pipeline my_solution.pipeline:MyPipeline
python scripts/check_submission.py public_test            # kiểm tra trước khi nộp
#   nộp outputs/public_test_submission.zip

# 4. Chatbot
python -m streamlit run app/chat.py
```

> **Quan trọng:** lệnh `neorag run` / `neorag predict` mặc định chạy **baseline của BTC**. Luôn thêm `--pipeline my_solution.pipeline:MyPipeline`. Chatbot đã mặc định dùng pipeline của giải pháp.

| Việc | Lệnh |
|---|---|
| Chạy và chấm train (có judge, tốn API) | `neorag run train --pipeline my_solution.pipeline:MyPipeline` |
| Chấm train không qua judge (retrieval, String Presence, ROUGE-L) | `neorag run train --no-judge --pipeline my_solution.pipeline:MyPipeline` |
| Đo riêng retrieval trên train, không API nếu có cache | `python scripts/eval_retrieval.py --retriever full-a4-a5 --by type` |
| Đo generation trên mẫu train (mặc định chỉ ước tính chi phí) | `python scripts/eval_generation.py --sample 30` (thêm `--yes` để chạy thật) |
| Ngày Private Test | `neorag predict private_test --pipeline my_solution.pipeline:MyPipeline`, rồi `python scripts/check_submission.py private_test` |
| Chạy test | `pytest` (132 test, chạy offline, LLM/embedding/reranker đều là bản giả) |

## Yêu cầu và chi phí

- Python 3.10+. Reranker `BAAI/bge-reranker-v2-m3` (khoảng 2.3 GB) tải về lần chạy đầu. **Nên có GPU**: trên RTX 3050 6 GB mỗi câu rerank khoảng 1 giây, trên CPU khoảng 27 giây.
- Mỗi câu hỏi gọi LLM 4 lần (viết lại, xếp lại A4, đọc corpus A5, sinh câu trả lời), khoảng **$0.003/câu**, tức khoảng $0.12 cho 42 câu khi cache trống. Mọi lời gọi đều được cache trong `.cache/` (đã `.gitignore`), nên chạy lại cùng câu hỏi trong cùng ngày không tốn thêm.
- Thời gian khi cache trống: Public Test 42 câu mất khoảng 28 phút (median 35 giây/câu khi gateway chậm; 3–5 giây/câu khi gateway nhanh). Pipeline chạy tuần tự từng câu.
- Giới hạn gateway: 100 request/phút, 8 request đồng thời, ngân sách 5 USD/key. Chatbot chỉ nên chạy trên máy mình, không mở công khai (người lạ hỏi sẽ tiêu ngân sách).

## Kiến trúc

```
câu hỏi (có thể không dấu, viết tắt)
  │
  ├─ Viết lại (LLM)           bản có dấu, giải nghĩa viết tắt + câu hỏi con nếu hỏi nhiều ý   query.py
  │
  ├─ Tìm 30 ứng viên          mỗi câu (gốc, viết lại, câu con) chạy hybrid, gộp RRF             retrievers.py
  │     BM25 tự cài (bỏ dấu, viết tắt, bigram)  ⊕  bge-m3 qua gateway   (RRF 1.5 : 1)
  │
  ├─ Cross-encoder (GPU)      bge-reranker-v2-m3, gộp RRF với bảng cũ                           rerank.py
  ├─ A4  LLM xếp lại          15 đoạn đầu, listwise, gộp RRF 1 : 1                              llm_rerank.py
  ├─ A5  LLM đọc cả corpus    118 đoạn (~20 nghìn token) vào một prompt, chọn top-10,           corpus_rank.py
  │                           gộp RRF 1 : 1; quá 20 giây hoặc lỗi thì dùng kết quả A4
  │
  └─ Sinh câu trả lời         8 đoạn trong prompt (prompt v2), nộp 10 id                        prompts.py, pipeline.py
```

Các lựa chọn thiết kế chính:

- **Corpus đã được BTC chia sẵn** 118 đoạn tự đủ ngữ cảnh (trung bình 380 ký tự), nên không chunk lại. Mỗi đoạn được biểu diễn là `"{CLB}. {chủ đề}. {nội dung}"` để mọi đoạn đều mang tên CLB.
- **BM25 và dense bổ trợ nhau:** BM25 mạnh với câu không dấu/teencode (bỏ dấu cả hai phía, giải nghĩa viết tắt theo từng token, thêm bigram âm tiết: +4.7 điểm so với chỉ unigram); bge-m3 bắt được ý gần nghĩa. Trọng số 1.5 : 1.
- **Gộp tín hiệu bằng RRF** thay vì thay thế. `reserve=2` giữ chỗ cho top-2 của mỗi câu hỏi con, để các đoạn chung chung không đẩy đoạn đúng ra khỏi ứng viên.
- **A4 và A5 là hai bước dùng LLM sắp xếp lại** (reranker listwise và chọn trên toàn corpus). Mọi lỗi (JSON hỏng, id lạ, timeout) quay về kết quả bước trước, nên pipeline không chết giữa chừng.
- **Prompt sinh câu trả lời** (v2) rút ra từ log chấm: nộp đủ 10 id (không bị phạt thêm id), câu trả lời ngắn, từ chối ngắn gọn khi không có thông tin.

## Kết quả qua các lần nộp Public Test

| Lần | Thay đổi | Retrieval | Raw |
|---|---|---:|---:|
| Baseline BTC | embedding + top-k | 42.31 | 52.40 |
| 1 | hybrid BM25 + bge-m3, prompt v1 | 64.03 | 69.06 |
| 2 | + viết lại câu hỏi, cross-encoder | 81.26 | 80.78 |
| 3 | prompt v2 | 81.26 | 81.44 |
| 4 | sửa bước gộp (`reserve=2`) | 87.83 | 85.20 |
| 5 | + 30 ứng viên + doc2query (A3) | 86.97 | 84.86 |
| 6 | chỉ 30 ứng viên (tắt A3) | 87.94 | 85.55 |
| 7 | + A4 (LLM xếp lại) | 92.01 | 87.22 |
| 8 | + A5 (LLM đọc corpus) | 93.84 | 86.04 |
| 9 | cấu hình cuối, chạy cache trống | **95.81** | **90.14** |

Lần 8 và 9 cùng cấu hình: chênh lệch cho thấy mức dao động giữa các lần sinh lại.

**Đã thử và bỏ** (đều đo bằng cửa đặt trước, chi tiết trong nhật ký): doc2query (tăng train, giảm Public Test), Step-Back (train 93.2 → 92.6), prompt v3 (String Presence −3.9), embedding `AITeamVN/Vietnamese_Embedding_v2` chạy trên máy (full 93.2 → 92.3; phụ thuộc dấu rất mạnh: bỏ dấu câu viết lại làm dense mất khoảng 30 điểm).

## Cấu trúc mã nguồn

| Đường dẫn | Nội dung |
|---|---|
| `src/my_solution/pipeline.py` | `MyPipeline`: ghép các bước, các cờ bật/tắt (mặc định là cấu hình cuối) |
| `src/my_solution/text.py` | bỏ dấu, giải nghĩa viết tắt, nhận diện CLB, tokenize + bigram |
| `src/my_solution/retrievers.py` | BM25, dense, hybrid, `MultiQueryRetriever`, `rrf` |
| `src/my_solution/query.py` | viết lại câu hỏi và tách câu hỏi con |
| `src/my_solution/rerank.py`, `llm_rerank.py`, `corpus_rank.py` | cross-encoder, A4, A5 |
| `src/my_solution/prompts.py` | prompt sinh câu trả lời v1/v2/v3 (mặc định v2) |
| `src/my_solution/enrich.py`, `stepback.py` | A3 (doc2query) và Step-Back, đều tắt mặc định |
| `src/my_solution/trace.py`, `history.py`, `chat_view.py` | ghi vết từng bước, lịch sử và HTML cho chatbot |
| `app/chat.py`, `.streamlit/config.toml` | chatbot Streamlit |
| `scripts/` | `eval_retrieval.py`, `eval_generation.py`, `check_submission.py`, `analyze_predictions.py` |
| `tests/test_my_solution.py` | test của giải pháp |

## Chatbot

Chạy `python -m streamlit run app/chat.py` rồi mở địa chỉ hiện trong terminal (mặc định `localhost:8501`). Lần đầu mất khoảng 2 phút để nạp model và xây chỉ mục.

- **Thanh bên:** lịch sử câu hỏi theo ngày, mở lại không gọi API, xóa từng mục, nút "Hỏi mới". Lưu ở `.cache/chat_history.json`.
- **Cột "quá trình trả lời":** thanh các bước kèm thời gian từng bước; câu gốc so với bản viết lại (chữ đỏ là dấu được khôi phục, gạch chấm là từ viết tắt được giải nghĩa); các câu hỏi con; nguồn tham khảo theo màu CLB (đoạn nào đã đưa vào prompt, đoạn nào dự phòng).
- Mỗi câu hỏi được trả lời độc lập (pipeline không giữ ngữ cảnh hội thoại). Câu hỏi tối đa 600 ký tự.
- **Chạy từ thư mục gốc của repo.** Cache LLM, lịch sử và `.streamlit/config.toml` (màu, theme) đều tính theo thư mục đang đứng; chạy từ nơi khác app vẫn chạy nhưng mất cache (chậm, tốn API) và sai màu.
- Lỗi được báo gọn trên giao diện, chi tiết ghi ở terminal: không nạp được pipeline (thiếu key, GPU hết bộ nhớ), một câu không trả lời được (mạng, hạn mức; câu đó không lưu vào lịch sử), không ghi được lịch sử (câu vẫn hiện, kèm ghi chú chưa lưu).
- Đổi pipeline: `NEORAG_PIPELINE=module:Class`. Pipeline khác vẫn chạy, chỉ không có chi tiết từng bước.

## Danh sách việc cho ngày Private Test (08/11)

1. BTC đăng `data/private_test/questions.jsonl` đầu ngày: tải về đặt đúng đường dẫn. **Không sửa mã nguồn** (quy định).
2. Xóa `.cache/` nếu muốn chạy đúng điều kiện máy sạch, hoặc giữ nguyên (cache chỉ ảnh hưởng tốc độ và chi phí, không ảnh hưởng đúng sai).
3. Chạy `neorag predict private_test --pipeline my_solution.pipeline:MyPipeline` ngay khi có đề (khoảng 15–30 phút).
4. `python scripts/check_submission.py private_test`: đủ 42 câu, đủ 10 id, không có câu rỗng; xem số câu quá 60 giây và cảnh báo câu dự phòng.
5. Nộp `outputs/private_test_submission.zip`. Hệ thống chỉ tính **lần nộp gần nhất** (tối đa 5 lần, bài Failed không tính lượt); nếu nộp lại, nộp bản mình muốn tính cuối cùng.
6. Pipeline in cảnh báo `A5 lùi về A4 ...` khi bước A5 quá 20 giây; vài câu như vậy là bình thường, chỉ giảm nhẹ chất lượng.

## Giới hạn đã biết

- Điểm dao động theo lần chạy (retrieval ±2, Abstention có thể đổi 100 ↔ 66.67 vì chỉ chấm trên 3 câu không có đáp án). Không nên so hai lần nộp chênh dưới vài điểm.
- Độ trễ cao khi gateway chậm (4 lời gọi LLM mỗi câu, tuần tự).
- Mọi bước đều cần gateway (LLM và embedding bge-m3). Khi một bước LLM phụ trợ (viết lại, A4, A5) lỗi hoặc quá hạn, pipeline lùi về kết quả bước trước; riêng sinh câu trả lời lỗi thì dùng câu dự phòng.
- Dữ liệu bổ sung ngoài `corpus.jsonl`: không dùng.

## Tuân thủ quy định

- Mô hình ngôn ngữ duy nhất: `claude-haiku-5-5` qua gateway. Embedding `bge-m3` qua gateway; reranker `BAAI/bge-reranker-v2-m3` open-source chạy trên máy (cho phép ở `docs/rules.md` mục 1).
- Không gán nhãn thủ công, không hard-code câu trả lời hay `retrieved_ids`. Câu hỏi Public Test chỉ dùng để nộp và chẩn đoán, không dùng để chỉnh tham số theo từng câu; ví dụ trong chatbot lấy từ tập train.
- Key chỉ nằm trong `.env` (đã `.gitignore`).
