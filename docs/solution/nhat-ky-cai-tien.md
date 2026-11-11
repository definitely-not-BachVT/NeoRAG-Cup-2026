# Nhật ký cải tiến hệ thống RAG – NeoRAG Cup 2026

Tài liệu này ghi lại toàn bộ quá trình xây dựng và cải tiến pipeline trong `src/my_solution/`: các quyết định, lý do đằng sau, số liệu đo được, các thử nghiệm thất bại và những bài học rút ra. Nó là nguồn chính cho phần trình bày (pitching) và README giải pháp.

Mọi con số retrieval "trên train" là điểm `mean(Recall@5, MRR@5, NDCG@5) × 100` trên 122 câu train có `relevant_ids`, đo bằng `scripts/eval_retrieval.py` (dùng đúng hàm `score_query` và cấu hình của BTC). Mọi con số "Public Test" là log chấm chính thức trên trang thi.

---

## 1. Bài toán và cách chấm điểm

### 1.1. Dữ liệu
- **Corpus:** 118 đoạn văn (id 1–78 CLB Lập Trình PTIT, 79–84 cộng đồng S2B, 85–96 CLB Multimedia PTIT, 97–118 CLB Nhà Sáng Tạo Game PTIT). Mỗi đoạn dài 122–709 ký tự, tổng khoảng 45 nghìn ký tự.
  → **Không cần chunking**: mỗi đoạn đã là một đơn vị thông tin trọn vẹn và đủ ngắn.
- **Train:** 126 câu, có nhãn loại câu hỏi: không dấu/viết tắt 26, diễn đạt khác 22, nhiều đoạn 20, so sánh 20, thời gian 12, không có đáp án 11, trực tiếp 8, thiếu dữ liệu 5, mâu thuẫn nguồn 2.
  - 50% câu cần từ 2 đoạn trở lên; 13 câu cần đoạn của nhiều CLB.
- **Public Test:** 42 câu, khoảng **một nửa viết không dấu hoặc tiếng lóng** (`clb da phuong tien`, `clb mm`, `clb lam game cua hv`, `clb code`). Đây là đặc điểm quyết định nhiều lựa chọn kỹ thuật bên dưới.

### 1.2. Cách tính điểm (đọc từ `src/neorag/` và `configs/eval.toml`)
- **Raw score = 50% Retrieval + 50% Generation.**
- Retrieval = trung bình Recall@5, MRR@5, NDCG@5. Mỗi điểm retrieval tăng thêm cộng thẳng 0.5 điểm raw.
- Generation = trung bình String Presence, ROUGE-L, Groundedness, Response Relevancy, 1 − Noise Sensitivity, Abstention.

Những chi tiết của bộ chấm ảnh hưởng trực tiếp đến thiết kế:

| Chi tiết | Hệ quả cho thiết kế |
|---|---|
| Groundedness chấm trên văn bản của **tất cả** `retrieved_ids` đã nộp | Luôn nộp đủ 10 id: Recall/MRR/NDCG chỉ xét top-5 nên không bị phạt, còn ngữ cảnh để judge kiểm chứng thì rộng hơn |
| String Presence so khớp **nguyên văn** key fact sau khi chuẩn hóa | Giữ nguyên ngày tháng, con số, tên riêng, link như trong tài liệu |
| Noise Sensitivity gắn nhãn `incorrect` cho câu "sai, mâu thuẫn, **hoặc là thông tin không có trong tham chiếu**" | Một câu **đúng** nhưng lấy từ đoạn không thuộc gold vẫn bị tính là sai → không nói thừa |
| Response Relevancy: judge sinh ngược 3 câu hỏi từ câu trả lời, so cosine (bge-m3) với câu hỏi gốc | Câu trả lời lan man → câu hỏi sinh ngược lệch → điểm thấp. Câu hỏi gốc không dấu là giới hạn trần không kiểm soát được |
| Câu **không có đáp án** chỉ được chấm Abstention; Groundedness/Noise/Relevancy chỉ tính trên câu có đáp án | Câu từ chối nên ngắn và rõ; sự thừa thãi chỉ gây hại ở câu có đáp án |
| `claude-haiku-5-5` không nhận `temperature`, chấm lại có thể lệch ±1 điểm | Chênh lệch dưới 1 điểm giữa hai lần nộp chưa chắc là thật |

### 1.3. Ràng buộc
- LLM duy nhất: `claude-haiku-5-5` qua gateway; embedding `bge-m3` qua gateway hoặc model open-source không sinh văn bản chạy trên máy.
- Ngân sách 5 USD cho cả cuộc thi; 100 request/phút; tối đa 8 request đồng thời.
- Interface `RAGPipeline`: `MyPipeline()` phải khởi tạo được không tham số; `run(query: str)` chỉ nhận một câu hỏi.
- Không được sửa code trong ngày Private Test (08/11) → pipeline phải chạy ổn định, không phụ thuộc thao tác tay.

---

## 2. Kiến trúc cuối cùng

```
câu hỏi
  │
  ├─► QueryRewriter (query.py)          1 lời gọi LLM: viết lại có dấu + tách câu hỏi con
  │       QueryPlan(original, rewrite, subqueries)
  │
  ├─► MultiQueryRetriever (retrievers.py)
  │       với MỖI câu trong {gốc, viết lại, các câu con}:
  │           HybridRetriever = BM25 (bỏ dấu, viết tắt, bigram) ⊕ bge-m3, gộp RRF (1.5 : 1)
  │       gộp các bảng xếp hạng bằng RRF → 30 ứng viên (bước 9)
  │       (luôn giữ chỗ cho top-2 của mỗi câu con – bước 8)
  │
  ├─► RerankedRetriever (rerank.py)     chạy trên GPU, ~1 giây/câu
  │       bge-reranker-v2-m3 chấm (bản viết lại / từng câu con, đoạn), lấy max
  │       câu không có câu con: gộp bảng retriever và bảng reranker bằng RRF (1 : 1)
  │       câu có câu con:       xếp hoàn toàn theo reranker (bước 8)          → top-10
  │
  ├─► ListwiseReranker (llm_rerank.py)  A4, 1 lời gọi LLM: xếp lại 15 đoạn đầu, gộp RRF (1 : 1) với bảng cũ
  │
  └─► claude-haiku-5-5 (prompts.py)     1 lời gọi LLM: sinh câu trả lời từ 8 đoạn đầu (prompt v2)
```

Đây là cấu hình của lần nộp 7 (**87.22**, bản tốt nhất; A4 bật mặc định). A3 (doc2query, bước 10) có trong code nhưng **tắt mặc định** vì làm giảm retrieval trên Public Test; A2 (`cover`) cũng tắt.

Mỗi câu hỏi tốn **3 lời gọi LLM** (viết lại, xếp lại A4, sinh câu trả lời) và khoảng 2 request embedding. Mọi lời gọi đều được cache trong `.cache/` (SQLite), nên chạy lại không tốn thêm tiền.

---

## 3. Quá trình cải tiến theo từng bước

### Bước 0 – Phân tích trước khi code
Trước khi viết dòng code nào, nhóm đọc toàn bộ bộ chấm và phân loại dữ liệu (mục 1). Hai kết luận định hướng cả quá trình:
1. Câu không dấu/tiếng lóng chiếm khoảng một nửa Public Test → **xử lý tiếng Việt không dấu là ưu tiên số một**.
2. Một nửa số câu cần nhiều đoạn → cần cơ chế đưa đoạn của từng CLB vào top-5.

### Bước 1 – Xử lý văn bản (`text.py`)
- `strip_accents`: chuẩn hóa NFD rồi bỏ các ký tự dấu rời (Unicode category `Mn`). Riêng `đ` là chữ cái độc lập nên phải thay tay.
- `flatten`: dùng `neorag.text.normalize` của BTC (thống nhất `hoà` = `hòa`), bỏ dấu, **thay dấu câu bằng khoảng trắng** (không xóa hẳn, để `vay,thi` không dính thành `vaythi`).
- `ABBREVIATIONS`: từ điển teencode chung (`ko`, `dk`, `tv`, `hv`, `bcn`, `t8`...). Thay **theo từng token**, không thay chuỗi con, để `k` trong `k26` không bị thay nhầm.
  - Quan sát từ dữ liệu: `dk` luôn mang nghĩa "đăng ký"; "được" được viết là `dc`. Vì vậy bỏ cơ chế ghép cụm hai token.
  - Nguyên tắc: chỉ thêm teencode **chung**, không thêm luật nhắm vào một câu test cụ thể (luật cấm hard-code).
- `detect_clubs`: so khớp **nguyên cụm, có khoảng trắng hai đầu** (`mpc` không khớp `mpcx`, `mmm` không khớp `mm`). Không chắc thì trả về rỗng, không đoán.
- `expand_query`: câu gốc + bản đã giải nghĩa viết tắt + tên chính thức của CLB. **Không** thêm tên S2B (xem bài học ở bước 3).

### Bước 2 – BM25 tự cài (`retrievers.py`)
- Tự cài Okapi BM25 (k1 = 1.5, b = 0.75) khoảng 30 dòng thay vì dùng thư viện, để hiểu từng thành phần. IDF dùng biến thể Lucene `log(1 + (N − df + 0.5)/(df + 0.5))` để luôn dương.
- Token = âm tiết đã bỏ dấu + **bigram âm tiết** (`thanh_vien`), giữ được đơn vị từ mà không cần công cụ tách từ.
- Văn bản của mỗi đoạn = `"{club}. {topic}. {text}"`: nhiều đoạn chỉ viết "CLB" trong thân bài, ghép tên CLB vào thì đoạn nào cũng khớp được với tên chính thức.

| Cấu hình | Train |
|---|---|
| Baseline BTC (bge-m3, câu gốc) | 72.2 |
| BM25, không mở rộng câu hỏi | 74.6 |
| **BM25 + `expand_query`** | **78.7** |

BM25 đơn thuần đã vượt baseline dense. Phần mở rộng câu hỏi của bước 1 đóng góp +4.1 điểm.

### Bước 3 – Dense, RRF, hybrid
- `DenseRetriever`: bge-m3 qua gateway, chuẩn hóa L2 nên tích vô hướng là cosine. Embed `expand_query(query)` thay vì câu gốc: 72.0 → 74.8.
- `rrf`: Reciprocal Rank Fusion, gộp **theo thứ hạng** vì điểm BM25 (0–30) và cosine (0–1) khác thang đo. Đoạn đứng hạng 2 ở cả hai bảng (2/62) thắng đoạn hạng 1 ở một bảng (1/61).

| Cấu hình | Train | Nhóm "không dấu" |
|---|---|---|
| Hybrid, trọng số 1 : 1 | 81.0 | 63.2 |
| **Hybrid, trọng số BM25 : dense = 1.5 : 1** | **82.3** | 65.4 |

**Quan sát quan trọng:** bge-m3 hiểu kém tiếng Việt không dấu. BM25 riêng đạt 78.8 ở nhóm "không dấu", nhưng gộp 1 : 1 với dense kéo xuống 63.2. Vì vậy BM25 được tin hơn.

**Thử nghiệm không áp dụng:**
- `k = 10` trong RRF cho điểm cao hơn khoảng 1 (83.4). Không áp dụng, vì train chỉ có 122 câu, chênh lệch 1 điểm dễ là do chọn tham số khớp riêng với tập này.
- **Tăng điểm cho đoạn thuộc CLB được nhắc tới (club boost)** làm điểm **giảm** từ 80.7 xuống 78.7. Ví dụ "Trong 3 CLB thuộc S2B…": các đoạn chung của S2B được tăng điểm và chiếm hết top-5, đẩy đoạn của từng CLB ra ngoài. Đây cũng là lý do `expand_query` không thêm tên S2B.

### Bước 4 – Prompt và pipeline (`prompts.py`, `pipeline.py`)
- Mỗi quy tắc trong prompt ứng với một metric: giữ nguyên key fact (String Presence), ngắn gọn văn xuôi (ROUGE-L), không bịa (Groundedness), câu từ chối mở đầu cố định "Tài liệu hiện có không có thông tin về…" (Abstention).
- Ngày hôm nay được đưa vào prompt (`date.today()`), để các câu "năm nay", "còn kịp không" tính đúng.
- Hai lỗi phát hiện khi chạy thử 6 câu và đã sửa:
  - Model tự bình luận "hôm nay 02/11 nên đã qua hạn" dù câu hỏi đã nêu sẵn tình huống → chỉ dùng ngày hôm nay khi câu hỏi hỏi về hiện tại.
  - Model tự gợi ý "liên hệ fanpage…" → cấm thêm thông tin ngoài điều được hỏi.
- Trên train: String Presence 84.6, ROUGE-L 61.2.
- **Public Test lần 1: 69.06** (baseline 52.40).

### Bước 5 – Viết lại câu hỏi bằng LLM (`query.py`)
Mục tiêu: sửa hai điểm yếu mà quy tắc ở bước 1 không sửa được.
- **Viết lại** câu hỏi thành tiếng Việt có dấu, giải nghĩa tiếng lóng, gọi CLB bằng tên chính thức.
- **Tách câu hỏi con** khi câu hỏi có nhiều ý hoặc hỏi về nhiều CLB (tối đa 4 câu).
- Prompt nói rõ "Bạn **KHÔNG** trả lời câu hỏi", vì nếu model chèn đáp án bịa vào bản viết lại thì retrieval sẽ bị kéo đi sai hướng.
- `parse_plan` không bao giờ ném lỗi: JSON hỏng ở chỗ nào thì dùng lại câu gốc ở chỗ đó. Lời gọi LLM lỗi cũng quay về câu gốc. Điều này rất quan trọng cho ngày thi Private Test.
- `MultiQueryRetriever` chạy hybrid cho **từng** câu (gốc, viết lại, các câu con) rồi gộp bằng RRF. Câu con "CLB Multimedia PTIT thành lập năm nào?" đưa đúng đoạn của Multimedia lên hạng 1 trong bảng riêng của nó, nên đoạn đó vào được top-5. Câu gốc vẫn được giữ lại để phòng trường hợp viết lại sai.

### Bước 6 – Rerank bằng cross-encoder (`rerank.py`)
- `BAAI/bge-reranker-v2-m3` chạy trên máy: đa ngôn ngữ, open-source, không sinh văn bản (đúng luật), **không tốn ngân sách API**.
- Chỉ rerank 20 ứng viên đầu, vì cross-encoder phải chạy model cho từng cặp (câu hỏi, đoạn văn).
- Chấm bằng **bản viết lại và từng câu con**, không dùng câu gốc không dấu. Mỗi đoạn lấy **điểm cao nhất**, để đoạn chỉ liên quan tới một câu con không bị chìm.
- **Gộp** bảng của retriever với bảng của reranker bằng RRF thay vì thay thế hẳn.

| Trọng số (retriever : reranker) | Train |
|---|---|
| **1 : 1** | **87.8** |
| 1 : 2 | 87.2 |
| 1 : 3 | 86.9 |
| 0 : 1 (chỉ dùng reranker) | 86.5 |
| 2 : 1 | 85.2 |

Chỉ dùng reranker cho kết quả thấp hơn gộp hai bảng: reranker mạnh nhưng vẫn có lúc sai.

### Tổng hợp retrieval trên train

| Cấu hình | SCORE | R@5 | MRR@5 | NDCG@5 | Không dấu | So sánh | Nhiều đoạn |
|---|---|---|---|---|---|---|---|
| Hybrid (bước 3) | 82.3 | 85.7 | 82.2 | 79.1 | 65.4 | 81.4 | 82.7 |
| + viết lại câu hỏi | 86.1 | 90.2 | 85.3 | 82.9 | 75.8 | **87.6** | 81.9 |
| + reranker | 87.8 | 90.8 | 87.7 | 84.9 | 75.2 | 83.2 | 86.2 |
| + cả hai (lần nộp 2–3) | 91.4 | 94.6 | 90.7 | 88.8 | 87.8 | 88.8 | 87.7 |
| + sửa bước gộp (bước 8, lần 4) | 92.1 | 95.2 | 91.4 | 89.6 | 88.3 | 91.1 | 89.8 |
| **+ `n_candidates=30` (bước 9, lần 6)** | **93.2** | **96.0** | **92.6** | **90.9** | **89.5** | **94.0** | **90.4** |
| + A3 doc2query (bước 10, lần 5; đã tắt) | 93.9 | 96.4 | 93.4 | 92.0 | 90.8 | 94.0 | 88.8 |

Các dòng từ bước 8 trở đi được thêm sau; xem từng bước bên dưới. **Hai bước 5 và 6 cộng dồn được**: mỗi bước riêng tăng khoảng 4–5 điểm, bật cả hai tăng 9.1 điểm. Chúng sửa hai loại lỗi khác nhau: viết lại giúp *tìm thấy* đoạn đúng (đặc biệt câu so sánh nhờ câu con), reranker giúp *sắp xếp* các ứng viên chính xác hơn, và reranker chấm trên câu đã có dấu nên hiệu quả hơn hẳn. Nhóm "không dấu" tăng 22 điểm.

- **Public Test lần 2: 80.78** (+11.72). Lần này chỉ đổi retrieval, nên mức tăng đo đúng tác dụng của bước 5 và 6.

### Bước 7 – Tinh chỉnh prompt theo log chấm (prompt v2)
Log lần 2 cho thấy hai metric thấp nhất là **Response Relevancy (57.85)** và **ROUGE-L (57.48)**. Cả hai đều do câu trả lời chưa ngắn gọn, chưa đúng trọng tâm. Ví dụ câu trả lời thật ở lần 2:

> *"Tài liệu hiện có không có thông tin về việc CLB Nhà Sáng Tạo Game PTIT (GCC) có chỗ làm việc riêng… **Tài liệu chỉ nêu GCC tổ chức các buổi training, dự án làm game bằng Scratch… nhưng không đề cập địa điểm thực hiện.**"*

Câu in đậm đúng sự thật nhưng không trả lời câu hỏi. Judge có thể chấm nó là "incorrect" (không có trong tham chiếu), và câu hỏi sinh ngược từ nó sẽ lệch khỏi câu hỏi gốc.

Thay đổi của v2 so với v1 (v1 vẫn được giữ trong code để so sánh và quay lại):

| Quy tắc | v1 | v2 |
|---|---|---|
| Câu mở đầu | "Trả lời thẳng vào câu hỏi" | Dùng lại chủ ngữ và từ khóa của câu hỏi; câu có/không mở bằng "Có."/"Không." |
| Độ dài | 1–3 câu | 1–2 câu, 25–60 từ; mỗi câu phải trả lời một phần câu hỏi |
| Chi tiết then chốt | Giữ ngày, số, tên, link | Chép nguyên văn cả tên ban, tên sự kiện, cụm từ trong danh sách; liệt kê đủ mục |
| Thiếu thông tin | Nói rõ phần nào thiếu | Một câu từ chối; thông tin một phần thì thêm một vế ngắn, không kể "tài liệu chỉ nêu" |
| Few-shot | Không có | 4 ví dụ từ train (trực tiếp, không dấu, so sánh, từ chối) |

Các câu train dùng làm few-shot (`FEW_SHOT_IDS`) được loại khỏi mẫu đánh giá, nếu không điểm sẽ bị thổi phồng.

**Kết quả trên mẫu 30 câu train** (phân tầng, đã loại câu few-shot; `scripts/eval_generation.py --sample 30`):

| Prompt | String Presence | ROUGE-L | Độ dài / đáp án mẫu | `missing_fact` | `false_abstain` |
|---|---|---|---|---|---|
| v1 | 92.9 | 62.4 | 1.19 | 2 | 0 |
| v2 | 89.9 | 62.4 | **1.03** | 4 | 2 |

- v2 ngắn hơn đúng như mong muốn (độ dài gần bằng đáp án mẫu), ROUGE-L giữ nguyên.
- Cái giá: String Presence giảm 3 điểm. Khi rút gọn, model bỏ các chi tiết phụ mà đáp án mẫu lại tính là key fact (ví dụ `train-0015` bỏ mất "tiền bối", `train-0074` bỏ mất "CDIT").
- Hai câu `false_abstain` mới đều là câu mà cả v1 và v2 đều trả lời sai vì retrieval trượt; v2 chỉ diễn đạt phần thiếu bằng câu "Tài liệu hiện có không có thông tin…" nên bị nhận diện là từ chối.
- Mẫu chỉ có 30 câu: 3 điểm String Presence tương đương khoảng 1 key fact trên 1 câu, chưa đủ để kết luận. Relevancy và Noise (mục tiêu chính của v2) không đo được trên máy nếu không chạy judge, nên phải đo bằng Public Test lần 3.

**Public Test lần 3 (chỉ đổi prompt, retrieved_ids giống hệt lần 2): 81.44** (+0.66).

| Metric | Lần 2 (v1) | Lần 3 (v2) | Thay đổi |
|---|---|---|---|
| String Presence | 84.40 | 86.32 | +1.92 |
| ROUGE-L | 57.48 | 58.13 | +0.65 |
| Groundedness | 95.73 | 97.01 | +1.28 |
| Response Relevancy | 57.85 | 59.88 | +2.03 |
| 1 − Noise Sensitivity | 86.32 | 88.46 | +2.14 |
| **Generation** | 80.30 | **81.63** | +1.33 |

- **Cả 5 metric generation đều tăng**, đúng những metric v2 nhắm tới (Relevancy, Noise). Mức tăng tổng (+0.66) nằm trong biên dao động ±1 điểm của judge, nhưng việc mọi metric cùng tăng cho thấy cải thiện là thật.
- **String Presence trên Public Test lại tăng (+1.9)**, ngược với mẫu train (−3.0). Bài học: mẫu 30 câu quá nhỏ, chênh lệch vài điểm trên đó không đáng tin; nó chỉ hữu ích để bắt các thay đổi lớn hoặc lỗi rõ ràng.

---

### Bước 8 – Sửa lỗi gộp làm loãng tín hiệu của câu hỏi con

Phát hiện khi phân tích các câu Public Test bị từ chối sau lần nộp 3 (chi tiết ở mục 6.5): câu hỏi con tìm đúng đoạn (hạng 1–3 trong bảng riêng), nhưng gộp 4–6 bảng bằng RRF làm đoạn đúng tụt xuống hạng 16–20 hoặc ra ngoài 20 ứng viên, rồi bước gộp cuối với retriever kéo nó xuống lần nữa.

- `MultiQueryRetriever(reserve=2)`: top-2 của mỗi câu con luôn có mặt trong danh sách ứng viên.
- `RerankedRetriever(multi_weights=(0, 1))`: với câu có câu con, xếp hạng hoàn toàn theo reranker (điểm reranker lấy max theo từng câu con nên không bị loãng).

| | Train | Public Test |
|---|---|---|
| Trước (lần 3) | 91.4 | 81.26 |
| **Sau (lần 4)** | **92.1** (+0.7) | **87.83** (+6.57) |

Raw score Public Test: 81.44 → **85.20**.

### Bước 9 – Tinh chỉnh lại tham số (A1) và phủ ý trong top-5 (A2)

Sau bước 5–8, nhiều tham số được chọn từ trước không còn chắc là tốt nhất: tỉ lệ BM25 : dense = 1.5 : 1 được chọn khi dense còn nhận câu không dấu; `n_candidates = 20` được chọn khi reranker còn chạy CPU (chậm). Hai ý tưởng được đo cùng lúc trên train, không tốn API:

- **A1:** tỉ lệ BM25 : dense ∈ {1.5 : 1, 1 : 1, 1 : 1.5} × `n_candidates` ∈ {20, 30, 40}.
- **A2 (`cover`):** với câu có câu con, bảo đảm đoạn tốt nhất của **mỗi** câu con (theo điểm reranker của riêng câu con đó) nằm trong top-5, vì điểm max có thể bị một câu con "dễ" chiếm hết top-5.

| BM25 : dense | `n_candidates` | `cover` | SCORE | R@5 | Câu có câu con | Không dấu | Nhiều đoạn |
|---|---|---|---|---|---|---|---|
| 1.5 : 1 | 20 | 0 (trước đây) | 92.1 | 95.2 | 91.4 | 88.3 | 89.8 |
| 1.5 : 1 | 20 | 1 | 92.3 | 95.6 | 91.9 | 88.3 | 91.0 |
| **1.5 : 1** | **30** | **0** | **93.2** | 96.0 | 93.0 | 89.5 | 90.4 |
| 1.5 : 1 | 30 | 1 | 93.4 | 96.4 | 93.5 | 89.5 | 91.6 |
| 1.5 : 1 | 40 | 0 / 1 | 93.2 | 96.0 | 93.0 | 89.5 | 90.4 |
| 1 : 1 | 20 / 30 / 40 | 0 | 92.7 / 93.1 / 93.2 | | | 89.7–91.2 | 87.9–88.3 |
| 1 : 1 | 20 / 30 / 40 | 1 | 92.9 / 93.0 / 92.9 | | | | |
| 1 : 1.5 | 20–40 | 0 / 1 | 89.6–90.2 | | | **76.8** | |

- **`n_candidates` 20 → 30: +1.1**, nhất quán ở cả hai tỉ lệ 1.5 : 1 và 1 : 1; lên 40 không tăng thêm. Đổi lại reranker chấm thêm khoảng 50% số cặp (trên GPU vẫn khoảng 1–2 giây mỗi câu). **Áp dụng.**
- **Tỉ lệ BM25 : dense:** 1 : 1 ngang 1.5 : 1; 1 : 1.5 làm nhóm không dấu tụt xuống 76.8 (câu gốc không dấu vẫn nằm trong danh sách truy vấn và dense hiểu kém nó). **Giữ 1.5 : 1.**
- **A2 (`cover`): +0.2 với 1.5 : 1 nhưng −0.1 đến −0.4 với 1 : 1** – nhỏ và không nhất quán, nên theo bài học số 6 **không áp dụng** (mặc định `cover=0`, giữ code để thử lại khi retrieval thay đổi).
- Retrieval trên train: **92.1 → 93.2** (R@5 96.0, MRR@5 92.6, NDCG@5 90.9).
- Trên Public Test: 16/42 câu có tập top-5 khác lần 4, 29/42 câu có thứ tự 8 đoạn ngữ cảnh khác.
- **Public Test lần 6 (chỉ A1): 85.55**, retrieval 87.83 → 87.94 (+0.11, Recall@5 +0.48). Mức tăng nhỏ hơn nhiều so với train (+1.1) nhưng không gây hại, và sửa được `public-0033` (slogan Multimedia). Đây là bản tốt nhất (mục 6.5d).

### Bước 10 – Làm giàu tài liệu lúc index (A3, doc2query) – thử nghiệm, đã tắt

**Vấn đề nhắm tới:** câu hỏi và đoạn văn là hai loại văn bản rất khác nhau (văn nói không dấu so với văn viết trang trọng), và có thông tin nằm lẫn trong đoạn có chủ đề khác (`public-0040`: "dành cho sinh viên mọi ngành" nằm trong đoạn *Lịch sử & thành lập*).

**Cách làm** (`src/my_solution/enrich.py`): lúc index, `claude-haiku-5-5` đọc từng đoạn và viết 6 câu hỏi mà đoạn trả lời được (2 câu chuẩn, 2 câu văn nói, 2 câu không dấu kiểu tin nhắn), phủ cả chi tiết phụ, cùng tối đa 10 từ khóa (kể cả từ đồng nghĩa không có trong đoạn).
- Kết quả: 118/118 đoạn, 754 câu hỏi, 1.182 từ khóa; khoảng 118 lời gọi LLM và 23 request embedding, ước tính khoảng $0.05, chỉ một lần.
- **Lưu riêng** ở `artifacts/doc2query.jsonl`, không trộn vào `data/corpus.jsonl`. Mỗi dòng gắn mã hash nội dung của đoạn: đoạn đổi thì sinh lại, đoạn không đổi thì đọc từ file (chạy lại cho kết quả giống hệt, không gọi API).
- Phần làm giàu **chỉ dùng để tìm kiếm**; reranker và prompt sinh câu trả lời vẫn dùng văn bản gốc (Groundedness chấm trên văn bản gốc).
- Hợp lệ theo luật: là một kỹ thuật đánh chỉ mục dùng đúng `claude-haiku-5-5`, không thêm thông tin ngoài corpus, không dùng câu hỏi test, không sửa tay kết quả.

**Đo trên train** (lượt đo chạy với client mạng bị gỡ):

| Cấu hình | SCORE | Câu có câu con | Câu không có | Không dấu | Diễn đạt khác | So sánh | Nhiều đoạn |
|---|---|---|---|---|---|---|---|
| Tắt | 93.2 | 93.0 | 93.3 | 89.5 | 93.8 | 94.0 | 90.4 |
| Nối vào văn bản BM25 của đoạn | 93.9 | **91.9** | 95.2 | 89.8 | 97.8 | **91.4** | 89.1 |
| Chỉ cho dense (vector riêng cho từng câu hỏi, lấy max) | 93.2 | 93.0 | 93.3 | 89.4 | 94.5 | 94.0 | 90.4 |
| Nối BM25 + dense | 93.3 | 91.9 | 94.1 | 88.3 | 97.8 | 91.4 | 88.5 |
| Bảng BM25 riêng, trọng số 0.5 | 93.7 | 93.0 | 94.2 | 90.6 | 96.2 | 94.0 | 89.1 |
| **Bảng BM25 riêng, trọng số 1.0** | **93.9** | **93.0** | 94.6 | **90.8** | **97.6** | **94.0** | 88.8 |
| Bảng BM25 riêng, trọng số 1.5 | 93.8 | 93.0 | 94.4 | 90.9 | 97.5 | 94.0 | 88.1 |

- **Nối thẳng vào văn bản của đoạn làm hại câu có câu con** (−1.1, so sánh −2.6). Giả thuyết: mỗi câu hỏi sinh ra đều nhắc tên CLB, nên mọi đoạn cùng một CLB khớp mạnh với tên CLB và BM25 khó phân biệt các đoạn trong cùng CLB.
- **Kiểm chứng giả thuyết:** đánh chỉ mục phần làm giàu thành **một bảng BM25 riêng** (bảng thứ ba trong RRF), văn bản của đoạn giữ nguyên. Mức tăng +0.7 giữ nguyên, nhóm câu có câu con **không còn bị giảm**, và kết quả ổn định ở cả ba trọng số 0.5 / 1.0 / 1.5.
- **Cấu hình chọn trên train:** bảng BM25 riêng, trọng số 1.0, tắt phần cho dense (`HybridRetriever.with_doc2query()`). Retrieval trên train **93.2 → 93.9** (R@5 96.4, MRR@5 93.4, NDCG@5 92.0).
- **Giới hạn:** với Public Test, 21/42 câu có tập top-5 khác lần 4, nhưng câu `public-0040` (động lực ban đầu của A3) **vẫn chưa được sửa**: câu này có câu con nên được xếp hoàn toàn theo reranker (bước 8), mà reranker chỉ đọc văn bản gốc của đoạn, nơi chi tiết "mọi ngành" vẫn bị chìm. A3 giúp đoạn đó vào danh sách ứng viên, nhưng không quyết định thứ hạng cuối.
- **Kết quả trên Public Test (lần nộp 5 và 6, mục 6.5c–6.5d): A3 làm retrieval giảm 0.97** (Recall@5 −2.19). Tăng trên train nhưng giảm trên test, nên **mặc định tắt**. Code vẫn giữ để bật lại bằng `MyPipeline(doc2query=True)`.

### Bước 11 – Nghiên cứu các hướng tối ưu tiếp theo (A4–A7, B1–B2, Step-Back) – chưa thực hiện

Sau lần nộp 6 (85.55) mục này ghi lại các hướng đã cân nhắc, kèm phân tích, **trước khi thử bất kỳ hướng nào**. Các con số dưới đây đo offline trên train bằng dữ liệu đã cache, không tốn tiền API.

**Điểm đang mất ở đâu** (lần nộp 6):

| Thành phần | Hiện tại | Tăng 1 điểm metric thì điểm tổng tăng | Dư địa |
|---|---|---|---|
| **Retrieval** | 87.94 | **+0.50** | Lớn nhất |
| ROUGE-L | 57.76 | +0.08 | Vừa |
| Response Relevancy | 58.33 | +0.08 | Chưa rõ, có thể đã gần trần |
| String Presence | 88.03 | +0.08 | Tăng theo retrieval |
| Groundedness, 1 − Noise, Abstention | 98.72 / 96.15 / 100 | +0.08 | Gần như hết |

Retrieval vẫn là đòn bẩy chính: 1 điểm retrieval có giá trị gấp khoảng 6 lần 1 điểm của một metric generation.

**Phát hiện: nút thắt nằm ở khâu xếp hạng, không phải khâu tìm ứng viên.** Trên 196 đoạn đúng của train (cấu hình hiện tại):
- **96.9%** (190/196) đã nằm trong 30 ứng viên; chỉ 6 đoạn bị trượt từ đầu (`train-0028`, `0044`, `0059`, `0099`, `0113` ×2).
- Nhưng chỉ **92.9%** (182/196) vào top-5 sau reranker, và 95.4% vào top-10.
- Các câu tệ nhất cho thấy cross-encoder khớp bề mặt câu chữ, không hiểu *ý định*: `train-0057` ("clb multimedia co phong rieng de chup anh quay phim ko") đoạn đúng *Quyền lợi* bị xếp thứ 5, sau *Đào tạo* và *Giới thiệu*; `train-0073` (người giỏi nói chuyện, thích kết nối đối tác nên vào ban nào) đoạn *Cơ cấu* bị xếp thứ 4. Đây là câu hỏi nói về *tình huống của người hỏi* trong khi tài liệu viết theo *chủ đề*, và dạng này chiếm nhiều ở Public Test (`public-0003`, `0006`, `0008`, `0024`, `0040`).

#### Nhóm A – retrieval

**A4. LLM xếp hạng lại cả danh sách (listwise rerank, kiểu RankGPT)** – *hướng ưu tiên số 1*
- Cách làm: đưa 20–30 ứng viên (id, tên CLB, chủ đề, văn bản) cho `claude-haiku-5-5`, yêu cầu trả về danh sách id theo thứ tự liên quan; gộp với bảng của cross-encoder bằng RRF.
- Ưu điểm:
  - Sửa đúng loại lỗi vừa thấy: LLM suy luận được đoạn *Quyền lợi* có nói về phòng chụp ảnh.
  - Không có tham số nào chỉnh theo train, nên ít rơi vào tình huống "tăng trên train, giảm trên test" như A3 (bài học 14).
  - Chi phí thấp: khoảng $0.0006 mỗi câu, train + Public Test khoảng **$0.10** (~164 lời gọi).
- Nhược điểm:
  - Thêm 1 lời gọi LLM mỗi câu (độ trễ, và rate limit 100 request/phút khi chạy ngày thi).
  - Không tất định (Haiku không đặt được temperature); cache chỉ làm kết quả lần đầu thành cố định.
  - Không sửa được 3.1% đoạn đúng nằm ngoài 30 ứng viên.
- Giảm rủi ro: gộp RRF với cross-encoder thay vì thay thế; lời gọi lỗi hoặc JSON hỏng thì quay về xếp hạng hiện tại.

**A5. Cho LLM đọc toàn bộ corpus để chọn đoạn**
- Corpus chỉ 118 đoạn, ~45 nghìn ký tự (ước tính 20–25 nghìn token) nên vừa một prompt.
- Ưu điểm: sửa được cả 3.1% đoạn đúng đang nằm ngoài ứng viên.
- Nhược điểm: đắt hơn A4 khoảng 4 lần (~$0.0025 mỗi câu, train + Public Test ~**$0.40**); danh sách dài nên LLM dễ thiên về đoạn đứng đầu hoặc cuối prompt.
- Nên dùng để bổ sung hoặc so sánh với A4, không chạy cả hai cùng lúc ngay từ đầu.

**A6. Ghép thêm một cross-encoder thứ hai chạy trên máy**
- Ví dụ `jina-reranker-v2-base-multilingual` hoặc `gte-multilingual-reranker-base`, ghép với bge-reranker-v2-m3 bằng RRF.
- Ưu điểm: không tốn tiền API, đúng luật (model không sinh văn bản).
- Nhược điểm: mức tăng thường nhỏ hơn A4 vì hai model cùng mắc kiểu lỗi khớp bề mặt; thêm RAM/thời gian trên GPU.
- Các reranker dựng trên LLM (Qwen3-Reranker, bge-reranker-v2-gemma) mạnh hơn nhưng thuộc vùng xám của luật "model không sinh văn bản": phải hỏi BTC trước khi dùng.

**A7. Dùng chính câu trả lời để sắp lại top-5 (không tốn thêm lời gọi)**
- Cách làm: model sinh câu trả lời kèm id các đoạn đã dùng, đưa các đoạn đó lên đầu `retrieved_ids`.
- Ưu điểm: có thể tăng MRR và NDCG đáng kể mà không thêm lời gọi.
- Nhược điểm: phải đổi định dạng đầu ra của prompt sinh câu trả lời, có thể ảnh hưởng ROUGE-L và độ lan man; phần lớn bị trùng nếu đã làm A4.

**Step-Back prompting** (sinh câu hỏi khái quát hơn rồi tra cứu bằng câu đó)
- Cách làm: thêm trường `"stepback"` vào JSON của bước viết lại hiện có (không tốn thêm lời gọi), dùng như một truy vấn bổ sung. Ví dụ `train-0057` → "CLB Multimedia PTIT mang lại những quyền lợi, cơ sở vật chất gì cho thành viên?", khớp thẳng đoạn *Quyền lợi*.
- Ưu điểm: rẻ (đổi prompt viết lại nên phải gọi lại cache, khoảng 164 lời gọi nhỏ, dưới $0.03); bắc cầu đúng khoảng cách "tình huống cụ thể ↔ tài liệu theo chủ đề".
- Nhược điểm: câu khái quát rất chung chung, có thể kéo đoạn *Giới thiệu chung* lên top-5 và làm loãng xếp hạng, đúng cơ chế đã làm A3 hại. Cần đo hai biến thể: chỉ dùng để tìm ứng viên (không tham gia điểm reranker), hoặc tham gia với trọng số thấp.
- Chỉ áp dụng ở bước retrieval; không áp dụng ở bước sinh câu trả lời vì Noise và Groundedness chấm từng câu, phần suy luận trung gian lọt vào câu trả lời sẽ bị phạt.
- Bổ sung cho A4 (sửa ở phía câu hỏi, còn A4 sửa ở phía xếp hạng); nên nộp ở hai lần riêng.

#### Nhóm B – generation (dư địa nhỏ hơn)

**B1. Chọn ví dụ few-shot theo câu hỏi**
- Cách làm: lấy 3 câu train giống câu hỏi nhất làm ví dụ, thay 4 ví dụ cố định.
- Ưu điểm: đáp án mẫu có văn phong khá đều (27/126 mở đầu bằng "Theo …", 20 bằng "Có."; độ dài trung vị 41 token), trong khi câu trả lời Public Test của ta dài trung vị 45.5, trung bình 50.6 token, nên ROUGE-L có thể tăng vài điểm.
- Nhược điểm: model có thể chép nội dung ví dụ vào câu trả lời và bị Noise phạt; phải loại câu few-shot khỏi tập đánh giá trên train (như `FEW_SHOT_IDS`).

**B2. Ước tính trần của Response Relevancy trước khi tối ưu**
- Metric so cosine giữa câu hỏi gốc (thường không dấu) và các câu hỏi sinh ngược có dấu, nên 58 có thể đã gần trần.
- Kiểm tra bằng khoảng 84 request embedding cho Public Test (< $0.01). Nếu gần trần thì bỏ hẳn hướng này.

#### Không khuyến nghị
- **Fine-tune reranker trên 126 câu train:** quá ít dữ liệu, và điểm train đã không còn dự báo được điểm test (bài học 14).
- **HyDE, contextual retrieval (LLM sinh thêm mô tả cho từng đoạn):** cùng họ với A3, vốn làm giảm điểm trên Public Test.

**Kết quả thực hiện Step-Back (sau lần nộp 7): không giúp, đã tắt.** Thiết kế (đã duyệt): lời gọi LLM riêng (`stepback.py`, không nhét vào JSON viết lại để khỏi sinh lại bản viết lại và câu con); chỉ cho câu **không có câu con**; câu khái quát chỉ thêm một bảng xếp hạng vào RRF tìm ứng viên của `MultiQueryRetriever`, không dùng để chấm cross-encoder và không đưa vào prompt của A4; `MyPipeline(stepback=False)` mặc định tắt. Điều kiện qua cửa đặt trước: retrieval train tăng ít nhất 0.5 và không nhóm nào giảm quá 1 điểm; không đạt thì không chạy A4 + Step-Back và không chạy Public Test. Đo giai đoạn 1 (không có A4, chỉ tốn khoảng 78 lời gọi Step-Back, khoảng $0.01):

| Cấu hình | SCORE | R@5 | MRR@5 | NDCG@5 | Không dấu | Nhiều đoạn | Mâu thuẫn nguồn (n=2) |
|---|---|---|---|---|---|---|---|
| Không Step-Back (bước 9) | **93.2** | 96.0 | 92.6 | 90.9 | 89.5 | 90.4 | 97.5 |
| Step-Back, trọng số 1.0 | 92.6 | 96.0 | 91.7 | 90.0 | 88.0 | 91.8 | 85.2 |
| Step-Back, trọng số 0.5 | 92.9 | 96.0 | 92.2 | 90.5 | 87.7 | 92.0 | 97.5 |

- **Không đạt cửa:** cả hai trọng số đều thấp hơn không dùng (−0.6 và −0.3). Recall@5 và Recall@7 không đổi (96.0 và 97.1), tức Step-Back không đưa thêm đoạn đúng vào top-5; phần giảm nằm ở MRR (câu khái quát kéo các đoạn chung chung lên trên, đúng cơ chế A3 đã gặp). Nhóm không dấu giảm 1.5–1.8, nhóm mâu thuẫn nguồn (chỉ 2 câu) giảm mạnh ở trọng số 1.0.
- Vì sao khác kỳ vọng: lý do ban đầu của Step-Back là cross-encoder xếp sai thứ tự, và A4 đã sửa phần lớn lỗi này. Phần còn lại Step-Back nhắm tới là đưa đoạn vào danh sách ứng viên, nhưng trên train 190/196 đoạn đúng đã nằm trong 30 ứng viên, nên gần như không còn gì để cứu (chưa đo riêng số đoạn được cứu).
- **Quyết định:** dừng ở giai đoạn 1; không chạy A4 + Step-Back, không nộp. Giữ code (`stepback.py`, `MyPipeline(stepback=True)`, `--retriever full-sb`, `full-sb05`, `full-a4-sb`, 12 test mới) như một thí nghiệm có ghi chép, mặc định tắt. Không dùng cách tinh chỉnh trọng số để "cứu" ý tưởng: sai lệch 0.3–0.6 trên 122 câu quá nhỏ để chọn tham số (bài học 6).

**Kết quả thực hiện sửa quy tắc 8 của prompt (prompt v3): không qua cửa, đã bỏ.** Cơ sở: lần 7 có 7/42 câu trả lời chứa vế "Tài liệu không có thông tin về …" (khoảng 17%), trong khi đáp án mẫu train chỉ có 5/115 câu (khoảng 4%) và đó là điều câu hỏi hỏi trực tiếp (loại `thiếu dữ liệu`: `train-0053`, `0095`, `0119`), nên không thể xoá hẳn các vế này. `v3` chỉ khác `v2` đúng quy tắc 8: chỉ nêu phần thiếu khi câu hỏi **hỏi trực tiếp** phần đó, không nhắc chi tiết người hỏi chỉ kể làm bối cảnh. Đo trên mẫu 30 câu train (loại câu few-shot, A4 bật, không qua judge, khoảng $0.03):

| Prompt | String Presence | ROUGE-L | Dài / mẫu | `missing_fact` | `false_abstain` | Câu có đáp án chứa vế "không có thông tin" (mixed / pure, 28 câu) |
|---|---|---|---|---|---|---|
| v2 | **90.2** | **66.4** | 0.98 | 1 | 1 | 2 / 1 |
| v3 | 86.3 | 62.6 | 0.97 | 1 | 1 | 1 / 1 |

- **Không đạt cửa** (String Presence không được giảm quá 1 điểm): −3.9, ROUGE-L −3.8. Số câu mixed chỉ 2 → 1, quá ít để nói v3 giảm được vế từ chối.
- **Cơ chế gây hại quan sát được:** `train-0062` (năm 2027 ProPTIT mở đơn khi nào): v2 vừa từ chối vừa nêu "các mốc chỉ áp dụng cho đợt tuyển năm 2026 … công bố trên fanpage" (đúng key fact); v3 rút gọn thành **một câu từ chối trơn**, mất cả hai key fact, nên nhóm "thời gian" tụt 100 → 66.7. Khi bị cấm nhắc phần thiếu, model bỏ luôn cả thông tin hữu ích đi kèm.
- Mẫu nhỏ (28 câu, chênh khoảng 1 key fact) và Haiku không tất định, nên không kết luận chắc về mức độ; nhưng cơ chế rõ và ngược hướng với mục tiêu, nên không tinh chỉnh lại lời văn dựa trên 30 câu này (bài học 6) và không nộp lần 8.
- **Quyết định:** giữ `v2` làm mặc định. Giữ `v3` trong `prompts.py` (chọn bằng `MyPipeline(prompt_version="v3")`, có test) như một thí nghiệm có ghi chép. Đã sửa `scripts/eval_generation.py` để ước tính chi phí đúng khi A4 bật.

**Cập nhật sau lần nộp 7:** A4 đã thực hiện, kết quả ở mục 6.5e (retrieval 87.94 → 92.01, raw 85.55 → 87.22).

#### Thứ tự đề xuất
1. **A4** (~$0.10): đo trên train, đếm số câu Public Test đổi top-5, nộp riêng một lần (lần 7).
2. Nếu A4 tăng: thử A5 để so sánh, hoặc A6 vì không tốn tiền; Step-Back ở một lần nộp riêng.
3. B1 ở một lần nộp riêng; B2 khi còn thời gian.

Giữ nguyên tắc **mỗi lần nộp chỉ đổi một thứ**.

---

### Bước 12 – Thay embedding bge-m3 bằng AITeamVN/Vietnamese_Embedding_v2 (v10) – không qua cửa, giữ bge-m3

**Ý tưởng:** `AITeamVN/Vietnamese_Embedding_v2` là bản bge-m3 tinh chỉnh cho tiếng Việt (khoảng 0.6B tham số, 1024 chiều, Apache 2.0, chạy trên máy bằng sentence-transformers, không cần tách từ hay tiền tố). Quy định (mục 1) cho phép embedding open-source chạy trên máy. Kỳ vọng: hiểu tiếng Việt tốt hơn bge-m3 gốc; không tốn request gateway.

**Cài đặt:** `DenseRetriever(local_model=...)` dùng `LocalEmbedder` có sẵn của BTC (mặc định tắt, không đổi hành vi cũ); các retriever `dense-vn`, `hybrid-vn`, `full-vn`, `full-vn-a4` trong `scripts/eval_retrieval.py`. Model tải về khoảng 2.3 GB.

**Cửa đặt trước (không chỉnh trọng số hybrid 1.5 : 1 hay tham số nào khác theo train):** `full-vn` phải tăng ít nhất +1.0 so với `full` trên train (122 câu) và nhóm không dấu không giảm quá 1.0; đo trên máy, chặn mạng, chi phí $0.

**Kết quả (train, 122 câu):**

| Cấu hình | bge-m3 (gateway) | Vietnamese_Embedding_v2 | Thay đổi |
|---|---|---|---|
| Dense riêng | 74.8 | 68.8 | −6.0 |
| Hybrid (BM25 + dense) | 82.3 | 80.5 | −1.8 |
| Full (viết lại + reranker) | 93.2 | 92.3 | **−0.9** |

Nhóm không dấu: dense 48.0 → 41.6, full 89.5 → 88.8. Nhóm nhiều đoạn full 90.4 → 88.8.

**Kết luận:** không qua cửa (−0.9, cần +1.0), **giữ bge-m3**, không sinh file nộp lần 10 nên không tốn lượt nộp hay API. Mô hình "tốt cho tiếng Việt" nói chung không bằng bge-m3 trên bộ câu hỏi này: câu hỏi hay không dấu/viết tắt và đoạn văn ngắn, nên khả năng cao bản tinh chỉnh học trên văn phong khác (chưa kiểm chứng nguyên nhân). Sau cross-encoder và A4/A5, khác biệt ở bước tìm ứng viên còn bị thu nhỏ, nên không có lý do để thử tiếp. Rủi ro bổ sung nếu dùng: thêm 2.3 GB phải tải trên máy khác, GPU 6 GB phải chứa cả reranker và embedding.

---

## 4. Các lần nộp Public Test

| Metric | Baseline BTC | Lần 1 (bước 4) | Lần 2 (bước 5 + 6) | Lần 3 (prompt v2) | Lần 4 (sửa bước gộp) | Lần 5 (A1 + A3) | Lần 6 (chỉ A1) | Lần 7 (A4) | Lần 8 (A4 + A5) | Lần 9 (A4 + A5, cache trống) |
|---|---|---|---|---|---|---|---|---|---|---|
| Recall@5 | 43.45 | 65.48 | 81.71 | 81.71 | 88.65 | 86.94 | **89.13** | 90.71 | **93.29** | **95.67** |
| MRR@5 | 44.21 | 65.67 | 84.76 | 84.76 | 89.76 | 89.37 | 89.37 | **95.24** | **95.83** | **97.02** |
| NDCG@5 | 39.29 | 60.93 | 77.30 | 77.30 | 85.06 | 84.60 | **85.32** | **90.07** | **92.39** | **94.74** |
| **Retrieval** | 42.31 | 64.03 | 81.26 | 81.26 | 87.83 | 86.97 | **87.94** | **92.01** | **93.84** | **95.81** |
| String Presence | 52.14 | 71.79 | 84.40 | 86.32 | 87.61 | 87.18 | **88.03** | **88.46** | **91.88** | 90.60 |
| ROUGE-L | 33.11 | 51.34 | 57.48 | 58.13 | 58.84 | 57.61 | 57.76 | **59.67** | **60.92** | 59.63 |
| Groundedness | 83.29 | 93.59 | 95.73 | 97.01 | 97.86 | 97.86 | **98.72** | 96.15 | 96.79 | **100.00** |
| Response Relevancy | 60.07 | 59.03 | 57.85 | 59.88 | 59.73 | 59.80 | 58.33 | **59.71** | **59.98** | **60.21** |
| 1 − Noise Sensitivity | 46.26 | 68.80 | 86.32 | 88.46 | 91.45 | 94.02 | **96.15** | 90.60 | 93.16 | **96.37** |
| Abstention | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | **66.67** | 100 |
| **Generation** | 62.48 | 74.09 | 80.30 | 81.63 | 82.58 | 82.74 | **83.17** | 82.43 | 78.23 | **84.47** |
| **Raw score** | 52.40 | 69.06 | 80.78 | 81.44 | 85.20 | 84.86 | **85.55** | **87.22** | 86.04 | **90.14** |
| Thời gian mỗi câu (median) | – | 2.6 giây | 27.3 giây (CPU) | 27 giây khi chưa có cache | 2.7 giây (GPU) | 3.2 giây (GPU) | 1.5 giây (GPU, có cache một phần) | 3.2 giây (GPU, cache trống) |

Nguyên tắc: **mỗi lần nộp chỉ đổi một thứ**, để biết chính xác thay đổi nào có tác dụng.

Lần 8 (A4 + A5) có retrieval cao nhất nhưng raw **thấp hơn** lần 7 vì Abstention 100 → 66.67 (một trong 3 câu không có đáp án trả lời thay vì từ chối): mục 6.5f.

Lần 9 là **cùng cấu hình với lần 8** (A4 + A5, thêm timeout), chạy lại với cache trống: **90.14**, bản tốt nhất. Lần 8 và 9 chỉ khác nhau ở lần sinh lại (A4, A5, câu trả lời đều do LLM không tất định): mục 6.5g.

Lần 7 nộp hai lần với cùng một file: **87.22** (ID 33139) và **87.34** (ID 33141) → judge dao động cỡ 0.1–0.3 raw (mục 6.5e).

**Quan sát từ lần 1 sang lần 2:**
- Retrieval trên Public Test (64.03) thấp hơn train (82.3) 18 điểm, đúng như dự đoán vì Public Test có nhiều câu không dấu hơn.
- Retrieval tốt hơn kéo theo cả generation: String Presence +12.6, Noise +17.5. Phần lớn câu trả lời "sai" ở lần 1 thật ra là do model không được thấy đoạn đúng. Số câu trả lời "không có thông tin" giảm từ 18 xuống 12.
- Response Relevancy không tăng mà còn giảm nhẹ. Đây là metric phụ thuộc vào văn phong câu trả lời, không phụ thuộc retrieval.

**Quan sát từ lần 3 sang lần 4** (chỉ đổi cách gộp, prompt giữ nguyên v2):
- Retrieval +6.57 (81.26 → 87.83), trong khi trên train cùng thay đổi chỉ +0.7. Lỗi "gộp làm loãng tín hiệu của câu hỏi con" xảy ra ở Public Test nhiều hơn hẳn ở train, đúng như chẩn đoán ở mục 6.5.
- Khoảng cách retrieval giữa Public Test và train thu hẹp từ 10.1 điểm (81.26 so với 91.4) còn 4.3 điểm (87.83 so với 92.1).
- Retrieval tốt hơn tiếp tục kéo theo generation: Noise +3.0, String Presence +1.3. Response Relevancy gần như đứng yên (59.88 → 59.73), lại xác nhận metric này không phụ thuộc retrieval.

**Quan sát từ lần 4 sang lần 5 và 6** (chi tiết ở mục 6.5c–6.5d):
- Lần 5 đổi hai thứ (A1 + A3) và **giảm** (85.20 → 84.86, retrieval −0.86), dù trên train tăng 1.8. Không biết lỗi ở thay đổi nào nên phải tốn thêm lần 6 để tách.
- Lần 6 chỉ giữ A1: **85.55**, bản tốt nhất. Suy ra A3 làm retrieval Public Test giảm 0.97 (Recall@5 −2.19), còn A1 trung tính đến hơi tốt (+0.11).
- Khoảng cách retrieval giữa train và Public Test: 93.2 so với 87.94, còn 5.3 điểm. Hai thay đổi gần nhất đều tăng trên train nhiều hơn hẳn trên Public Test: tăng trên train đang dần **không còn dự báo được** tăng trên test.
- Noise tăng đều qua ba lần (91.45 → 94.02 → 96.15) và Groundedness lên 98.72; Response Relevancy giảm còn 58.33. Các thay đổi này nằm trong biên ±1 đến ±2 của judge khi câu trả lời được sinh lại.

---

## 5. Phân tích lỗi câu trả lời

`scripts/analyze_predictions.py` quy mỗi key fact bị thiếu về **đúng một** nguyên nhân. Kết quả trên bản bước 4 (126 câu train):

| Lỗi | Số câu | Ý nghĩa |
|---|---|---|
| `retrieval_miss` | 17 | Đoạn chứa đáp án không có trong 8 đoạn đưa vào prompt |
| `missing_fact` | 15 | Key fact có sẵn trong prompt nhưng model không viết ra |
| `false_abstain` | 13 | Câu có đáp án nhưng trả lời "không có thông tin" (phần lớn là hệ quả của `retrieval_miss`) |
| `fact_not_literal` | 4 | Đáp án mẫu tự suy ra hoặc đổi định dạng |
| `missed_abstain` | 0 | Không có câu nào lẽ ra phải từ chối mà không từ chối |

**Bài học khi viết công cụ phân tích:** bản đầu đếm một fact vào cả hai loại lỗi, khiến `fact_not_literal` bị thổi lên 17 (con số đúng là 4), chỉ sai hướng sửa. Mỗi lỗi phải có đúng một nguyên nhân.

---

## 6. Sổ nhật ký kiểm tra lỗi sau các lần nộp

Mục này ghi lại, theo thứ tự thời gian, những gì đã được kiểm tra trước và sau mỗi lần nộp, lỗi nào được phát hiện, nguyên nhân gốc và quyết định tiếp theo. Mục 4 trả lời câu hỏi "điểm là bao nhiêu"; mục này trả lời "vì sao điểm như vậy và đã tìm ra lỗi bằng cách nào".

### 6.1. Quy trình kiểm tra áp dụng cho mọi lần nộp

**Trước khi nộp** (không tốn API):
1. `neorag validate public_test --predictions <file>`: đủ số câu, đúng định dạng, id có trong corpus.
2. Mỗi câu đủ 10 `retrieved_ids`, không trùng; câu trả lời không rỗng, dưới 2.000 ký tự.
3. So với lần nộp trước: số câu có top-5 thay đổi, độ dài trung bình, số câu trả lời "không có thông tin". Khi lần nộp chỉ định đổi prompt, kiểm tra `retrieved_ids` **giống hệt** lần trước để chắc chắn chỉ một yếu tố thay đổi.
4. Đọc vài câu trả lời của bản cũ và bản mới cạnh nhau.

**Sau khi nộp:**
1. Lấy log **Scoring output** (trang thi → Actions → LOGS).
2. So từng metric với lần nộp trước và với baseline, không chỉ nhìn Raw score.
3. Với metric bất thường: đọc lại prompt judge trong `src/neorag/metrics/judge.py` để hiểu metric đó phạt hành vi nào, rồi tìm câu trả lời có hành vi đó.
4. Phân loại lỗi về đúng tầng (retrieval / sinh câu trả lời / tiếng lóng / dữ liệu thật sự thiếu) trước khi sửa.
5. Ghi kết quả vào mục này và mục 4.

### 6.2. Lần nộp 1 – 69.06 (hybrid + prompt v1)

**Kiểm tra trước khi nộp:** file hợp lệ (42 câu, mỗi câu 10 id). Đọc 5 câu trả lời đầu: đúng ý, câu từ chối đúng mẫu.

**Đọc log:**

| Quan sát | Diễn giải |
|---|---|
| Retrieval 64.03, trong khi trên train là 82.3 (lệch 18 điểm) | Public Test có khoảng một nửa câu không dấu/tiếng lóng; trên train, chính nhóm "không dấu" là nhóm yếu nhất (65.4). Lệch không phải do lỗi code mà do phân bố câu hỏi khác nhau |
| Response Relevancy 59.03, **thấp hơn baseline** (60.07) | Câu trả lời lan sang ý khác → câu hỏi do judge sinh ngược cũng lệch theo |
| 1 − Noise Sensitivity 68.80: khoảng 31% số câu bị chấm "sai" | Đọc `judge.py`: nhãn `incorrect` gồm cả "thông tin không có trong tham chiếu". **Lỗi phát hiện:** câu đúng sự thật nhưng lấy từ đoạn không thuộc gold vẫn bị trừ điểm |
| Groundedness 93.59, Abstention 100 | Hệ thống gần như không bịa và từ chối đúng |

**Kiểm tra bổ sung trên train** (`scripts/analyze_predictions.py`, offline): 17 câu `retrieval_miss`, 13 câu `false_abstain` (phần lớn là hệ quả của retrieval trượt), 15 câu `missing_fact`, 0 câu `missed_abstain`.

**Lỗi gốc xác định được:**
1. Retrieval kém với câu không dấu (bge-m3 hiểu kém câu không dấu).
2. Câu hỏi nhiều CLB: các đoạn chung của S2B chiếm hết top-5 (ví dụ `train-0017` được 0 điểm).

**Quyết định:** làm bước 5 (viết lại câu hỏi) và bước 6 (reranker) trước khi đụng đến prompt, vì retrieval vừa chiếm 50% điểm vừa kéo theo String Presence và Noise.

### 6.3. Lần nộp 2 – 80.78 (+ bước 5 viết lại câu hỏi, bước 6 reranker)

**Kiểm tra trước khi nộp:**
- 39/42 câu có top-5 thay đổi so với lần 1; số câu "không có thông tin" giảm từ 18 xuống 12.
- Câu *"e ko co hoa tay, quay chup cung ko biet… vao clb mm co dat dien ko"*: lần 1 từ chối, lần 2 trả lời đúng ý.
- **Lỗi phát hiện trước khi nộp:** thời gian 27 giây mỗi câu do reranker chạy trên CPU. Chấp nhận được cho file nộp, không chấp nhận được cho chatbot.
- **Lỗi phát hiện trước khi nộp:** câu trả lời vẫn nói thừa, ví dụ *"Tài liệu chỉ nêu GCC tổ chức các buổi training… nhưng không đề cập địa điểm"*. Cố ý chưa sửa, để lần nộp này chỉ đo tác dụng của retrieval.

**Đọc log:**

| Quan sát | Diễn giải |
|---|---|
| Retrieval +17.2 (64.03 → 81.26) | Bước 5 + 6 có tác dụng trên dữ liệu thật đúng như trên train |
| String Presence +12.6, Noise +17.5 | Phần lớn câu "sai" ở lần 1 thật ra do model không được thấy đoạn đúng |
| Response Relevancy giảm nhẹ (59.03 → 57.85) | Metric này phụ thuộc văn phong, không phụ thuộc retrieval → việc của prompt |
| ROUGE-L 57.48, thấp thứ hai | Câu trả lời dài khoảng 270 ký tự, đáp án mẫu khoảng 200 ký tự |

**Quyết định:** bước 7 – prompt v2 nhắm vào Relevancy và Noise: câu đầu trả lời thẳng, mỗi câu phải trả lời một phần câu hỏi, câu từ chối chỉ một câu.

### 6.4. Lần nộp 3 – 81.44 (prompt v2)

**Kiểm tra trước khi nộp:**
- `retrieved_ids` giống hệt lần 2 → lần nộp này chỉ đo tác dụng của prompt.
- Độ dài trung bình giảm từ 271 xuống 227 ký tự; số câu từ chối giữ nguyên 12.
- So sánh trên mẫu 30 câu train (`scripts/eval_generation.py`): String Presence **giảm 3 điểm** (92.9 → 89.9). Đọc từng câu:
  - **Lỗi thật:** khi rút gọn, model bỏ chi tiết phụ mà đáp án mẫu tính là key fact ("tiền bối" ở `train-0015`, "CDIT" ở `train-0074`).
  - **Báo động giả:** 2 câu `false_abstain` mới đều là câu cả v1 lẫn v2 cùng sai vì retrieval trượt; v2 chỉ viết phần thiếu bằng câu "Tài liệu hiện có không có thông tin…" nên công cụ nhận nhầm là câu từ chối. Đây là **giới hạn của công cụ phân tích**: cụm từ từ chối xuất hiện cả trong câu trả lời một phần.
  - Kết luận lúc đó: 30 câu là quá ít để quyết định, vẫn nộp để đo bằng judge thật.

**Đọc log:** cả 5 metric generation đều tăng (String Presence +1.9, ROUGE-L +0.65, Groundedness +1.3, Relevancy +2.0, Noise +2.1). Raw +0.66, nằm trong biên dao động ±1 của judge, nhưng việc mọi metric cùng tăng cho thấy cải thiện là thật.

**Lỗi trong cách đánh giá được phát hiện:** mẫu train dự đoán String Presence giảm, Public Test lại tăng. Từ đây, mẫu nhỏ chỉ dùng để bắt lỗi rõ ràng (câu trả lời rỗng, sai định dạng, giảm mạnh), không dùng để chọn giữa hai bản gần nhau.

### 6.5. Kiểm tra sau lần nộp 3 – vì sao retrieval Public Test (81.26) vẫn thấp hơn train (91.4)?

Public Test không có đáp án chuẩn, nên kiểm tra theo cách: lấy 12 câu trả lời có cụm "không có thông tin", dò từ khóa của câu hỏi trong **toàn bộ corpus**, rồi đọc các đoạn tìm được (offline, không tốn API).

| Loại | Số câu | Ví dụ | Đánh giá |
|---|---|---|---|
| Corpus thật sự không có | 4 | phòng riêng của GCC, mức phí cụ thể, phần mềm dựng video, engine game của ProPTIT | Từ chối đúng |
| Câu hỏi có một phần không trả lời được | 3 | "so với trung tâm bên ngoài thì…", "sao không học thẳng Python" | Hợp lý |
| **Retrieval trượt: đoạn đúng có trong corpus** | **5** | link đăng ký ProPTIT (đoạn 6), số vòng tuyển ProPTIT (đoạn 5), slogan Multimedia (đoạn 86), thời hạn xin nghỉ (đoạn 72), blog ProPTIT (đoạn 78) | **Lỗi cần sửa** |

**Lần theo từng tầng** để tìm chỗ đoạn đúng bị mất (hạng của đoạn đúng ở mỗi tầng):

| Câu | Hạng trong bảng của câu hỏi con tương ứng | Sau khi gộp các câu | Hạng theo reranker | Hạng cuối |
|---|---|---|---|---|
| Thời hạn xin nghỉ | **1** | 16 | **2** | 9 (ngoài 8 đoạn đưa vào prompt) |
| Link đăng ký ProPTIT | **2** | ngoài 20 | – | ngoài 10 |
| Số vòng tuyển ProPTIT | **1** | ngoài 20 | – | ngoài 10 |
| Slogan Multimedia | **3** | 19 | **3** | 11 |

**Lỗi gốc:** câu hỏi con tìm đúng đoạn, nhưng
1. gộp 4–6 bảng xếp hạng bằng RRF làm **loãng** tín hiệu: đoạn đúng chỉ đứng cao trong một bảng, còn các đoạn chung chung (giới thiệu S2B, tuyển thành viên GCC) đứng vừa phải ở nhiều bảng nên cộng dồn được nhiều điểm hơn;
2. bước gộp cuối (retriever : reranker = 1 : 1) kéo đoạn đúng xuống lần nữa, kể cả khi reranker đã xếp nó lên hạng 2–3.

Lỗi này ít lộ ra trên train vì câu hỏi train ít được tách thành nhiều câu con như vậy; Public Test có nhiều câu "cả 3 CLB" hoặc hỏi 2–3 ý trong một câu.

**Hướng sửa đề xuất** (áp dụng chung, không nhắm vào câu test cụ thể): giữ chỗ top-2/3 của mỗi câu hỏi con trong danh sách ứng viên; gộp bước cuối theo thứ hạng tốt nhất hoặc tăng trọng số reranker khi có câu hỏi con; tăng `n_context` từ 8 lên 10. Đo trước trên train (offline, cần GPU).

**Sửa và đo (chỉ chọn tham số trên train).** Năm câu Public Test ở trên chỉ dùng để chẩn đoán loại lỗi, không dùng làm thước đo chọn cấu hình: dùng chúng để tinh chỉnh sẽ gần với "tra cứu thủ công câu hỏi Public Test", điều luật cấm.

Hai tham số mới: `MultiQueryRetriever(reserve=k)` giữ chỗ top-k của mỗi câu con trong danh sách ứng viên; `RerankedRetriever(multi_weights=...)` dùng trọng số gộp riêng cho câu có câu con. Lượt đo chạy với client mạng bị gỡ khỏi `LLMClient` (cache thiếu thì báo lỗi ngay), xác nhận không tốn API.

| reserve | Trọng số cho câu có câu con | SCORE (122 câu) | 48 câu có câu con | 74 câu còn lại | So sánh | Nhiều đoạn |
|---|---|---|---|---|---|---|
| 0 | (1, 1) – cũ | 91.4 | 89.7 | 92.5 | 88.8 | 87.7 |
| 0 | (1, 2) | 91.7 | 90.5 | 92.5 | 89.1 | 88.5 |
| 0 | (1, 3) | 91.4 | 89.6 | 92.5 | 89.3 | 88.5 |
| 0 | **(0, 1)** | **92.1** | **91.4** | 92.5 | **91.1** | **89.8** |
| 1–3 | như trên | giống hệt từng dòng tương ứng (± 0.1) | | | | |

- **Trọng số (0, 1) cho câu có câu con** – tức xếp hạng hoàn toàn theo reranker – tốt nhất: +1.7 trên nhóm có câu con, nhóm còn lại không đổi (đúng thiết kế). Khớp với chẩn đoán: thứ hạng của retriever bị loãng, còn điểm reranker lấy max theo từng câu con nên không bị loãng.
- **`reserve` không đổi điểm trên train**, vì lỗi "đoạn đúng bị đẩy khỏi 20 ứng viên" hầu như không xảy ra ở train. Vẫn bật `reserve=2` vì nó nhắm đúng lỗi quan sát được ở Public Test mà không làm hại train.
- **`n_context` giữ 8:** với cấu hình mới, không đoạn đúng nào trên train nằm ở hạng 9–10 (0/196), nên tăng lên 10 chỉ thêm nhiễu và token.
- Retrieval trên train: **91.4 → 92.1** (R@5 95.2, MRR@5 91.4, NDCG@5 89.6).

### 6.5b. Lần nộp 4 – 85.20 (sửa bước gộp: `reserve=2`, `multi_weights=(0, 1)`)

**Giả thuyết:** Retrieval trên Public Test tăng, vì lỗi "gộp làm loãng tín hiệu của câu hỏi con" (mục 6.5) xảy ra nhiều ở Public Test hơn ở train. Prompt giữ nguyên v2 để chỉ đo tác dụng của retrieval.

**Kiểm tra trước khi nộp:**
- File hợp lệ, mỗi câu đủ 10 id không trùng; độ dài trung bình 227 → 229 ký tự.
- 25/42 câu có tập top-5 khác lần 3 – thay đổi lớn, vì Public Test có nhiều câu được tách thành câu con.
- Số câu có cụm "không có thông tin": 12 → 10. **3 câu được sửa** (thời hạn xin nghỉ, link đăng ký ProPTIT, số vòng tuyển ProPTIT: cả ba giờ trả lời được từ đúng đoạn tài liệu).
- **1 câu mới bị từ chối sai** (`public-0040`, "ngành Marketing có tham gia CLB làm game được không"): thông tin "dành cho sinh viên mọi ngành học" nằm lẫn trong đoạn 98 có chủ đề *Lịch sử & thành lập*. Reranker chấm đoạn này thấp vì chủ đề chính của nó không khớp câu hỏi; khi xếp hạng hoàn toàn theo reranker, đoạn tụt từ hạng 6 xuống hạng 10, ra ngoài prompt. Đây là **cái giá của việc tin hoàn toàn vào reranker**: một thông tin nằm lạc trong đoạn có chủ đề khác. Chấp nhận, vì trên train cấu hình mới tốt hơn (+0.7) và trên Public Test số câu được sửa nhiều hơn số câu bị hỏng.

**Log (so với lần 3):**

| Metric | Lần 3 | Lần 4 | Thay đổi |
|---|---|---|---|
| Recall@5 | 81.71 | 88.65 | +6.94 |
| MRR@5 | 84.76 | 89.76 | +5.00 |
| NDCG@5 | 77.30 | 85.06 | +7.76 |
| **Retrieval** | 81.26 | **87.83** | **+6.57** |
| String Presence | 86.32 | 87.61 | +1.29 |
| ROUGE-L | 58.13 | 58.84 | +0.71 |
| Groundedness | 97.01 | 97.86 | +0.85 |
| Response Relevancy | 59.88 | 59.73 | −0.15 |
| 1 − Noise Sensitivity | 88.46 | 91.45 | +2.99 |
| Abstention | 100 | 100 | = |
| **Generation** | 81.63 | **82.58** | +0.95 |
| **Raw score** | 81.44 | **85.20** | **+3.76** |

**Giả thuyết đúng.** Đây là mức tăng lớn nhất kể từ lần 2, và nằm xa ngoài biên dao động ±1 của judge.

- **Mức tăng trên Public Test (+6.57 retrieval) lớn gấp khoảng 9 lần trên train (+0.7).** Nếu chỉ nhìn train, thay đổi này trông không đáng làm. Lỗi chỉ lộ ra khi đọc kỹ các câu Public Test bị từ chối và lần theo hạng của đoạn đúng qua từng tầng. Bài học: train không chứa đủ các dạng lỗi của test; phân tích lỗi trên test (chỉ để chẩn đoán, không để chọn tham số) tìm ra được những lỗi mà điểm train che mất.
- Cách làm vẫn tuân thủ luật: chẩn đoán trên Public Test để tìm **loại** lỗi, sửa bằng một cơ chế **chung**, chọn tham số và kiểm tra "không làm hại" **trên train**.
- Retrieval tốt hơn tiếp tục kéo generation lên (Noise +3.0, String Presence +1.3), dù prompt không đổi.
- Câu `public-0040` bị hỏng (mục trên) không làm tổng điểm giảm: số câu được sửa nhiều hơn.
- Response Relevancy (59.73) giờ là metric thấp nhất sau ROUGE-L và gần như không nhúc nhích qua 4 lần nộp (57.85–59.88). Giới hạn trần có thể nằm ở chỗ câu hỏi gốc viết không dấu, mà judge so sánh nó với câu hỏi sinh ngược (có dấu) bằng bge-m3.

**Quyết định:** giữ cấu hình mới. Bản này (85.20) là bản tốt nhất đến nay.

### 6.5c. Lần nộp 5 – 84.86 (A1 `n_candidates=30` + A3 doc2query bảng BM25 riêng)

**Ngoại lệ của nguyên tắc "mỗi lần chỉ đổi một thứ":** lần này đổi hai thứ (A1 và A3), nên chỉ đo được tác dụng gộp. Chấp nhận vì cả hai chỉ tác động lên retrieval, đã được kiểm tra riêng rẽ trên train (A1 +1.1, A3 +0.7), và lượt nộp còn hạn chế.

**Giả thuyết:** Retrieval trên Public Test tăng, nhiều nhất ở nhóm không dấu và diễn đạt khác (hai nhóm A3 giúp nhiều nhất trên train, và chiếm phần lớn Public Test). Prompt giữ nguyên v2.

**Kiểm tra trước khi nộp** (`outputs/public_test_v5.jsonl`, 42 lời gọi sinh câu trả lời vì sang ngày mới, các bước khác dùng cache):
- File hợp lệ, mỗi câu đủ 10 id; không câu nào rơi vào câu dự phòng; độ dài trung bình 48.2 → 47.4 từ.
- 21/42 câu có tập top-5 khác lần 4, nhưng chỉ 4/42 câu đổi đoạn hạng 1: phần lớn thay đổi nằm ở hạng 2–5.
- Số câu có cụm từ chối: 10 → 10, nhưng có hai câu đổi chiều:
  - **`public-0033` được sửa** ("mỗi CLB trong S2B dùng slogan nào"): lần 4 thiếu slogan của Multimedia; lần 5 đoạn 86 lọt vào top-5 và câu trả lời đủ cả ba CLB. Đúng loại lỗi A3 nhắm tới.
  - `public-0044` (đội đoạt giải Huawei có ai từng làm chủ nhiệm): retrieval giống hệt lần 4; câu trả lời vẫn đúng nhưng thêm một câu "không có thông tin về hai thành viên còn lại". Đây không phải từ chối thật mà là dao động của LLM khi sinh lại (cache mất vì sang ngày mới); câu thừa có thể làm giảm nhẹ Noise.
- `public-0040` vẫn chưa được sửa (đoạn 98 nằm ngoài top-10, xem giới hạn ở bước 10).

**Log (so với lần 4):**

| Metric | Lần 4 | Lần 5 | Thay đổi |
|---|---|---|---|
| Recall@5 | 88.65 | 86.94 | −1.71 |
| MRR@5 | 89.76 | 89.37 | −0.39 |
| NDCG@5 | 85.06 | 84.60 | −0.46 |
| **Retrieval** | 87.83 | 86.97 | **−0.86** |
| String Presence | 87.61 | 87.18 | −0.43 |
| ROUGE-L | 58.84 | 57.61 | −1.23 |
| Groundedness | 97.86 | 97.86 | = |
| Response Relevancy | 59.73 | 59.80 | +0.07 |
| 1 − Noise Sensitivity | 91.45 | 94.02 | +2.57 |
| Abstention | 100 | 100 | = |
| **Generation** | 82.58 | 82.74 | +0.16 |
| **Raw score** | 85.20 | 84.86 | **−0.34** |

**Giả thuyết sai trên Public Test.** Retrieval giảm thay vì tăng.

- **Retrieval −0.86 là thật**, vì metric retrieval tính tất định, không qua judge. Nhưng mức giảm nhỏ: trên 42 câu, −0.86 tương đương khoảng một câu đổi thứ hạng đáng kể. Recall@5 giảm nhiều nhất (−1.71), nghĩa là có đoạn đúng bị đẩy ra khỏi top-5. Như vậy A1+A3 làm lợi (`public-0033`) ở một số câu nhưng làm hại nhiều hơn ở câu khác.
- **Generation +0.16 nằm trong biên dao động ±1 của judge**; Noise +2.57 nhưng ROUGE-L −1.23, phần lớn do câu trả lời được sinh lại (ngày mới, mất cache), không phải do retrieval.
- **Train và Public Test mâu thuẫn**: train (122 câu) nói A1+A3 +1.8, Public Test (42 câu) nói −0.86. Cả hai mẫu đều nhỏ.

**Không quy được về A1 hay A3**, vì đổi hai thứ cùng lúc: đây chính là cái giá của ngoại lệ đã ghi ở trên. Tái tạo lại xếp hạng trên Public Test (không gọi API, khớp 42/42 với cả file lần 4 và lần 5) cho thấy:

| So với lần 4 | Câu đổi tập top-5 | Câu đổi hạng 1 |
|---|---|---|
| Chỉ A1 (`n_candidates=30`) | 16 | 1 |
| Chỉ A3 (doc2query) | 16 | 3 |
| A1 + A3 (lần 5) | 21 | 4 |

Cả hai đều tác động đáng kể; không có đáp án nên không biết thay đổi nào là đúng. Không đọc từng câu Public Test để chọn cấu hình (sẽ thành tinh chỉnh trên test).

**Quyết định:** chưa kết luận. Bản tốt nhất vẫn là lần 4 (85.20). Bước tiếp: nộp **lần 6 = chỉ A1** (tắt A3) để tách riêng tác dụng của A3.

### 6.5d. Lần nộp 6 – 85.55 (chỉ A1: `n_candidates=30`, tắt A3)

**Mục đích:** tách riêng tác dụng của A3. So với lần 5 chỉ đổi một thứ (tắt doc2query); so với lần 4 chỉ đổi một thứ (`n_candidates` 20 → 30).

**Cách chạy (không sửa code của giải pháp):** một lớp con tạm `A1Only(MyPipeline)` truyền `doc2query=False`, đặt ngoài repo và nạp qua `PYTHONPATH`, rồi `neorag predict public_test --pipeline v6_pipeline:A1Only --out outputs/public_test_v6.jsonl`.

**Kiểm tra trước khi nộp:**
- File hợp lệ, đủ 10 id mỗi câu, không có câu dự phòng; độ dài trung bình 49.2 từ.
- Retrieval khớp đúng bản tái tạo ở mục 6.5c: so với lần 4 đổi top-5 ở 16 câu, đổi hạng 1 ở 1 câu (`public-0027`).
- 20/42 câu trả lời giống hệt lần 5 (cùng ngữ cảnh, dùng cache), khoảng 22 lời gọi mới.
- Số câu từ chối 10, giống lần 4 và lần 5. `public-0033` (slogan S2B) **cũng được sửa khi chỉ có A1**: đoạn Multimedia lọt vào top nhờ tăng số ứng viên cho reranker, không cần A3. `public-0044` có câu thừa giống lần 5 (dao động khi sinh lại).

**Cách đọc kết quả:**
- Lần 6 > lần 5: A3 có hại trên Public Test → mặc định `doc2query=False`.
- Lần 6 ≤ lần 5: A1 là nguyên nhân → quay về `n_candidates=20`, cân nhắc giữ A3.
- Lần 6 vẫn < 85.20: quay về cấu hình lần 4.

**Log:**

| Metric | Lần 4 (gốc) | Lần 5 (A1 + A3) | Lần 6 (chỉ A1) | 6 so với 5 = tác dụng của A3 (đảo dấu) | 6 so với 4 = tác dụng của A1 |
|---|---|---|---|---|---|
| Recall@5 | 88.65 | 86.94 | 89.13 | +2.19 | +0.48 |
| MRR@5 | 89.76 | 89.37 | 89.37 | = | −0.39 |
| NDCG@5 | 85.06 | 84.60 | 85.32 | +0.72 | +0.26 |
| **Retrieval** | 87.83 | 86.97 | **87.94** | **+0.97** | +0.11 |
| String Presence | 87.61 | 87.18 | 88.03 | +0.85 | +0.42 |
| ROUGE-L | 58.84 | 57.61 | 57.76 | +0.15 | −1.08 |
| Groundedness | 97.86 | 97.86 | 98.72 | +0.86 | +0.86 |
| Response Relevancy | 59.73 | 59.80 | 58.33 | −1.47 | −1.40 |
| 1 − Noise Sensitivity | 91.45 | 94.02 | 96.15 | +2.13 | +4.70 |
| Abstention | 100 | 100 | 100 | = | = |
| **Generation** | 82.58 | 82.74 | **83.17** | +0.43 | +0.59 |
| **Raw score** | 85.20 | 84.86 | **85.55** | **+0.69** | +0.35 |

**Kết luận:**
- **A3 có hại trên Public Test**: tắt A3 thì retrieval +0.97, gần như toàn bộ đến từ Recall@5 (+2.19), tức A3 đẩy đoạn đúng ra khỏi top-5 ở vài câu. Ngược chiều với train (+0.7). Retrieval tính tất định nên đây không phải dao động của judge.
- **A1 trung tính đến hơi tốt**: retrieval +0.11 so với lần 4 (Recall@5 +0.48, MRR −0.39), và sửa được `public-0033` mà trước đó tưởng là công của A3.
- Phần generation (+0.43, +0.59) nằm trong biên ±1 của judge; Noise tăng đều qua 3 lần nộp nhưng Response Relevancy giảm, không nên đọc quá nhiều vào đó.
- Giả thuyết vì sao A3 hại trên Public Test nhưng giúp trên train: câu hỏi sinh thêm được viết từ chính đoạn văn, nên giống phong cách câu hỏi train (cũng do người viết từ đoạn văn). Câu Public Test xa đoạn văn hơn, và các câu hỏi sinh thêm, vốn lặp tên CLB và từ ngữ chung, kéo lên những đoạn "có vẻ khớp" về chủ đề nhưng không chứa thông tin. Chưa kiểm chứng, và không kiểm chứng bằng cách đọc từng câu Public Test.

**Quyết định:** cấu hình lần 6 là bản tốt nhất (**85.55**). Đổi mặc định `MyPipeline(doc2query=False)`; giữ code A3 (`enrich.py`, `with_doc2query`, `--retriever full-a3`) như một thí nghiệm có ghi chép, bật lại bằng `MyPipeline(doc2query=True)`.

### 6.5e. Lần nộp 7 – 87.22 (chỉ A4: LLM xếp hạng lại 15 ứng viên đầu)

**Mục đích:** đo tác dụng của A4 (bước 11) so với lần 6. Chỉ đổi một thứ: `MyPipeline(llm_rerank=True)`; mọi tham số khác giữ như lần 6.

**Thiết kế (đã duyệt trước khi viết code):** `ListwiseReranker` (`llm_rerank.py`) lấy 15 đoạn đầu sau cross-encoder, 1 lời gọi Haiku trả về `{"ranking": [id...]}`, gộp RRF trọng số 1 : 1 với bảng cũ. Số ứng viên (15) và trọng số đặt trước, **không chỉnh theo train**. Lỗi (JSON hỏng, id lạ/trùng, lỗi mạng) đều quay về bảng cũ; prompt không chứa ngày hôm nay.

**Đo trước khi nộp:**
- Train (122 câu, khoảng 122 lời gọi): retrieval **93.2 → 97.2** (R@5 96.0 → 97.5, MRR@5 92.6 → 97.8, NDCG@5 90.9 → 96.1). Mọi nhóm đều tăng (không dấu 89.5 → 93.9, so sánh 94.0 → 98.5, nhiều đoạn 90.4 → 94.2). Các câu còn sai đều là câu cần 2–3 đoạn mà đoạn thiếu không nằm trong top-5.
- Public Test (khoảng 42 lời gọi): **30/42 câu đổi tập top-5**, 40/42 đổi thứ tự, 7 câu đổi hạng 1, nhiều hơn hẳn A1 (16 câu) và A3 (21 câu). Cảnh báo trước khi nộp: thay đổi mạnh nên điểm có thể tăng hoặc giảm. Đọc các câu đổi (chỉ để chẩn đoán): `public-0027`, `0003`, `0028`, `0018`, `0016` trông đúng hướng; `public-0044` có thể mất đoạn 68; `0002`, `0006`, `0014` có đoạn bị thay bằng đoạn kém liên quan hơn; nhiều câu chỉ đổi chỗ giữa các đoạn cùng chủ đề.
- File nộp: 42 câu, đủ 10 id không trùng, không có câu dự phòng, median 3.2 giây/câu (cache trống), chỉ 3/42 câu trả lời giống lần 6. Số câu chứa "không có thông tin" 10 → 9: `public-0040` và `public-0044` từ chối → trả lời được.

**Log:**

| Metric | Lần 6 | Lần 7 (A4) | Thay đổi |
|---|---|---|---|
| Recall@5 | 89.13 | 90.71 | +1.58 |
| MRR@5 | 89.37 | **95.24** | **+5.87** |
| NDCG@5 | 85.32 | 90.07 | +4.75 |
| **Retrieval** | 87.94 | **92.01** | **+4.07** |
| String Presence | 88.03 | 88.46 | +0.43 |
| ROUGE-L | 57.76 | 59.67 | +1.91 |
| Groundedness | 98.72 | 96.15 | **−2.57** |
| Response Relevancy | 58.33 | 59.71 | +1.38 |
| 1 − Noise Sensitivity | 96.15 | 90.60 | **−5.55** |
| Abstention | 100 | 100 | = |
| **Generation** | 83.17 | 82.43 | **−0.74** |
| **Raw score** | 85.55 | **87.22** | **+1.67** |

**Kết luận:**
- **A4 giúp rõ rệt**: retrieval +4.07 trên Public Test, gần đúng mức +4.0 trên train (lần này train dự báo được test, khác A3). Tiêu chí đặt trước (retrieval lớn hơn 87.94) đạt → **bật A4 mặc định** (`llm_rerank=True`).
- Mức tăng nằm chủ yếu ở MRR (+5.87): LLM đưa đoạn đúng lên hạng đầu. Recall@5 chỉ +1.58 vì A4 không thêm được đoạn ngoài 15 ứng viên: Recall@5 (90.71) vẫn thấp hơn MRR@5 (95.24) nhiều, tức còn thiếu đoạn trong top-5 ở câu nhiều đoạn.
- Retrieval +4.07 đóng góp khoảng +2.04 vào raw, Generation −0.74 làm mất khoảng 0.37: ròng +1.67.
- **Hai điểm đi lùi, không được dự đoán trước:** Noise −5.55 và Groundedness −2.57. Nguyên nhân chưa kiểm chứng: (a) dao động của judge và việc sinh lại câu trả lời (Noise qua các lần nộp: 91.45 → 94.02 → 96.15 → 90.60, lần 7 chỉ quay về mức lần 4); (b) câu trả lời mới đưa thêm chi tiết ở các câu từng từ chối (`0040`, `0044`), nhiều câu hơn để judge gắn nhãn `incorrect`/`unsupported`; (c) thứ tự đoạn đổi nên câu trả lời đổi gần hết (chỉ 3/42 giống lần 6).
- Chi phí lần 7: đo train khoảng $0.07, đo Public Test khoảng $0.02, sinh file nộp khoảng $0.03.

**Nộp lại đúng file lần 7 (ID 33141): 87.34** (lần đầu 33139: 87.22; điểm chuẩn hoá 85.23 so với 84.93). Retrieval tính tất định nên chênh +0.12 raw (generation khoảng +0.24) gần như hoàn toàn do judge: **judge chỉ dao động cỡ 0.1–0.3 raw**. Suy ra Noise −5.55 và Groundedness −2.57 **không phải do judge nhiễu** mà do chính bộ câu trả lời lần 7 (model sinh lại câu trả lời mỗi khi ngữ cảnh đổi, Haiku không tất định). Giả thuyết (a) ở trên được rút lại; nguồn nhiễu lớn là dao động giữa các lần sinh câu trả lời.

**Chẩn đoán trên máy, không qua judge** (trang thi chỉ trả điểm tổng hợp, không có log từng câu; so 42 câu trả lời lần 6 và lần 7, không gọi API; chỉ là suy đoán vì Public Test không có đáp án mẫu):
- Không thấy dấu hiệu A4 làm model bịa: độ phủ ngữ cảnh của câu trả lời (tỉ lệ từ nội dung có trong 10 đoạn đã nộp) 0.93 → 0.914. Số câu có độ phủ dưới 0.7 tăng 4 → 7, nhưng cả 3 câu mới (`0003`, `0036`, `0039`) đều là **câu từ chối** ("Tài liệu hiện có không có thông tin về …"), không phải câu khẳng định sai.
- Tổng số câu trong câu trả lời giảm 89 → 81 (−9%), nên mỗi câu nặng hơn. Số câu trả lời có cả đáp án lẫn một câu từ chối: 8 → 7; câu trả lời chỉ từ chối: 2 → 3.
- Nghi ngờ chính: câu "không có thông tin về …" chèn vào câu trả lời có đáp án (do quy tắc 8 của prompt v2). Judge Noise gắn `incorrect` cho "thông tin không có trong tham chiếu", Groundedness có thể gắn `unsupported`. Hai câu mới có thêm vế này: `public-0045` ("không có thông tin về địa điểm bàn đón tân sinh viên"), `public-0026` ("không nêu team riêng nào dành cho ngành Khoa học Máy tính"); `public-0015`, `0036` từ 3 câu xuống 2 nên tỉ lệ câu từ chối tăng khoảng 1/3 → 1/2. Cộng nhẩm khoảng 1–1.5 câu hỏi, gần khớp mức giảm khoảng 2 câu hỏi của Noise (chưa kiểm chứng). `public-0040` và `0044` chuyển từ có vế từ chối sang không có, giảm bớt thiệt hại này.
- `public-0027` (Premiere): chỉ còn một câu từ chối, bỏ câu thừa của lần 6; không đoạn nào trong top-8 nhắc phần mềm dựng video nên từ chối hợp lý. `public-0003`: A4 đưa đoạn *Cơ cấu* (id 94) lên top-5 nhưng đoạn này không nói rõ ban nào lo mảng lên kế hoạch, nên từ chối vẫn có lý.
- Kết luận: Noise/Groundedness giảm nhiều khả năng do model chèn thêm vế "không có thông tin về …" ở vài câu cộng dao động sinh lại, **không phải lỗi của A4**. Cơ chế này có ở cả lần 6 (8 câu) và lần 7 (7 câu); lần 6 may mắn hơn.

**Việc tiếp theo (chưa làm, cần chọn):**
1. Sửa quy tắc 8 của prompt v2 (đã trả lời được ý chính thì không thêm vế "tài liệu không có thông tin về phần phụ"); kỳ vọng Noise +3–5 điểm, tức raw +0.3–0.4; rủi ro với câu loại `thiếu dữ liệu`; nộp riêng thành lần 8 (khoảng $0.03).
2. Retrieval (đòn bẩy lớn hơn nhưng chưa chắc): Recall@5 90.71 còn thấp hơn MRR@5 95.24 nhiều; thử Step-Back hoặc A5.
3. Đo độ dao động sinh lại câu trả lời (sinh lại lần 7 với cache trống, khoảng $0.03, tốn một lượt nộp) để biết ±bao nhiêu điểm là nhiễu khi so các lần nộp sau.

### 6.5f. Lần nộp 8 – 86.04 (A4 + A5: LLM đọc toàn bộ corpus)

**Mục đích:** đo A5 (bước 11) so với lần 7. Chỉ đổi một thứ: `corpus_rank=True`. Mọi tham số khác giữ như lần 7.

**Thiết kế (đặt trước khi đo, không chỉnh theo train):** `CorpusRanker` (`corpus_rank.py`) bọc retriever A4. Corpus chỉ có 118 đoạn (khoảng 50 nghìn ký tự, khoảng 20 nghìn token) nên vừa một prompt. 1 lời gọi Haiku nhận toàn bộ corpus ở system message, câu hỏi (kèm bản viết lại và các ý cần tra) ở user message, trả về `{"ranking": [id...]}` tối đa 10 id; gộp RRF trọng số 1 : 1 với danh sách của A4. Khác A4, A5 nhìn được cả đoạn nằm ngoài 15 ứng viên (6/196 đoạn đúng trên train). Lỗi (JSON hỏng, id lạ/trùng, lỗi mạng) quay về kết quả A4. Chi phí khoảng 20 nghìn token đầu vào mỗi câu, khoảng $0.002.

**Đo trước khi nộp:**
- Unit test (parse, prompt chứa đủ corpus và không có ngày, đưa được đoạn ngoài danh sách cũ, trọng số 0 giữ nguyên thứ tự, lùi về A4 khi lỗi).
- Train (122 câu): retrieval **97.2 → 99.2**. Tiêu chí đặt trước (không thấp hơn A4) đạt.
- Public Test: retrieval **92.01 → 93.84 (+1.83)**.

**Log:**

| Metric | Lần 7 (A4) | Lần 8 (A4 + A5) | Thay đổi |
|---|---|---|---|
| Recall@5 | 90.71 | 93.29 | +2.58 |
| MRR@5 | 95.24 | 95.83 | +0.59 |
| NDCG@5 | 90.07 | 92.39 | +2.32 |
| **Retrieval** | 92.01 | **93.84** | **+1.83** |
| String Presence | 88.46 | 91.88 | +3.42 |
| ROUGE-L | 59.67 | 60.92 | +1.25 |
| Groundedness | 96.15 | 96.79 | +0.64 |
| Response Relevancy | 59.71 | 59.98 | +0.27 |
| 1 − Noise Sensitivity | 90.60 | 93.16 | +2.56 |
| Abstention | 100 | **66.67** | **−33.33** |
| **Generation** | 82.43 | 78.23 | −4.20 |
| **Raw score** | 87.22 | **86.04** | **−1.18** |

**Kết luận:**
- **8/9 chỉ số tăng.** Retrieval tăng đúng chỗ A5 nhắm tới (Recall@5 +2.58: thêm được đoạn thiếu trong top-5 ở câu nhiều đoạn), String Presence +3.42, Noise +2.56.
- **Raw giảm chỉ vì Abstention.** Abstention chỉ chấm trên câu không có đáp án (Public Test có 3 câu), nên một câu đổi chiều làm Abstention −33.33, tức Generation −5.55 và raw −2.78. Nếu giữ Abstention 100, Generation là 83.78 và raw khoảng **88.8** (phép tính ngược, chưa kiểm chứng).
- Câu nghi ngờ là `public-0003`: câu hỏi gián tiếp về hoàn cảnh cá nhân; A4 đã đưa đoạn *Cơ cấu* (id 94) lên top-5 từ lần 7, A5 đưa thêm đoạn *Lĩnh vực & hoạt động* (id 87) vào ngữ cảnh, và ở lần 7 câu trả lời của câu này là từ chối. Chưa có log từng câu của trang thi nên đây là suy đoán, không phải kết quả đo.
- **Độ trễ:** 7 phút 52 giây cho 42 câu, 4/42 câu mất hơn 60 giây (gateway chậm ở lời gọi A5, cả corpus ở đầu vào). Với giới hạn thời gian của ngày thi, đây là rủi ro.

**Kiểm tra sau lần nộp (không qua judge, khoảng $0.04):**
- Chạy 10 câu `không có đáp án` của train (bỏ câu few-shot) với A4 và với A4 + A5, so câu trả lời. **Cả hai từ chối đúng 10/10 câu về nội dung**; 8/10 câu trả lời giống hệt. Hai câu còn lại chỉ thêm một vế thông tin liên quan sau khi đã từ chối (`train-0050`: "Tài liệu chỉ nêu Big Game S2B là hoạt động ngoại khóa…", `train-0103`: liệt kê kênh liên hệ). A5 không làm hỏng việc từ chối nói chung trên train.
- **Giới hạn:** chỉ 10 câu, và đều là câu hỏi thẳng một chi tiết không có (tên bài hát, biệt danh, quỹ hằng tháng), khác `public-0003` (câu gián tiếp, tài liệu trả lời được một phần). 0/10 loại trừ được rủi ro lớn, không loại trừ được rủi ro nhỏ. Giả thuyết hợp lý nhất: dao động khi sinh câu trả lời trên một câu nhập nhằng (cùng cơ chế dao động đã thấy ở lần 7), không phải lỗi của A5.

**Sửa độ trễ:** `A5_TIMEOUT = 20` giây, không thử lại, client riêng cho A5 (các lời gọi khác giữ 60 giây, thử lại 5 lần). Quá hạn thì `CorpusRanker` trả kết quả A4 (có test). Chưa kiểm chứng với gateway chậm thật: lần đo train này A5 median 1.7 giây, tối đa 5.5 giây.

**Quyết định: bật A5 mặc định (`corpus_rank=True`).** Lý do: tiêu chí đặt trước (retrieval Public Test lớn hơn 92.01) đạt với biên rõ (93.84), train khớp hướng (97.2 → 99.2), 8/9 chỉ số tăng. Raw giảm 1.18 đến từ một câu Abstention, mức ước 2.78 raw mỗi câu không có đáp án, dao động theo lần sinh. **Rủi ro đã biết:** Abstention là chỉ số có phương sai cao nhất (3 câu Public Test; ngày thi chưa biết số câu), và A5 có thể làm tăng khả năng trả lời ở câu từ chối một phần. Chưa nộp lại để xác nhận (câu trả lời sinh lại có dao động, nộp lại không chứng minh được nguyên nhân).

### 6.5g. Lần nộp 9 – 90.14 (chạy thử cache trống với cấu hình cuối: A4 + A5 + timeout 20 giây)

**Mục đích:** diễn tập ngày thi (`.cache` đã chuyển đi, tức cache trống, đúng như trên máy sạch) và xem `public-0003` có trả lời "Có…" lần nữa không. Không đổi gì so với lần 8 ngoài bộ đếm và cảnh báo khi A5 lùi về A4 (`calls`, `fallbacks`, log). Chi phí khoảng $0.12.

**Log:**

| Metric | Lần 8 | Lần 9 | Thay đổi |
|---|---|---|---|
| Recall@5 | 93.29 | 95.67 | +2.38 |
| MRR@5 | 95.83 | 97.02 | +1.19 |
| NDCG@5 | 92.39 | 94.74 | +2.35 |
| **Retrieval** | 93.84 | **95.81** | +1.97 |
| String Presence | 91.88 | 90.60 | −1.28 |
| ROUGE-L | 60.92 | 59.63 | −1.29 |
| Groundedness | 96.79 | 100.00 | +3.21 |
| Response Relevancy | 59.98 | 60.21 | +0.23 |
| 1 − Noise Sensitivity | 93.16 | 96.37 | +3.21 |
| Abstention | 66.67 | 100 | +33.33 |
| **Generation** | 78.23 | **84.47** | +6.24 |
| **Raw score** | 86.04 | **90.14** | **+4.10** |

**Kết luận:**
- **Giả thuyết ở mục 6.5f về `public-0003` sai.** Ở lần 9 câu này vẫn trả lời "Có, CLB Multimedia PTIT dành cho các bạn muốn tham gia tổ chức sự kiện…" mà Abstention vẫn là 100. Vậy `public-0003` không phải câu bị lật ở lần 8 (hoặc judge chấp nhận câu trả lời đó). Câu thật sự bị lật ở lần 8 không xác định được (trang thi không có log từng câu): so câu trả lời lần 8 và 9 không thấy ứng viên rõ ràng. Đây là bài học về suy diễn từ một câu trả lời cụ thể khi không có log: nên ghi là "suy đoán", đúng như đã ghi, và không dựa vào đó để tắt A5.
- **A5 không làm hại Abstention một cách ổn định.** Lần 8: 66.67; lần 9 cùng cấu hình: 100. Chênh +4.10 raw giữa hai lần chạy **cùng một code** cho thấy dao động giữa các lần sinh lại là mức vài điểm raw.
- **Retrieval không tất định khi cache trống.** A4 và A5 là LLM, không đặt được `temperature`. Lần 8 và 9 chỉ cùng tập top-5 ở 16/42 câu (cùng hạng 1 ở 38/42, top-5 trùng trung bình 83%); retrieval 93.84 và 95.81 cho cùng một cấu hình. Nhận định "retrieval là tín hiệu tất định, quyết định trên retrieval" chỉ đúng khi có cache; so hai lần nộp khác nhau về retrieval chênh dưới khoảng 2 điểm không kết luận được.
- **Độ trễ:** median 35.1 giây, tổng 28 phút 26 giây, tối đa 1 phút 40 giây (lần 8 chạy với cache một phần: median 5.1 giây). Pipeline chạy tuần tự, 4 lời gọi LLM mỗi câu; gateway tối hôm đó chậm hơn các lần trước (chưa tách được riêng phần A5). Chỉ 1/42 câu A5 quá 20 giây và lùi về A4 (có cảnh báo trong log). Tài liệu không nêu giới hạn thời gian, Private Test mở cả ngày 08/11.
- **Quyết định: giữ A5 bật mặc định, khóa cấu hình A4 + A5.** Điểm cao nhất 90.14, Abstention 100, retrieval 95.81.
- **Kỳ vọng thực tế cho Private Test:** 86–90 raw tùy lần sinh lại (lần 8: 86.04 và lần 9: 90.14 cho cùng cấu hình). Đừng đọc 90.14 như mức chắc chắn.

### 6.6. Lỗi phát hiện trong chính công cụ và quy trình

Không phải lỗi nào cũng nằm trong pipeline; một số nằm trong công cụ đo, và nếu không phát hiện thì sẽ dẫn đến quyết định sai.

| Lỗi | Phát hiện bằng cách | Hậu quả nếu không sửa | Cách sửa |
|---|---|---|---|
| `analyze_predictions.py` đếm một key fact bị thiếu vào cả `retrieval_miss` lẫn `fact_not_literal` | Con số `fact_not_literal` = 17 quá lớn so với cảm nhận khi đọc câu trả lời | Tưởng đáp án mẫu "khó" và bỏ qua lỗi retrieval (con số đúng là 4) | Mỗi key fact chỉ quy về một nguyên nhân |
| `eval_generation.py --sample 30` chỉ lấy được 26 câu | Đếm lại số câu in ra | Mẫu lệch, thiếu câu ở các nhóm nhỏ | Phương pháp phần dư lớn nhất |
| Test `rrf` dùng ví dụ sai (doc hạng 3 + hạng 1 nhỉnh hơn doc hạng 2 + hạng 2) | Test đỏ; tính tay 1/63 + 1/61 > 2/62 | Sửa nhầm code đúng | Đổi ví dụ cho đúng ý "đồng thuận thắng" |
| Phát hiện câu từ chối bằng cụm "không có thông tin" nhận nhầm câu trả lời một phần | Đọc từng câu `false_abstain` mới của prompt v2 | Kết luận sai rằng v2 từ chối nhiều hơn | Chưa sửa; khi đọc số `false_abstain` phải xem kèm câu trả lời |
| `scripts/install.py --local` (của BTC) cài nhầm PyTorch bản CPU | Reranker chậm; `torch.cuda.is_available()` trả về `False` dù có GPU; đọc script thấy regex `CUDA Version:` không khớp `CUDA UMD Version: 13.3` của driver mới | Reranker chậm gấp 23 lần, thiếu RAM | Cài trực tiếp bản cu130 bằng pip, không sửa script của BTC |
| Gộp A1 và A3 vào một lần nộp (lần 5) | Điểm giảm nhưng không biết do thay đổi nào | Có thể bỏ nhầm A1 (thay đổi tốt) cùng với A3 | Tái tạo thứ hạng offline để xem mỗi thay đổi tác động bao nhiêu câu, rồi nộp lần 6 chỉ với A1 |
| System prompt chứa ngày hôm nay, mà ngày nằm trong khóa cache | Đọc cách `LLMClient` tạo khóa cache | Ước tính chi phí sai: sang ngày mới, lời gọi sinh câu trả lời không dùng lại được cache | Tính vào ước tính chi phí của mỗi lần chạy |

### 6.7. Mẫu ghi cho các lần nộp tiếp theo

```
### Lần nộp N – <điểm> (<thay đổi duy nhất so với lần N-1>)
Giả thuyết: thay đổi này sẽ làm metric <...> tăng vì <...>.
Kiểm tra trước khi nộp: validate; retrieved_ids thay đổi bao nhiêu câu; độ dài; số câu từ chối; 2–3 câu mẫu cũ/mới.
Log: bảng metric so với lần N-1.
Giả thuyết đúng hay sai? Metric nào thay đổi ngoài dự kiến, vì sao?
Lỗi phát hiện: <lỗi> – phát hiện bằng <cách> – nguyên nhân gốc <...>.
Quyết định: <giữ / quay lại / sửa tiếp>.
```

---

## 7. Quản lý chi phí

- Mọi lời gọi LLM và embedding đi qua `LLMClient` có cache SQLite; điểm reranker có cache riêng (`.cache/rerank.sqlite`). Chạy lại cùng câu hỏi không tốn tiền.
- **Không chạy judge trên máy**: điểm judge lấy bằng cách nộp Public Test (miễn phí, 10 lượt/ngày).
- Trước mỗi lần chạy có gọi API: ước tính số request và chi phí rồi mới chạy. `scripts/eval_generation.py` mặc định chỉ ước tính (có kiểm tra cache chính xác cho lời gọi viết lại), phải thêm `--yes` mới chạy thật; thử prompt trên mẫu 30 câu phân tầng thay vì cả 126 câu.
- Giá niêm yết của `claude-haiku-5-5`: $0.10 / 1 triệu token đầu vào, $0.50 / 1 triệu token đầu ra. Mỗi lượt chạy đến nay tốn khoảng $0.02–0.10.
- Các lượt gần đây (ước tính): sinh doc2query cho 118 đoạn khoảng $0.05 (một lần, lưu ở `artifacts/doc2query.jsonl`); file nộp lần 5 khoảng 42 lời gọi, khoảng $0.03; lần 6 khoảng 22 lời gọi, dưới $0.02 (20/42 câu dùng lại cache của lần 5 vì cùng ngày và cùng ngữ cảnh); A4 (lần 7): đo train khoảng 122 lời gọi xếp hạng (khoảng $0.07), Public Test 42 lời gọi (khoảng $0.02), sinh file nộp 42 lời gọi (khoảng $0.03); Step-Back: khoảng 78 lời gọi đo train (khoảng $0.01); prompt v3: hai lần chạy mẫu 30 câu (khoảng $0.03, đa số lời gọi xếp hạng A4 đã có cache); A5 (lần 8): mỗi lời gọi khoảng 20 nghìn token đầu vào (khoảng $0.002), đo train 122 câu, Public Test 42 câu, sinh file nộp 42 câu, kiểm tra Abstention train 10 câu × 2 cấu hình (khoảng $0.04).
- **Mẹo tiết kiệm lượt nộp:** tái tạo thứ hạng Public Test của nhiều cấu hình ngay trên máy (không gọi API, chỉ dùng cache viết lại câu hỏi và embedding) để biết trước mỗi cấu hình đổi bao nhiêu câu, và ước tính chính xác số lời gọi mới (câu nào có 8 đoạn ngữ cảnh giống bản đã chạy trong ngày thì trúng cache).
- **Phát hiện:** system prompt chứa ngày hôm nay, mà ngày là một phần của khóa cache, nên sang ngày mới mọi lời gọi sinh câu trả lời phải gọi lại (bước viết lại và reranker không bị ảnh hưởng).

---

## 8. Sự cố môi trường và cách xử lý

| Sự cố | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: my_solution` | Lệnh `python` trong PowerShell trỏ vào Python 3.13 của conda, không phải `.venv` | Kích hoạt `.venv` (`conda deactivate` rồi `.venv\Scripts\Activate.ps1`) hoặc gọi thẳng `.venv\Scripts\python.exe` |
| Thư mục bản sao chạy code của repo gốc | `.venv` được chép nguyên: `_editable_impl_neorag.pth` vẫn trỏ về `D:\NeoRAG-Cup-2026\src` | Xóa `.venv` của bản sao, tạo lại và chạy `python scripts/install.py`; kiểm tra bằng `python -c "import my_solution; print(my_solution.__file__)"` |
| `SyntaxError: from __future__ imports must occur at the beginning of the file` | Docstring bị dán hai lần; chuỗi thứ hai thành câu lệnh đứng trước `from __future__` | Chỉ giữ một docstring đầu file |
| Reranker lỗi hết RAM (`DefaultCPUAllocator: not enough memory`) | PyTorch bản CPU: model 2.3 GB (fp32) nằm trong RAM; máy 11.7 GB nhưng còn trống 0.4 GB | Giảm `batch_size` 16 → 4, tắt bớt ứng dụng (WSL, Discord, VS Code...). Cache điểm cho phép chạy tiếp từ chỗ dừng |
| Rerank chậm: 27 giây mỗi câu | Model chạy trên CPU, một phần bị đẩy ra pagefile | Cài PyTorch bản CUDA: reranker từ khoảng 100 lên khoảng 2.300 cặp/phút (nhanh gấp 23 lần), khoảng 1 giây mỗi câu hỏi |
| `scripts/install.py --local` cài nhầm bản CPU dù máy có GPU | Driver NVIDIA mới (610.74) in `CUDA UMD Version: 13.3`, trong khi script tìm chuỗi `CUDA Version:` nên kết luận không có GPU | Cài trực tiếp: `pip install --no-cache-dir "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130` (bản cu130 mới nhất có cho Python 3.14). Không sửa script của BTC |

---

## 9. Bài học rút ra

1. **Đọc bộ chấm trước khi tối ưu.** Các quyết định hiệu quả nhất (nộp đủ 10 id, không nói thừa, câu từ chối ngắn) đến từ việc đọc `judge.py` và `evaluate.py`, không phải từ thử sai.
2. **Hiểu dữ liệu test.** Tỷ lệ câu không dấu ở Public Test cao hơn hẳn train; nhóm yếu nhất trên train cũng chính là nhóm chiếm nửa Public Test.
3. **Retrieval là đòn bẩy lớn nhất.** Nó chiếm 50% điểm và còn kéo theo String Presence, Noise Sensitivity.
4. **Ý tưởng "hiển nhiên" có thể làm tệ đi.** Club boost nghe hợp lý nhưng giảm 2 điểm; luôn đo trước khi giữ lại.
5. **Gộp tín hiệu thay vì thay thế.** RRF giữa BM25 và dense, giữa retriever và reranker đều tốt hơn dùng riêng thành phần mạnh nhất.
6. **Không chọn tham số theo chênh lệch nhỏ trên tập nhỏ.** 1 điểm trên 122 câu chưa đủ để tin.
7. **Thiết kế để không chết giữa chừng.** Viết lại câu hỏi lỗi thì quay về câu gốc; JSON hỏng thì quay về câu gốc; câu trả lời rỗng thì có câu dự phòng.
8. **Mỗi lần nộp chỉ đổi một thứ** để quy được mức tăng điểm về đúng nguyên nhân.
9. **Test bảo vệ các hành vi đã thiết kế có chủ đích** (36 unit test của giải pháp, 78 test tính cả test của BTC, chạy offline, LLM/embedding/reranker/doc2query đều là bản giả). Viết test cũng từng phát hiện chính ví dụ sai của người viết test.
10. **Mẫu nhỏ trên máy chỉ để bắt lỗi lớn.** Prompt v2 giảm 3 điểm String Presence trên mẫu 30 câu train nhưng tăng 1.9 điểm trên Public Test; quyết định cuối cùng dựa trên judge thật.
11. **Điểm train có thể che mất lỗi của test.** Thay đổi ở bước 8 chỉ +0.7 trên train nhưng +6.57 retrieval trên Public Test. Đọc kỹ các câu test bị sai (để chẩn đoán, không để chọn tham số) tìm ra dạng lỗi mà train không có.
12. **Kiểm chứng giả thuyết bằng một biến thể nhỏ trước khi bỏ ý tưởng.** A3 lúc đầu tăng tổng nhưng làm hại câu nhiều ý; đổi cách tích hợp (bảng riêng thay vì nối vào đoạn) giữ được phần lợi mà bỏ được phần hại.
13. **Đổi hai thứ một lúc thì khi điểm giảm sẽ không biết lỗi ở đâu.** Lần 5 gộp A1 và A3 để tiết kiệm lượt nộp; điểm giảm 0.34 và phải tốn thêm một lượt nộp để tách riêng. Ngoại lệ của bài học 8 không đáng.
14. **Tăng trên train chưa chắc tăng trên test, nhất là với kỹ thuật sinh dữ liệu bằng LLM.** A3 +0.7 trên train nhưng −0.97 retrieval trên Public Test. Kỹ thuật càng "học theo" phong cách của đoạn văn thì càng dễ khớp với train (câu hỏi viết từ đoạn văn) hơn là với người hỏi thật.
15. **Kỹ thuật không có tham số chỉnh theo train thì dự báo được test.** A4 (số ứng viên 15 và trọng số 1 : 1 đặt trước) tăng +4.0 trên train và +4.07 trên Public Test, trái với A3. Một thay đổi nhắm vào lỗi xếp hạng (cross-encoder khớp bề mặt) bằng cách để LLM đọc và so sánh các đoạn không "học theo" văn phong tài liệu.
16. **Thay đổi retrieval lớn có thể làm generation đi lùi.** Retrieval +4.07 nhưng Noise −5.55 và Groundedness −2.57; câu trả lời đổi gần hết (3/42 giống lần trước) và một số câu từng từ chối nay trả lời, nên cần theo dõi cả hai nửa điểm sau mỗi lần nộp, không chỉ retrieval.
17. **Đo độ nhiễu của judge bằng cách nộp lại đúng file cũ.** Cùng file lần 7 cho 87.22 và 87.34 (+0.12 raw): judge nhiễu rất nhỏ so với dao động của chính câu trả lời khi sinh lại (Noise đổi 4–6 điểm giữa các lần). Khi một metric generation đổi vài điểm, nghi ngờ trước hết là bộ câu trả lời, không phải judge.
18. **Đặt điều kiện dừng trước khi đo giúp bỏ ý tưởng nhanh và rẻ.** Step-Back không qua cửa train (93.2 → 92.6) sau khoảng $0.01, nên không tốn lượt nộp nào. Khi một thay đổi trước đó (A4) đã sửa đúng lỗi mà ý tưởng mới nhắm tới, phải hỏi lại phần lỗi còn lại là gì trước khi làm; ở đây đó là việc đưa đoạn vào ứng viên, mà train gần như không còn thiếu đoạn nào.
19. **Cấm một hành vi sai có thể làm mất luôn phần đúng đi kèm.** Prompt v3 (cấm nhắc phần phụ không có thông tin) làm model rút gọn câu trả lời thành một câu từ chối trơn, mất key fact và ROUGE-L (String Presence −3.9). Chẩn đoán đúng nguyên nhân Noise vẫn chưa đủ để sửa bằng một quy tắc cấm; mỗi thay đổi prompt phải đo cả mặt được lẫn mặt mất trước khi nộp.
20. **Chỉ số nhỏ mẫu có thể lấn át mức tăng của cả nửa điểm.** Lần 8 tăng 8/9 chỉ số, retrieval +1.83, nhưng một câu Abstention (1/3 câu không có đáp án) làm raw −2.78 và đưa tổng xuống dưới lần 7. Khi có chỉ số chỉ chấm trên vài câu, so các lần nộp phải tách riêng chỉ số đó; không kết luận "A5 tệ" từ raw.
21. **Kiểm tra giả thuyết trên train phải đúng dạng câu hỏi bị nghi.** 10 câu train không có đáp án đều hỏi thẳng một chi tiết vắng mặt, còn câu bị lật ở Public Test là câu gián tiếp, tài liệu trả lời được một phần. Kết quả 0/10 chỉ loại trừ được rủi ro lớn.
22. **Lời gọi LLM nặng cần timeout ngắn và đường lùi.** A5 gửi cả corpus mỗi câu; 4/42 câu vượt 60 giây. Timeout riêng 20 giây không thử lại, quá hạn thì dùng kết quả bước trước, giữ thời gian mỗi câu có giới hạn.
23. **Chạy lại cùng cấu hình mới là phép đo nhiễu đúng nghĩa.** Lần 8 (86.04) và lần 9 (90.14) cùng code, khác 4.10 raw: retrieval ±2 (A4/A5 là LLM, không tất định khi cache trống), Abstention 66.67 so với 100. Một lần nộp là một mẫu; chênh dưới vài điểm giữa hai cấu hình không đủ để kết luận cấu hình nào tốt hơn.
24. **Giả thuyết về một câu cụ thể khi không có log là suy đoán, kể cả khi nghe hợp lý.** Mình quy Abstention lần 8 cho `public-0003`, rồi lần 9 câu này vẫn "Có" mà Abstention là 100. Tách rõ "quan sát" và "suy đoán" trong nhật ký giúp rút lại giả thuyết mà không phải sửa kết luận.
25. **Mô hình tinh chỉnh cho tiếng Việt không đảm bảo tốt hơn trên miền của mình.** Vietnamese_Embedding_v2 thấp hơn bge-m3 6.0 điểm ở dense riêng và 0.9 ở full trên train. Đo ở tầng thấp nhất (dense riêng) trước và đặt cửa ở tầng cuối (full) cho thấy ngay hướng đi không đáng, với chi phí $0 và không tốn lượt nộp.

---

## 10. Việc còn lại

- [x] Ghi kết quả bước 7 (mẫu train và Public Test lần 3) vào mục 3 và mục 4.
- [x] Sửa lỗi gộp làm loãng tín hiệu của câu hỏi con (mục 6.5), đo trên train: 91.4 → 92.1.
- [x] Nộp lần 4 để đo trên Public Test: **85.20** (retrieval 87.83).
- [x] Ghi mỗi lần nộp vào mục 6 theo mẫu 6.7 (đến lần 9).
- [x] A1 (tinh chỉnh lại `n_candidates`, tỉ lệ BM25 : dense) và A2 (phủ ý trong top-5): train 92.1 → 93.2, chỉ áp dụng `n_candidates=30`.
- [x] **A3 – doc2query** (bước 10): train 93.2 → 93.9, dùng bảng BM25 riêng.
- [x] Nộp lần 5 (gồm A1 và A3): **84.86**, retrieval 86.97, thấp hơn lần 4 (mục 6.5c).
- [x] Nộp lần 6 = chỉ A1 (tắt A3): **85.55**, retrieval 87.94, bản tốt nhất. A3 hại trên Public Test (−0.97 retrieval) → mặc định `doc2query=False` (mục 6.5d).
- [x] Nghiên cứu các hướng tối ưu tiếp theo, ghi ưu nhược điểm (bước 11): A4–A7, B1–B2, Step-Back.
- [x] Thực hiện A4 (LLM xếp hạng lại listwise) và nộp lần 7: **87.22**, retrieval 92.01, bản tốt nhất; bật mặc định (mục 6.5e).
- [x] Nộp lại đúng file lần 7: **87.34** (judge dao động cỡ 0.1–0.3 raw); trang thi không có log từng câu nên chẩn đoán Noise −5.55 / Groundedness −2.57 trên máy, không qua judge (mục 6.5e).
- [x] Step-Back: đo train không đạt cửa (93.2 → 92.6 với trọng số 1.0, 92.9 với 0.5), đã tắt (mục Bước 11).
- [x] Sửa quy tắc 8 của prompt (v3): mẫu 30 câu train không qua cửa (String Presence 90.2 → 86.3), đã bỏ, giữ v2 (mục Bước 11).
- [x] A5 (LLM đọc toàn bộ corpus): train 97.2 → 99.2, nộp lần 8: **86.04** (retrieval 93.84 cao nhất, Abstention 66.67 kéo raw xuống); kiểm tra Abstention trên train không thấy khác biệt; thêm timeout 20 giây; bật mặc định (mục 6.5f).
- [x] Chạy thử cache trống với cấu hình cuối (A4 + A5): **90.14**, retrieval 95.81, Abstention 100; độ dao động giữa các lần chạy khoảng vài điểm raw (mục 6.5g). Đã đo độ dao động sinh lại, **dừng tối ưu điểm**, khóa A4 + A5.
- [x] Đo ảnh hưởng của bigram trên BM25 (train, 122 câu): unigram + bigram 78.7, chỉ unigram 74.0, thêm trigram 77.1, không mở rộng câu hỏi 74.6. Giữ bigram.
- [x] `artifacts/doc2query.jsonl` (file rỗng, A3 tắt mặc định) đã bỏ khỏi repo; bật A3 thì `enrich.py` sinh lại (khoảng $0.05).
- [x] Cài PyTorch bản CUDA để rerank chạy trên GPU (tốc độ cho chatbot và ngày thi).
- [x] v10: thay embedding bge-m3 bằng AITeamVN/Vietnamese_Embedding_v2 (chạy trên máy): train full 93.2 → 92.3, không qua cửa, giữ bge-m3 (Bước 12).
- [x] Chatbot Streamlit (`app/chat.py`): lịch sử, bản viết lại có tô dấu, câu hỏi con, thời gian từng bước, nguồn; lõi ghi vết ở `trace.py` không sửa pipeline thi, có test so sánh với `run()`.
- [x] README giải pháp (`docs/solution/README.md`): kiến trúc, bảng các lần nộp, cách chạy, danh sách việc ngày thi, giới hạn đã biết. Phát hiện khi viết: `neorag run` mặc định chạy baseline của BTC, phải thêm `--pipeline my_solution.pipeline:MyPipeline` (đã ghi đậm ở README); thêm `scripts/check_submission.py` kiểm tra file nộp.
- [x] Chốt bản cuối trước 08/11: clone sạch từ git, chạy 132 test với mã trong clone đều qua, nạp pipeline mặc định (A5 bọc A4) thành công. Từ đây không sửa code, chỉ chạy `neorag predict private_test --pipeline my_solution.pipeline:MyPipeline` (README giải pháp).
