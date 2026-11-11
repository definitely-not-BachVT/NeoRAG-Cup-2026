"""Chấm riêng phần RETRIEVAL trên tập train, ngay trên máy, không gọi judge.

Dùng đúng hàm score_query và cấu hình (k, metric) của bộ chấm BTC, nên con số khớp với
nửa "retrieval" của điểm Benchmark.

    python scripts/eval_retrieval.py                       # BM25 có mở rộng câu hỏi
    python scripts/eval_retrieval.py --retriever bm25-raw  # BM25 không mở rộng (để so sánh)
    python scripts/eval_retrieval.py --retriever hybrid    # BM25 + bge-m3 gộp bằng RRF
    python scripts/eval_retrieval.py --retriever full      # + viết lại câu hỏi + reranker
    python scripts/eval_retrieval.py --by type             # điểm theo từng loại câu hỏi
    python scripts/eval_retrieval.py --only "không dấu"    # chỉ chấm một loại câu
    python scripts/eval_retrieval.py --fails 10            # in 10 câu tệ nhất kèm top-5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from statistics import mean

from my_solution.corpus_rank import CorpusRanker
from my_solution.llm_rerank import ListwiseReranker
from my_solution.rerank import RerankedRetriever
from my_solution.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MultiQueryRetriever,
)
from neorag.config import load_config
from neorag.data import DATA_DIR, load_corpus, read_jsonl
from neorag.metrics.retrieval import score_query

VN_EMBEDDING = "AITeamVN/Vietnamese_Embedding_v2"

# Tên retriever -> hàm tạo. Thêm cấu hình mới vào đây để so sánh.
RETRIEVERS = {
    "bm25": lambda: BM25Retriever(expand=True),
    "bm25-raw": lambda: BM25Retriever(expand=False),
    "dense": lambda: DenseRetriever(expand=True),
    "dense-raw": lambda: DenseRetriever(expand=False),
    "hybrid": lambda: HybridRetriever(),
    # Bước 5: viết lại câu hỏi bằng LLM (1 lời gọi/câu, có cache)
    "multi": lambda: MultiQueryRetriever(),
    # Bước 6: cross-encoder chạy trên máy. "rerank" không viết lại câu hỏi nên không tốn API.
    "rerank": lambda: RerankedRetriever(MultiQueryRetriever(rewriter=None)),
    "full": lambda: RerankedRetriever(MultiQueryRetriever()),
    # A3: thêm bảng BM25 trên câu hỏi/từ khóa do LLM sinh (đọc artifacts/doc2query.jsonl)
    "full-a3": lambda: RerankedRetriever(
        MultiQueryRetriever(base=HybridRetriever.with_doc2query())
    ),
    # A4: LLM xếp hạng lại 15 ứng viên đầu (thêm 1 lời gọi LLM/câu, có cache)
    "full-a4": lambda: ListwiseReranker(RerankedRetriever(MultiQueryRetriever())),
    # Step-Back: câu khái quát thêm vào truy vấn tìm ứng viên (chỉ câu không có câu con)
    "full-sb": lambda: RerankedRetriever(MultiQueryRetriever(stepback=True)),
    "full-sb05": lambda: RerankedRetriever(MultiQueryRetriever(stepback=True, stepback_weight=0.5)),
    # A5: LLM đọc toàn bộ corpus (thêm 1 lời gọi ~20 nghìn token/câu), gộp RRF với A4
    "full-a4-a5": lambda: CorpusRanker(ListwiseReranker(RerankedRetriever(MultiQueryRetriever()))),
    "full-a4-a5w2": lambda: CorpusRanker(
        ListwiseReranker(RerankedRetriever(MultiQueryRetriever())), weights=(1.0, 2.0)
    ),
    # v10: embedding chạy trên máy (AITeamVN/Vietnamese_Embedding_v2) thay bge-m3 của gateway
    "dense-vn": lambda: DenseRetriever(local_model=VN_EMBEDDING),
    "hybrid-vn": lambda: HybridRetriever(dense=DenseRetriever(local_model=VN_EMBEDDING)),
    "full-vn": lambda: RerankedRetriever(
        MultiQueryRetriever(base=HybridRetriever(dense=DenseRetriever(local_model=VN_EMBEDDING)))
    ),
    "full-vn-a4": lambda: ListwiseReranker(
        RerankedRetriever(
            MultiQueryRetriever(
                base=HybridRetriever(dense=DenseRetriever(local_model=VN_EMBEDDING))
            )
        )
    ),
    "full-a4-sb": lambda: ListwiseReranker(RerankedRetriever(MultiQueryRetriever(stepback=True))),
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # terminal Windows mặc định không in được tiếng Việt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    parser.add_argument("--questions", default=str(DATA_DIR / "train" / "questions.jsonl"))
    parser.add_argument("--by", help="nhóm kết quả theo một trường: type, club, difficulty")
    parser.add_argument("--only", help="chỉ chấm các câu có trường type chứa chuỗi này")
    parser.add_argument("--fails", type=int, default=0, help="in N câu có điểm thấp nhất")
    args = parser.parse_args()

    cfg = load_config()
    k_values = cfg["retrieval"]["k_values"]  # [3, 5, 7]
    rank_k = cfg["score"]["rank_k"]  # 5
    main_metrics = [f"{m}@{rank_k}" for m in cfg["score"]["retrieval_metrics"]]  # recall, mrr, ndcg

    corpus = load_corpus()
    docs = {d.id: d for d in corpus}
    # read_jsonl thay vì load_queries: load_queries bỏ mất trường "type" (không có trong Query).
    rows = [r for r in read_jsonl(args.questions) if r.get("relevant_ids")]  # bỏ câu không đáp án
    if args.only:
        rows = [r for r in rows if args.only in r.get("type", "")]

    retriever = RETRIEVERS[args.retriever]()
    if hasattr(retriever, "prepare"):  # nạp model reranker trước khi chấm
        retriever.prepare()
    retriever.index(corpus)

    results = []  # mỗi phần tử: (row, retrieved_ids, metrics, score)
    for row in rows:
        retrieved = [doc_id for doc_id, _ in retriever.search(row["query"], top_n=10)]
        metrics = score_query(retrieved, set(row["relevant_ids"]), k_values)
        score = mean(metrics[m] for m in main_metrics)
        results.append((row, retrieved, metrics, score))

    def report(label: str, group: list) -> None:
        avg = {m: mean(r[2][m] for r in group) * 100 for m in main_metrics}
        recall7 = mean(r[2]["recall@7"] for r in group) * 100
        cols = "  ".join(f"{m} {v:5.1f}" for m, v in avg.items())
        print(
            f"{label:<22} n={len(group):<4} SCORE {mean(avg.values()):5.1f}  {cols}  recall@7 {recall7:5.1f}"
        )

    print(f"Retriever: {args.retriever}  |  {len(results)} câu có relevant_ids\n")
    report("TẤT CẢ", results)

    if args.by:
        groups = defaultdict(list)
        for r in results:
            groups[str(r[0].get(args.by, ""))].append(r)
        print()
        for key, group in sorted(groups.items(), key=lambda kv: mean(r[3] for r in kv[1])):
            report(key or "(trống)", group)  # nhóm yếu nhất in trước

    if args.fails:
        print(f"\n{args.fails} câu có điểm thấp nhất:")
        for row, retrieved, _metrics, score in sorted(results, key=lambda r: r[3])[: args.fails]:
            relevant = set(row["relevant_ids"])
            print(
                f"\n[{row['id']}] ({row.get('type', '')}) score {score * 100:.0f}  {row['query']}"
            )
            print(f"  cần: {sorted(relevant)}")
            for rank, doc_id in enumerate(retrieved[:5], 1):
                mark = "✓" if doc_id in relevant else " "
                d = docs[doc_id]
                print(f"  {mark} {rank}. [{doc_id:>3}] {d.club} | {d.topic}")
            missed = relevant - set(retrieved[:5])
            if missed:
                print(f"  trượt: {sorted(missed)}")


if __name__ == "__main__":
    main()
