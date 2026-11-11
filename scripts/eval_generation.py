"""Chạy MyPipeline trên MỘT MẪU câu hỏi train để thử nghiệm rẻ, kèm ước tính chi phí trước khi chạy.

`neorag run train` luôn chạy đủ 126 câu. Khi thử prompt (bước 7), mỗi biến thể chỉ cần một mẫu
nhỏ, chia đều theo loại câu hỏi, để so sánh. Mẫu được chọn với seed cố định, nên chạy lại
cùng mẫu sẽ dùng cache, không tốn thêm tiền.

Mặc định CHỈ ƯỚC TÍNH (không gọi API). Thêm --yes để chạy thật:

    python scripts/eval_generation.py --sample 30                    # ước tính chi phí
    python scripts/eval_generation.py --sample 30 --yes              # chạy, ghi outputs/gen_sample30.jsonl
    python scripts/eval_generation.py --type "không dấu" --yes       # chỉ một loại câu
    python scripts/eval_generation.py --sample 30 --no-rerank --name norerank --yes

Kết quả được phân tích ngay bằng analyze_predictions.py; so sánh hai lần chạy:
    python scripts/analyze_predictions.py --predictions outputs/gen_a.jsonl --baseline outputs/gen_b.jsonl
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from analyze_predictions import group, load_rows, print_table  # cùng thư mục scripts/

from my_solution.pipeline import MyPipeline
from my_solution.prompts import DEFAULT_PROMPT_VERSION, FEW_SHOT_IDS, SYSTEM_PROMPTS
from my_solution.query import REWRITE_PROMPT
from my_solution.retrievers import doc_text
from neorag.data import DATA_DIR, Query, load_corpus, read_jsonl, save_predictions
from neorag.llm import _key  # hàm tạo khóa cache của LLMClient (để biết lời gọi nào đã có cache)
from neorag.predict import predict

# Giá niêm yết của claude-haiku-5-5 (USD / 1 triệu token). Gateway có thể tính khác:
# đối chiếu số dư ở https://proptit.dekthe.dev/usage/ trước và sau khi chạy.
PRICE_IN, PRICE_OUT = 0.10, 0.50
CHARS_PER_TOKEN = 2.5  # tiếng Việt tốn token hơn tiếng Anh; chọn số nhỏ để ước tính dư
OUTPUT_TOKENS = {"rewrite": 500, "answer": 600}  # gồm cả phần suy luận ngầm của Haiku 5.5
AVG_SUBQUERIES = 0.7  # trung bình số câu con mỗi câu hỏi, để ước tính số request embedding


def sample_questions(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    """Lấy mẫu PHÂN TẦNG: mỗi loại câu hỏi góp số câu tỉ lệ với kích thước của nó (ít nhất 1).

    Lấy ngẫu nhiên thuần có thể bốc toàn câu "trực tiếp" dễ, làm kết quả lệch. Phân tầng giữ
    tỉ lệ các loại câu như tập đầy đủ. Thứ tự đầu ra theo id để mẫu ổn định giữa các lần chạy.

    Chia theo phương pháp phần dư lớn nhất: mỗi loại nhận phần nguyên của n * tỉ lệ (ít nhất 1),
    số câu còn thiếu chia cho các loại có phần thập phân lớn nhất, để tổng đúng bằng n.
    """
    if n is None or n >= len(rows):
        return rows
    by_type = defaultdict(list)
    for r in rows:
        by_type[r.get("type", "")].append(r)
    types = sorted(by_type)
    exact = {t: n * len(by_type[t]) / len(rows) for t in types}
    quota = {t: max(1, math.floor(exact[t])) for t in types}
    for t in sorted(types, key=lambda t: exact[t] - math.floor(exact[t]), reverse=True):
        if sum(quota.values()) >= n:
            break
        if quota[t] < len(by_type[t]):
            quota[t] += 1

    rng = random.Random(seed)
    picked = [r for t in types for r in rng.sample(by_type[t], min(quota[t], len(by_type[t])))]
    return sorted(picked, key=lambda r: r["id"])


def estimate(rows: list[dict], pipeline: MyPipeline, avg_doc_chars: float) -> dict:
    """Ước tính số request và chi phí. Lời gọi viết lại câu hỏi được kiểm tra cache CHÍNH XÁC;
    lời gọi sinh câu trả lời phụ thuộc kết quả truy xuất nên coi như chưa có cache (ước tính dư).
    """
    llm, model = pipeline.llm, pipeline.llm_model
    cache = llm._cache  # thuộc tính nội bộ của LLMClient: chỉ đọc, không ghi

    rewrite_calls = 0
    rewrite_in = 0.0
    if _uses_rewriter(pipeline):
        for r in rows:
            messages = [
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": r["query"]},
            ]
            if (
                cache is None
                or cache.get(_key("chat", llm._endpoint, model, None, messages)) is None
            ):
                rewrite_calls += 1
                rewrite_in += (len(REWRITE_PROMPT) + len(r["query"])) / CHARS_PER_TOKEN

    answer_calls = len(rows)
    # A4: 1 lời gọi xếp hạng mỗi câu (coi như chưa có cache: ước tính dư), đưa n_rerank đoạn vào prompt
    n_rerank = _find(pipeline.retriever, "n_rerank")
    rank_calls = len(rows) if n_rerank else 0
    rank_in = rank_calls * (n_rerank * avg_doc_chars / CHARS_PER_TOKEN + 400) if n_rerank else 0.0
    # A5: mỗi câu đưa cả corpus (118 đoạn) vào prompt (coi như chưa có cache: ước tính dư)
    corpus_calls = len(rows) if _find(pipeline.retriever, "system") is not None else 0
    corpus_in = corpus_calls * (CORPUS_DOCS * avg_doc_chars / CHARS_PER_TOKEN + 500)
    answer_in = sum(
        (
            len(SYSTEM_PROMPTS[pipeline.prompt_version])
            + pipeline.n_context * avg_doc_chars
            + 2 * len(r["query"])
        )
        / CHARS_PER_TOKEN
        for r in rows
    )
    embed_requests = round(rewrite_calls * (1 + AVG_SUBQUERIES))

    cost = (
        (rewrite_in + answer_in + rank_in + corpus_in) * PRICE_IN
        + (
            rewrite_calls * OUTPUT_TOKENS["rewrite"]
            + answer_calls * OUTPUT_TOKENS["answer"]
            + (rank_calls + corpus_calls) * 60
        )
        * PRICE_OUT
    ) / 1_000_000
    return {
        "rewrite_calls": rewrite_calls,
        "answer_calls": answer_calls,
        "rank_calls": rank_calls,
        "corpus_calls": corpus_calls,
        "embed_requests": embed_requests,
        "requests": rewrite_calls + answer_calls + rank_calls + corpus_calls + embed_requests,
        "usd": cost,
    }


CORPUS_DOCS = 118


def _find(retriever, attr):
    """Tìm thuộc tính `attr` trên retriever hoặc các lớp bọc bên trong (.base, .first_stage)."""
    while retriever is not None:
        if hasattr(retriever, attr):
            return getattr(retriever, attr)
        retriever = getattr(retriever, "base", None) or getattr(retriever, "first_stage", None)
    return None


def _first_stage(retriever):
    """Bóc các lớp bọc (A4 ListwiseReranker.base, RerankedRetriever.first_stage) tới MultiQueryRetriever."""
    while hasattr(retriever, "base") and not hasattr(retriever, "rewriter"):
        retriever = retriever.base
    return getattr(retriever, "first_stage", retriever)


def _uses_rewriter(pipeline: MyPipeline) -> bool:
    return _first_stage(pipeline.retriever).rewriter is not None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default=str(DATA_DIR / "train" / "questions.jsonl"))
    parser.add_argument("--sample", type=int, help="số câu lấy mẫu (mặc định: tất cả)")
    parser.add_argument("--type", help="chỉ lấy các câu có trường type chứa chuỗi này")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-context", type=int, default=8)
    parser.add_argument("--no-rewrite", action="store_true", help="tắt bước 5")
    parser.add_argument("--no-rerank", action="store_true", help="tắt bước 6")
    parser.add_argument(
        "--no-corpus-rank", action="store_true", help="tắt A5 (LLM đọc toàn corpus)"
    )
    parser.add_argument("--prompt", choices=sorted(SYSTEM_PROMPTS), default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--name", help="tên lần chạy, file ra là outputs/gen_<name>.jsonl")
    parser.add_argument("--yes", action="store_true", help="chạy thật (có gọi API)")
    args = parser.parse_args()

    # Bỏ các câu đã dùng làm few-shot trong prompt, để mọi phiên bản prompt được chấm trên cùng
    # một tập câu hỏi "chưa thấy".
    rows = [r for r in read_jsonl(args.questions) if r["id"] not in FEW_SHOT_IDS]
    if args.type:
        rows = [r for r in rows if args.type in r.get("type", "")]
    rows = sample_questions(rows, args.sample, args.seed)

    pipeline = MyPipeline(
        n_context=args.n_context,
        rewrite=not args.no_rewrite,
        rerank=not args.no_rerank,
        corpus_rank=not args.no_corpus_rank,
        prompt_version=args.prompt,
    )
    corpus = load_corpus()
    avg_doc_chars = sum(len(doc_text(d)) for d in corpus) / len(corpus)
    est = estimate(rows, pipeline, avg_doc_chars)

    print(f"{len(rows)} câu hỏi  (rewrite={'tắt' if args.no_rewrite else 'bật'}, "
          f"rerank={'tắt' if args.no_rerank else 'bật'}, n_context={args.n_context}, "
          f"prompt={args.prompt})")  # fmt: skip
    print(
        f"Ước tính: {est['rewrite_calls']} lời gọi viết lại (chưa có cache) + "
        f"{est['answer_calls']} lời gọi sinh câu trả lời + {est['rank_calls']} lời gọi xếp hạng A4 "
        f"+ {est['corpus_calls']} lời gọi A5 (cả corpus) + ~{est['embed_requests']} embedding "
        f"= ~{est['requests']} request, tối đa ~${est['usd']:.3f}"
    )
    if not args.yes:
        print("Chưa chạy. Thêm --yes để chạy thật.")
        return

    name = args.name or (f"sample{args.sample}" if args.sample else "all")
    out = Path("outputs") / f"gen_{name}.jsonl"
    pipeline.prepare()
    pipeline.index(corpus)
    queries = [Query(id=r["id"], query=r["query"]) for r in rows]
    save_predictions(out, predict(pipeline, queries, max_retrieved=10))
    print(f"Đã ghi {out}\n")

    golds = {r["id"]: r for r in rows}
    docs = {d.id: d for d in corpus}
    print_table(group(load_rows(out, golds, docs, args.n_context)), None)


if __name__ == "__main__":
    main()
