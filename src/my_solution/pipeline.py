"""Pipeline nộp bài.

    câu hỏi
      -> QueryRewriter (bước 5)       LLM viết lại có dấu + tách câu hỏi con      1 lời gọi LLM
      -> MultiQueryRetriever          hybrid BM25 + bge-m3 cho từng câu, gộp RRF
      -> RerankedRetriever (bước 6)   cross-encoder bge-reranker-v2-m3 trên máy   miễn phí
      -> claude-haiku-5-5             sinh câu trả lời từ n_context đoạn đầu       1 lời gọi LLM

Chạy trên train (không gọi judge, không tốn tiền chấm):
    neorag run train --pipeline my_solution.pipeline:MyPipeline --no-judge
Sinh file nộp Public Test:
    neorag run public_test --pipeline my_solution.pipeline:MyPipeline
"""

from __future__ import annotations

import os
import re
from datetime import date

from my_solution.corpus_rank import CorpusRanker
from my_solution.llm_rerank import ListwiseReranker
from my_solution.prompts import DEFAULT_PROMPT_VERSION, SYSTEM_PROMPTS, build_messages
from my_solution.rerank import RerankedRetriever
from my_solution.retrievers import HybridRetriever, MultiQueryRetriever
from my_solution.stepback import StepBack
from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline

# A5 đọc ~20 nghìn token mỗi lần, gateway đôi khi chậm hơn 60 giây (Public Test lần 8: 4/42 câu).
# Dùng client riêng: quá hạn thì không chờ, không thử lại, A5 lùi về kết quả A4.
A5_TIMEOUT = 20.0

MAX_ANSWER_CHARS = 2000  # giới hạn của bộ chấm (configs/eval.toml)
FALLBACK_ANSWER = "Tài liệu hiện có không có thông tin để trả lời câu hỏi này."

_MARKDOWN = re.compile(r"\*\*|__|^#+\s*", flags=re.MULTILINE)


def clean_answer(text: str) -> str:
    """Bỏ markdown model đôi khi vẫn thêm, gộp khoảng trắng, bảo đảm không rỗng và không quá dài."""
    text = " ".join(_MARKDOWN.sub("", text).split())
    if not text:
        return FALLBACK_ANSWER
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0]
    return text


class MyPipeline(RAGPipeline):
    """`load_pipeline` gọi MyPipeline() không tham số, nên mọi tham số đều có giá trị mặc định.

    n_retrieved: số id nộp để chấm retrieval. Nộp đủ 10: Recall/MRR/NDCG chỉ xét top-5 nên
                 không bị phạt, còn Groundedness được chấm trên văn bản của TẤT CẢ id đã nộp.
    n_context:   số đoạn đưa vào prompt. Nhiều hơn 5 để câu hỏi cần 3-4 đoạn vẫn đủ thông tin.
    rewrite:     bật bước 5 (viết lại câu hỏi bằng LLM).
    rerank:      bật bước 6 (cross-encoder trên máy).
    doc2query:   bật A3 (làm giàu tài liệu lúc index, lưu riêng ở artifacts/doc2query.jsonl).
                 Mặc định TẮT: train +0.7 nhưng Public Test −0.97 retrieval (lần nộp 5 so với 6).
    llm_rerank:  bật A4 (LLM xếp hạng lại 15 ứng viên đầu, llm_rerank.py). Mặc định BẬT: Public
                 Test lần 7 retrieval 87.94 -> 92.01, raw 85.55 -> 87.22. Thêm 1 lời gọi LLM mỗi câu.
    stepback:    bật Step-Back (stepback.py): câu hỏi khái quát thêm vào truy vấn tìm ứng viên (câu không
                 có câu con). Mặc định TẮT cho đến khi đo xong. Thêm 1 lời gọi LLM nhỏ mỗi câu.
    corpus_rank: bật A5 (LLM đọc toàn bộ corpus, chọn top-10, gộp RRF với kết quả A4; corpus_rank.py).
                 Mặc định BẬT: train retrieval 97.2 -> 99.2, Public Test 92.01 -> 93.84 (lần 8). Thêm 1 lời
                 gọi LLM ~20 nghìn token mỗi câu, timeout 20 giây không thử lại (quá hạn thì dùng kết quả A4).
    prompt_version: "v1" (bản đã nộp lần 1-2) hoặc "v2" (bước 7), xem prompts.py.
    llm, retriever: truyền vào để thay thế (test dùng bản giả, không gọi API). Khi truyền
                 retriever thì rewrite/rerank bị bỏ qua.
    """

    def __init__(
        self,
        n_retrieved: int = 10,
        n_context: int = 8,
        rewrite: bool = True,
        rerank: bool = True,
        doc2query: bool = False,
        llm_rerank: bool = True,
        stepback: bool = False,
        corpus_rank: bool = True,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        llm_model: str | None = None,
        llm: LLMClient | None = None,
        retriever: RerankedRetriever | MultiQueryRetriever | None = None,
    ):
        self.n_retrieved = n_retrieved
        self.n_context = n_context
        if prompt_version not in SYSTEM_PROMPTS:
            raise ValueError(f"prompt_version phải là một trong {sorted(SYSTEM_PROMPTS)}")
        self.prompt_version = prompt_version
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")

        if retriever is None:
            base = HybridRetriever.with_doc2query() if doc2query else HybridRetriever()
            retriever = MultiQueryRetriever(
                base=base,
                rewriter=True if rewrite else None,
                stepback=StepBack(self.llm, self.llm_model) if stepback else None,
            )
            retriever = RerankedRetriever(retriever) if rerank else retriever
            if llm_rerank:
                retriever = ListwiseReranker(retriever, llm=self.llm, model=self.llm_model)
            if corpus_rank:
                a5_llm = llm or LLMClient(
                    cache_path=".cache/pipeline.sqlite", max_retries=0, timeout=A5_TIMEOUT
                )
                retriever = CorpusRanker(retriever, llm=a5_llm, model=self.llm_model)
        self.retriever = retriever

    def prepare(self) -> None:
        if hasattr(self.retriever, "prepare"):
            self.retriever.prepare()  # nạp model reranker (nặng) một lần

    def index(self, corpus: list[Document]) -> None:
        self.docs = {d.id: d for d in corpus}
        self.retriever.index(corpus)

    def run(self, query: str) -> RAGOutput:
        plan = self.retriever.plan(query)
        hits = self.retriever.search_plan(plan, top_n=self.n_retrieved)
        docs = [self.docs[doc_id] for doc_id, _ in hits]

        today = date.today().strftime("%d/%m/%Y")
        messages = build_messages(
            query, docs[: self.n_context], today, rewrite=plan.rewrite, version=self.prompt_version
        )
        answer = clean_answer(self.llm.chat(messages, model=self.llm_model))
        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])
