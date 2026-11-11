"""Các bộ truy xuất (retriever): nhận câu hỏi, trả về danh sách đoạn văn xếp theo độ liên quan.

    BM25Retriever     khớp từ khóa (bỏ dấu, giải nghĩa viết tắt, bigram). Chạy trên máy, miễn phí.
    DenseRetriever    khớp ngữ nghĩa bằng embedding bge-m3 qua gateway (có cache).
    HybridRetriever   gộp hai bảng xếp hạng trên bằng Reciprocal Rank Fusion (rrf).
    MultiQueryRetriever  (bước 5) viết lại câu hỏi bằng LLM, chạy hybrid cho từng câu rồi gộp.

Mọi retriever có cùng hai phương thức:
    index(corpus)              chuẩn bị chỉ mục từ corpus được truyền vào
    search(query, top_n=10)    trả về [(doc_id, score), ...] giảm dần theo score
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict

import numpy as np

from my_solution.enrich import DocEnricher
from my_solution.query import QueryPlan, QueryRewriter
from my_solution.stepback import StepBack
from my_solution.text import expand_query, tokenize
from neorag.data import Document
from neorag.embeddings import LocalEmbedder
from neorag.llm import LLMClient


def doc_text(doc: Document) -> str:
    """Văn bản đại diện cho một đoạn: "{club}. {topic}. {text}".

    Nhiều đoạn chỉ viết "CLB" hoặc "câu lạc bộ" trong thân bài mà không nhắc tên. Ghép tên CLB
    và chủ đề vào đầu thì đoạn nào cũng chứa tên chính thức, nên câu hỏi có nhắc CLB
    (đã được expand_query thêm tên chính thức) khớp được đúng đoạn của CLB đó.
    """
    return f"{doc.club}. {doc.topic}. {doc.text}"


# ----------------------------------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------------------------------


class BM25:
    """Okapi BM25, tự cài để thấy rõ từng thành phần của công thức.

    score(q, d) = sum_{t in q} IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))

    - tf(t, d): số lần token t xuất hiện trong đoạn d.
    - IDF(t):   token càng hiếm (xuất hiện ở ít đoạn) càng có trọng số cao.
                "clb" có ở gần mọi đoạn nên IDF thấp; "tiktok" chỉ ở 1-2 đoạn nên IDF cao.
    - k1:       mức bão hòa của tf. Lặp từ lần thứ 5 không còn cộng thêm nhiều như lần thứ 1.
    - b:        mức phạt đoạn dài. Đoạn dài tự nhiên chứa nhiều từ hơn nên phải chia bớt.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs_tokens: list[list[str]]) -> BM25:
        self.tfs = [Counter(tokens) for tokens in docs_tokens]  # tf của từng đoạn
        self.lengths = [len(tokens) for tokens in docs_tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)

        n = len(docs_tokens)
        df = Counter(t for tokens in docs_tokens for t in set(tokens))  # số đoạn chứa token t
        # Biến thể IDF của Lucene: cộng 1 trong log để IDF luôn dương, kể cả token có ở mọi đoạn.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Điểm BM25 của câu hỏi với TỪNG đoạn, theo thứ tự đoạn lúc fit."""
        out = []
        for tf, length in zip(self.tfs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl)
            s = 0.0
            for t in query_tokens:  # token lặp trong câu hỏi được cộng nhiều lần (nhấn mạnh)
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class BM25Retriever:
    """Truy xuất theo từ khóa: tokenize (bỏ dấu, giải nghĩa viết tắt, bigram) + BM25.

    expand=True: câu hỏi đi qua expand_query() trước khi tokenize (thêm bản đã giải nghĩa và
    tên chính thức của CLB). Đặt expand=False để đo xem phần mở rộng giúp được bao nhiêu.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, expand: bool = True):
        self.bm25 = BM25(k1=k1, b=b)
        self.expand = expand

    def index(
        self, corpus: list[Document], extra: dict[int, str] | None = None, only_extra: bool = False
    ) -> None:
        """extra: văn bản bổ sung cho từng đoạn (A3 – câu hỏi và từ khóa do LLM sinh).

        only_extra=True: chỉ đánh chỉ mục phần bổ sung, không gồm văn bản gốc (bảng BM25 riêng).
        """
        extra = extra or {}
        self.ids = [doc.id for doc in corpus]
        texts = [
            extra.get(doc.id, "") if only_extra else f"{doc_text(doc)} {extra.get(doc.id, '')}"
            for doc in corpus
        ]
        self.bm25.fit([tokenize(t) for t in texts])

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        text = expand_query(query) if self.expand else query
        scores = self.bm25.scores(tokenize(text))
        # Sắp theo điểm giảm dần; hòa điểm thì id nhỏ đứng trước để kết quả luôn ổn định.
        ranked = sorted(zip(self.ids, scores, strict=True), key=lambda x: (-x[1], x[0]))
        return ranked[:top_n]


# ----------------------------------------------------------------------------------------------
# Dense (embedding)
# ----------------------------------------------------------------------------------------------


class DenseRetriever:
    """Truy xuất theo ngữ nghĩa: embedding bge-m3 qua gateway, so bằng cosine similarity.

    BM25 chỉ khớp được từ giống nhau. Embedding hiểu được "phụ trách mảng đào tạo" gần nghĩa
    với "trưởng ban chuyên môn", nên bù được đúng điểm yếu của BM25.

    - Mọi lời gọi embedding đi qua LLMClient có cache SQLite: chạy lại không tốn thêm tiền.
    - Vector được chuẩn hóa L2 nên tích vô hướng chính là cosine similarity.
    - expand=True: embed expand_query(query) (có thêm tên chính thức của CLB) thay vì câu gốc.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        model: str | None = None,
        expand: bool = True,
        local_model: str | None = None,
    ):
        """local_model: tên model Hugging Face chạy trên máy (ví dụ AITeamVN/Vietnamese_Embedding_v2)
        thay cho bge-m3 qua gateway; khi đó không tốn request/ngân sách gateway."""
        self.llm = llm or LLMClient(cache_path=".cache/embeddings.sqlite")
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.expand = expand
        self.local = LocalEmbedder(local_model) if local_model else None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.local is not None:
            return self.local.embed(texts)  # đã chuẩn hóa L2
        vectors = np.asarray(self.llm.embed(texts, model=self.model), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def index(self, corpus: list[Document], extra: dict[int, list[str]] | None = None) -> None:
        """extra: các câu bổ sung cho từng đoạn (A3 – câu hỏi do LLM sinh).

        Mỗi câu bổ sung được embed thành một vector RIÊNG gắn với đoạn của nó (multi-vector).
        Điểm của đoạn = max(cosine với đoạn, cosine với từng câu bổ sung), nên một câu hỏi khớp
        với chỉ một chi tiết phụ của đoạn vẫn kéo được đoạn đó lên.
        """
        self.ids = [doc.id for doc in corpus]
        self.matrix = self._embed([doc_text(doc) for doc in corpus])  # (số đoạn, 1024)
        pairs = [(i, t) for i, doc in enumerate(corpus) for t in (extra or {}).get(doc.id, [])]
        self.extra_owner = np.array([i for i, _ in pairs], dtype=int)
        self.extra_matrix = self._embed([t for _, t in pairs]) if pairs else None

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        text = expand_query(query) if self.expand else query
        q = self._embed([text])[0]
        sims = self.matrix @ q  # cosine với từng đoạn
        if self.extra_matrix is not None:
            np.maximum.at(sims, self.extra_owner, self.extra_matrix @ q)
        order = np.argsort(-sims, kind="stable")[:top_n]
        return [(self.ids[i], float(sims[i])) for i in order]


# ----------------------------------------------------------------------------------------------
# Hybrid: Reciprocal Rank Fusion
# ----------------------------------------------------------------------------------------------


def rrf(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: gộp nhiều bảng xếp hạng chỉ dựa vào THỨ HẠNG.

        rrf(d) = sum_i  weight_i / (k + rank_i(d))        rank bắt đầu từ 1

    Điểm BM25 (0..30) và cosine (0..1) khác thang đo nên không cộng thẳng được. RRF bỏ qua
    điểm, chỉ dùng thứ hạng. k=60 làm phẳng chênh lệch giữa các hạng đầu: một đoạn đứng
    hạng 3 ở CẢ HAI bảng sẽ thắng một đoạn hạng 1 ở bảng này nhưng không có ở bảng kia.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    """BM25 + Dense, gộp bằng RRF.

    depth: mỗi retriever lấy bao nhiêu đoạn để đưa vào RRF. Lấy sâu hơn top_n để một đoạn
    đứng hạng 15 ở bảng này nhưng hạng 2 ở bảng kia vẫn có cơ hội được kéo lên.

    weights=(1.5, 1.0): BM25 nặng hơn vì bge-m3 hiểu kém câu không dấu/teencode (nhóm
    "không dấu" trên train: BM25 78.8, dense 1:1 kéo hybrid xuống 63.2). Khi query rewriting
    (bước 5) khôi phục dấu cho câu hỏi, nên đo lại cặp trọng số này.
    """

    @classmethod
    def with_doc2query(cls, enricher: DocEnricher | None = None) -> HybridRetriever:
        """Cấu hình A3 đã chọn trên train: phần làm giàu thành bảng BM25 riêng, trọng số 1.0.

        Nối thẳng vào văn bản của đoạn cũng +0.7 nhưng làm câu có câu con giảm 1.1 (câu hỏi sinh
        ra lặp tên CLB, làm mờ khác biệt giữa các đoạn cùng CLB); bảng riêng giữ nguyên nhóm đó.
        Làm giàu cho dense không giúp gì nên tắt.
        """
        return cls(enricher=enricher or DocEnricher(), enrich_dense=False, enrich_separate=1.0)

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        weights: tuple[float, float] = (1.5, 1.0),  # (bm25, dense)
        depth: int = 30,
        k: int = 60,
        enricher: DocEnricher | None = None,
        enrich_bm25: bool = True,
        enrich_dense: bool = True,
        enrich_separate: float = 0.0,
    ):
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.weights = list(weights)
        self.depth = depth
        self.k = k
        self.enricher = enricher  # A3: None = không làm giàu tài liệu
        self.enrich_bm25 = enrich_bm25
        self.enrich_dense = enrich_dense
        # enrich_separate = w > 0: phần làm giàu thành một bảng BM25 RIÊNG (trọng số w trong RRF)
        # thay vì nối vào văn bản của đoạn; khi đó enrich_bm25 bị bỏ qua.
        self.enrich_separate = enrich_separate
        self.bm25_extra: BM25Retriever | None = None

    def index(self, corpus: list[Document]) -> None:
        if self.enricher is None:
            self.bm25.index(corpus)
            self.dense.index(corpus)
            return
        enriched = self.enricher.enrich(corpus)
        extra_text = {i: e.text for i, e in enriched.items()}
        if self.enrich_separate:
            self.bm25.index(corpus)
            self.bm25_extra = BM25Retriever(expand=self.bm25.expand)
            self.bm25_extra.index(corpus, extra_text, only_extra=True)
        else:
            self.bm25.index(corpus, extra_text if self.enrich_bm25 else None)
        self.dense.index(
            corpus, {i: e.questions for i, e in enriched.items()} if self.enrich_dense else None
        )

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        retrievers = [self.bm25, self.dense]
        weights = list(self.weights)
        if self.bm25_extra is not None:
            retrievers.append(self.bm25_extra)
            weights.append(self.enrich_separate)
        rankings = [
            [doc_id for doc_id, _ in retriever.search(query, top_n=self.depth)]
            for retriever in retrievers
        ]
        return rrf(rankings, weights, self.k)[:top_n]


# ----------------------------------------------------------------------------------------------
# Bước 5: nhiều câu truy vấn từ một câu hỏi
# ----------------------------------------------------------------------------------------------


class MultiQueryRetriever:
    """Chạy hybrid cho TỪNG câu trong QueryPlan (câu gốc, bản viết lại, các câu con), gộp bằng RRF.

    Vì sao cách này sửa được câu "Trong 3 CLB thuộc S2B, CLB nào lâu đời nhất?":
    mỗi câu con ("CLB Multimedia PTIT thành lập năm nào?") đưa đoạn đúng của CLB đó lên hạng 1
    trong bảng xếp hạng riêng. RRF cộng điểm theo thứ hạng, nên hạng 1 của mỗi bảng con đều
    được điểm cao, và top-5 cuối cùng có đủ đoạn của từng CLB thay vì toàn đoạn chung S2B.

    Vẫn giữ câu gốc trong danh sách: nếu LLM viết lại sai ý thì câu gốc vẫn kéo đúng đoạn lên.
    rewriter=None: tắt bước viết lại (không gọi LLM), dùng để so sánh hoặc khi hết ngân sách.

    reserve=k: GIỮ CHỖ cho top-k của mỗi câu hỏi con trong kết quả trả về.
        Lỗi tìm thấy sau lần nộp 3 (nhật ký mục 6.5): một câu hỏi được tách thành 4-6 câu truy
        xuất; đoạn đúng đứng hạng 1-2 trong bảng của câu con tương ứng, nhưng RRF cộng dồn điểm
        nên các đoạn chung chung (đứng vừa phải ở NHIỀU bảng) vượt lên, đẩy đoạn đúng xuống hạng
        16-20 hoặc ra ngoài danh sách ứng viên. Giữ chỗ bảo đảm đoạn đó vẫn đến được reranker.
        Trên train: không đổi điểm (lỗi này hầu như không xảy ra ở train); giữ reserve=2 vì nó
        nhắm đúng lỗi quan sát được ở Public Test mà không làm hại train.
    """

    def __init__(
        self,
        base: HybridRetriever | None = None,
        rewriter: QueryRewriter | None | bool = True,
        depth: int = 30,
        k: int = 60,
        reserve: int = 2,
        stepback: StepBack | None | bool = None,
        stepback_weight: float = 1.0,
    ):
        self.base = base or HybridRetriever()
        self.rewriter = QueryRewriter() if rewriter is True else (rewriter or None)
        self.depth = depth
        self.k = k
        self.reserve = reserve
        # Step-Back: câu khái quát thành thêm một bảng xếp hạng (trọng số stepback_weight) trong
        # RRF; chỉ cho câu KHÔNG có câu con. None = tắt (không gọi LLM).
        self.stepback = StepBack() if stepback is True else (stepback or None)
        self.stepback_weight = stepback_weight

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)

    def plan(self, query: str) -> QueryPlan:
        plan = QueryPlan(query, query) if self.rewriter is None else self.rewriter.plan(query)
        if self.stepback is not None and not plan.subqueries:
            plan.stepback = self.stepback.generate(plan)
        return plan

    def search_plan(self, plan: QueryPlan, top_n: int = 10) -> list[tuple[int, float]]:
        rankings = {
            q: [doc_id for doc_id, _ in self.base.search(q, top_n=self.depth)] for q in plan.queries
        }
        weights = [self.stepback_weight if q == plan.stepback else 1.0 for q in rankings]
        fused = rrf(list(rankings.values()), weights, k=self.k)
        head = fused[:top_n]
        if not (self.reserve and plan.subqueries):
            return head

        # Đoạn phải có mặt: top-`reserve` của mỗi câu con, lấy xen kẽ theo hạng (hạng 1 của mọi
        # câu con trước, rồi hạng 2...) để câu con nào cũng có phần khi không đủ chỗ.
        sub_rankings = [rankings[q] for q in plan.subqueries if q in rankings]
        must = list(
            dict.fromkeys(r[i] for i in range(self.reserve) for r in sub_rankings if i < len(r))
        )
        in_head = {d for d, _ in head}
        missing = [d for d in must if d not in in_head][:top_n]
        if not missing:
            return head
        # Thay phần cuối của danh sách bằng các đoạn còn thiếu; phần đầu giữ nguyên thứ tự RRF.
        score = dict(fused)
        return head[: top_n - len(missing)] + [(d, score[d]) for d in missing]

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        return self.search_plan(self.plan(query), top_n)
