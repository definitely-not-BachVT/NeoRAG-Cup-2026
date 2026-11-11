"""Bước 6: xếp hạng lại (rerank) bằng cross-encoder chạy trên máy.

Retriever (bi-encoder, BM25) mã hóa câu hỏi và đoạn văn RIÊNG RẼ rồi so sánh, nên nhanh nhưng
thô. Cross-encoder đọc CẢ CẶP (câu hỏi, đoạn văn) cùng lúc qua transformer, nhìn được từng từ
của câu hỏi tương tác với từng từ của đoạn văn, nên chấm độ liên quan chính xác hơn nhiều,
đổi lại chậm: phải chạy model một lần cho mỗi cặp. Vì vậy chỉ rerank ~20 ứng viên đầu.

Model: BAAI/bge-reranker-v2-m3 (đa ngôn ngữ, hiểu tiếng Việt, open-source, không sinh văn bản,
đúng luật). Chạy trên máy nên KHÔNG tốn ngân sách API. Lần đầu tải về khoảng 2.3 GB.
Cần sentence-transformers: python scripts/install.py --local (cài cả PyTorch bản CUDA nếu có GPU).

Điểm được cache trong .cache/rerank.sqlite: chạy lại không phải tính lại.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from my_solution.query import QueryPlan
from my_solution.retrievers import MultiQueryRetriever, doc_text, rrf
from neorag.data import Document
from neorag.embeddings import is_downloaded


class _ScoreCache:
    """Cache điểm (model, câu hỏi, đoạn văn) -> float trong SQLite."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value REAL)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        with self._lock:
            marks = ",".join("?" * len(keys))
            rows = self._conn.execute(
                f"SELECT key, value FROM scores WHERE key IN ({marks})", keys
            ).fetchall()
        return dict(rows)

    def set_many(self, items: dict[str, float]) -> None:
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?)", items.items())
            self._conn.commit()


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 512,
        batch_size: int = 4,  # nhỏ để vừa RAM khi chạy CPU; có GPU thì tăng lên 16-32
        cache_path: str | Path | None = ".cache/rerank.sqlite",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache = _ScoreCache(cache_path) if cache_path else None
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        """Nạp model (tải về nếu chưa có). Gọi trong prepare() để không tính vào thời gian trả lời."""
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    local_files_only=is_downloaded(self.model_name),
                )
        return self._model

    def scores(self, query: str, docs: list[Document]) -> list[float]:
        """Điểm liên quan của từng đoạn với câu hỏi (càng cao càng liên quan)."""
        texts = [doc_text(d) for d in docs]
        keys = [_ScoreCache.key(self.model_name, query, t) for t in texts]
        cached = self.cache.get_many(keys) if self.cache else {}

        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            model = self.load()
            pairs = [(query, texts[i]) for i in missing]
            with self._lock:  # chatbot/bộ chấm có thể gọi từ nhiều luồng
                new = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            fresh = {keys[i]: float(s) for i, s in zip(missing, new, strict=True)}
            cached.update(fresh)
            if self.cache:
                self.cache.set_many(fresh)
        return [cached[k] for k in keys]


class RerankedRetriever:
    """MultiQueryRetriever lấy n_candidates ứng viên -> cross-encoder chấm lại -> gộp hai bảng.

    Câu dùng để chấm: bản viết lại (có dấu, tên CLB đầy đủ) và từng câu con. Mỗi đoạn lấy điểm
    CAO NHẤT trong các câu đó: với câu hỏi so sánh 3 CLB, đoạn của Multimedia chỉ liên quan
    mạnh tới câu con về Multimedia, nên lấy max để nó không bị chìm.

    Không thay hẳn bảng của retriever bằng bảng của reranker mà gộp cả hai bằng RRF
    (weights = (retriever, reranker)): reranker mạnh nhưng có lúc sai, giữ tín hiệu của
    retriever làm bảng xếp hạng ổn định hơn.

    multi_weights: trọng số riêng cho câu hỏi CÓ câu con (None = dùng chung `weights`). Với các
    câu này, thứ hạng của retriever đã bị loãng khi gộp nhiều bảng (nhật ký mục 6.5), còn điểm
    reranker lấy max theo từng câu con nên không bị loãng -> có thể tin reranker hơn.
    Đo trên train (48 câu có câu con): (1,1) 89.7, (1,2) 90.5, (1,3) 89.6, (0,1) 91.4
    -> mặc định (0, 1): với câu có câu con, xếp hạng hoàn toàn theo reranker.

    cover=c (A2): với câu có câu con, bảo đảm top-c đoạn của MỖI câu con (theo điểm reranker của
    riêng câu con đó) có mặt trong top `cover_top` (= 5, mốc tính Recall@5). Lý do: điểm max có
    thể bị một câu con "dễ" chiếm hết top-5, đoạn của câu con còn lại rơi xuống hạng 6+.
    Các đoạn được bảo đảm vẫn sắp theo thứ hạng cuối cùng, chỉ đẩy ra các đoạn thừa ở cuối top-5.
    Đo trên train: +0.2 với BM25:dense = 1.5:1 nhưng -0.1 đến -0.4 với 1:1 -> nhỏ, không nhất quán,
    nên mặc định TẮT (cover=0). Giữ code để thử lại khi retrieval thay đổi.
    """

    def __init__(
        self,
        first_stage: MultiQueryRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        n_candidates: int = 30,  # A1: 20 -> 30 trên train 92.1 -> 93.2; 40 không tăng thêm
        weights: tuple[float, float] = (1.0, 1.0),
        k: int = 60,
        multi_weights: tuple[float, float] | None = (0.0, 1.0),
        cover: int = 0,
        cover_top: int = 5,
    ):
        self.first_stage = first_stage or MultiQueryRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.n_candidates = n_candidates
        self.weights = list(weights)
        self.k = k
        self.multi_weights = list(multi_weights) if multi_weights else None
        self.cover = cover
        self.cover_top = cover_top

    def prepare(self) -> None:
        self.reranker.load()

    def index(self, corpus: list[Document]) -> None:
        self.docs = {d.id: d for d in corpus}
        self.first_stage.index(corpus)

    def plan(self, query: str) -> QueryPlan:
        return self.first_stage.plan(query)

    def search_plan(self, plan: QueryPlan, top_n: int = 10) -> list[tuple[int, float]]:
        candidates = [d for d, _ in self.first_stage.search_plan(plan, top_n=self.n_candidates)]
        docs = [self.docs[d] for d in candidates]

        rerank_queries = list(dict.fromkeys([plan.rewrite, *plan.subqueries]))
        per_query = [self.reranker.scores(q, docs) for q in rerank_queries]
        best = [max(col) for col in zip(*per_query, strict=True)]
        by_reranker = [
            d for d, _ in sorted(zip(candidates, best, strict=True), key=lambda x: -x[1])
        ]

        weights = self.multi_weights if (plan.subqueries and self.multi_weights) else self.weights
        final = rrf([candidates, by_reranker], weights, self.k)
        if self.cover and plan.subqueries:
            scores_of = dict(zip(rerank_queries, per_query, strict=True))
            final = self._cover(final, candidates, [scores_of[q] for q in plan.subqueries])
        return final[:top_n]

    def _cover(
        self, final: list[tuple[int, float]], candidates: list[int], sub_scores: list[list[float]]
    ) -> list[tuple[int, float]]:
        """Đưa top-`cover` của mỗi câu con vào top-`cover_top`, giữ thứ tự của `final`."""
        sub_rankings = [
            [d for d, _ in sorted(zip(candidates, sc, strict=True), key=lambda x: -x[1])]
            for sc in sub_scores
        ]
        must = list(
            dict.fromkeys(r[i] for i in range(self.cover) for r in sub_rankings if i < len(r))
        )[: self.cover_top]
        order = [d for d, _ in final]
        rest = [d for d in order if d not in must]
        top = sorted(must + rest[: self.cover_top - len(must)], key=order.index)
        tail = [d for d in order if d not in top]
        score = dict(final)
        return [(d, score[d]) for d in top + tail]

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        return self.search_plan(self.plan(query), top_n)
