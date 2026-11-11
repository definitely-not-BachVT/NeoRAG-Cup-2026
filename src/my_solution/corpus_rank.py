"""A5: cho LLM đọc TOÀN BỘ corpus rồi chọn các đoạn liên quan nhất.

Corpus chỉ có 118 đoạn (~50 nghìn ký tự, ~20 nghìn token) nên vừa một prompt. Khác A4 (chỉ xếp
lại 15 ứng viên mà retriever đã chọn), A5 nhìn được cả các đoạn nằm NGOÀI danh sách ứng viên
(6/196 đoạn đúng trên train) và so sánh câu hỏi với mọi đoạn.

Cách làm: 1 lời gọi LLM trả về top-10 id; gộp RRF với kết quả của retriever bên trong (A4),
weights = (cũ, A5) mặc định (1, 1), không chỉnh theo train. Lỗi (JSON hỏng, id lạ/trùng, lỗi mạng)
đều quay về kết quả cũ.

Corpus nằm ở system message (phần đầu cố định, để gateway tận dụng cache tiền tố nếu có); câu
hỏi ở user message. Không có ngày hôm nay (ngày nằm trong khóa cache). Kết quả được cache trong
.cache/pipeline.sqlite.
"""

from __future__ import annotations

import json
import logging
import os
import re

from my_solution.query import QueryPlan
from my_solution.retrievers import rrf
from neorag.data import Document
from neorag.llm import LLMClient

CORPUS_PROMPT = """Bạn chọn các đoạn tài liệu giúp trả lời một câu hỏi về các câu lạc bộ thuộc Học viện Công nghệ Bưu chính Viễn thông (PTIT): CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC), CLB Nhà Sáng Tạo Game PTIT (GCC) và cộng đồng S2B. Bạn KHÔNG trả lời câu hỏi.

Dưới đây là TOÀN BỘ tài liệu. Mỗi đoạn có dạng [id=<số>] <CLB> | <chủ đề> rồi đến nội dung.

{corpus}

Quy tắc:
1. Chọn tối đa {top_k} đoạn CHỨA thông tin cần để trả lời, xếp đoạn quan trọng nhất lên đầu. Câu hỏi có thể viết không dấu, viết tắt hoặc dùng tiếng lóng; phần "Diễn giải" và "Các ý cần tra" (nếu có) là gợi ý để hiểu đúng ý.
2. Câu hỏi về tình huống cá nhân thường được trả lời bởi đoạn nói về chủ đề tương ứng (quyền lợi, cơ cấu, đào tạo, điều kiện tham gia...), dù diễn đạt khác.
3. Câu hỏi nhiều ý hoặc về nhiều CLB: chọn đoạn cho từng ý, từng CLB.
4. Đoạn chỉ cùng CLB hoặc cùng chủ đề chung nhưng không chứa thông tin cần thiết thì không chọn.
5. Chỉ dùng id có trong tài liệu, mỗi id một lần.

Chỉ trả về một object JSON, không giải thích:
{{"ranking": [<id>, <id>, ...]}}"""

log = logging.getLogger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


def parse_corpus_ranking(reply: str, valid_ids: set[int], top_k: int = 10) -> list[int] | None:
    """Danh sách id hợp lệ, bỏ trùng, tối đa top_k, giữ thứ tự; None nếu không đọc được gì."""
    match = _JSON_OBJECT.search(reply or "")
    try:
        data = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        return None
    raw = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    ranking: list[int] = []
    for item in raw:
        is_id = isinstance(item, int) and not isinstance(item, bool)
        if is_id and item in valid_ids and item not in ranking:
            ranking.append(item)
    return ranking[:top_k] or None


class CorpusRanker:
    """Bọc một retriever có plan/search_plan (A4); gộp RRF với danh sách do LLM chọn từ toàn corpus."""

    def __init__(
        self,
        base,
        llm: LLMClient | None = None,
        model: str | None = None,
        weights: tuple[float, float] = (1.0, 1.0),
        top_k: int = 10,
        k: int = 60,
    ):
        self.base = base
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.weights = list(weights)
        self.top_k = top_k
        self.k = k
        self.system = ""
        self.valid_ids: set[int] = set()
        self.calls = 0  # số câu đã gọi A5
        self.fallbacks = 0  # số câu A5 lỗi/quá hạn nên dùng kết quả A4

    def prepare(self) -> None:
        if hasattr(self.base, "prepare"):
            self.base.prepare()

    def index(self, corpus: list[Document]) -> None:
        self.base.index(corpus)
        self.valid_ids = {d.id for d in corpus}
        text = "\n\n".join(
            f"[id={d.id}] {d.club} | {d.topic}\n{d.text}"
            for d in sorted(corpus, key=lambda d: d.id)
        )
        self.system = CORPUS_PROMPT.format(corpus=text, top_k=self.top_k)

    def plan(self, query: str) -> QueryPlan:
        return self.base.plan(query)

    def build_messages(self, plan: QueryPlan) -> list[dict]:
        lines = [f"CÂU HỎI: {plan.original}"]
        if plan.rewrite.strip() != plan.original.strip():
            lines.append(f"Diễn giải: {plan.rewrite}")
        if plan.subqueries:
            lines.append("Các ý cần tra: " + " | ".join(plan.subqueries))
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def search_plan(self, plan: QueryPlan, top_n: int = 10) -> list[tuple[int, float]]:
        base = self.base.search_plan(plan, top_n=top_n)
        self.calls += 1
        try:
            reply = self.llm.chat(self.build_messages(plan), model=self.model)
        except Exception as exc:  # lỗi mạng/hết hạn mức: giữ nguyên kết quả cũ
            self.fallbacks += 1
            log.warning("A5 lùi về A4 (%s): %s", type(exc).__name__, plan.original[:60])
            return base
        ranking = parse_corpus_ranking(reply, self.valid_ids, self.top_k)
        if ranking is None:
            self.fallbacks += 1
            log.warning("A5 lùi về A4 (câu trả lời không đọc được): %s", plan.original[:60])
            return base
        return rrf([[d for d, _ in base], ranking], self.weights, self.k)[:top_n]

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        return self.search_plan(self.plan(query), top_n)
