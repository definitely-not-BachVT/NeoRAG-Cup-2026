"""A4: LLM xếp hạng lại cả danh sách ứng viên (listwise rerank, kiểu RankGPT).

Cross-encoder (rerank.py) chấm từng cặp (câu hỏi, đoạn) riêng lẻ và chủ yếu khớp bề mặt câu chữ.
Với câu hỏi nói về TÌNH HUỐNG của người hỏi ("có phòng riêng để chụp ảnh không") trong khi tài
liệu viết theo CHỦ ĐỀ (đoạn "Quyền lợi"), nó xếp đoạn đúng ở hạng 4-5. Ở đây LLM đọc cả
danh sách ứng viên cùng lúc, so sánh các đoạn với nhau và suy luận ý định của câu hỏi.

Cách làm:
    1. Lấy top `n_rerank` (= 15) của retriever bên trong (đã qua cross-encoder).
    2. 1 lời gọi LLM: trả về danh sách id theo thứ tự liên quan giảm dần (JSON).
    3. Gộp bảng của LLM với bảng cũ bằng RRF (weights = (cũ, LLM)), không thay hẳn:
       LLM không tất định và đôi khi sai, giữ bảng cũ làm điểm tựa.

Lỗi ở bất kỳ bước nào (mạng, JSON hỏng, id lạ/trùng) đều không làm hỏng kết quả: id sai bị bỏ,
id thiếu được bổ sung theo thứ tự cũ, lỗi mạng thì trả nguyên bảng cũ.

Prompt không chứa ngày hôm nay (ngày nằm trong khóa cache; sang ngày mới sẽ phải gọi lại hết).
Kết quả được cache trong .cache/pipeline.sqlite (qua LLMClient).
"""

from __future__ import annotations

import json
import os
import re

from my_solution.query import QueryPlan
from my_solution.retrievers import rrf
from neorag.data import Document
from neorag.llm import LLMClient

RANK_PROMPT = """Bạn xếp hạng các đoạn tài liệu theo mức độ giúp trả lời một câu hỏi về các câu lạc bộ thuộc Học viện Công nghệ Bưu chính Viễn thông (PTIT): CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC), CLB Nhà Sáng Tạo Game PTIT (GCC) và cộng đồng S2B. Bạn KHÔNG trả lời câu hỏi.

Mỗi đoạn có dạng [id=<số>] <CLB> | <chủ đề> rồi đến nội dung. Câu hỏi có thể viết không dấu, viết tắt hoặc dùng tiếng lóng; phần "Diễn giải" và "Các ý cần tra" (nếu có) là gợi ý để hiểu đúng ý.

Quy tắc:
1. Xếp đoạn CHỨA thông tin cần để trả lời lên trước, kể cả khi câu hỏi diễn đạt khác với tài liệu (câu hỏi về tình huống cá nhân thường được trả lời bởi đoạn nói về chủ đề tương ứng).
2. Câu hỏi nhiều ý hoặc về nhiều CLB: đoạn chứa thông tin cho từng ý, từng CLB đều phải được xếp cao.
3. Đoạn chỉ cùng chủ đề hoặc cùng CLB nhưng không chứa thông tin cần thiết thì xếp thấp.
4. Liệt kê MỌI id đã cho, mỗi id đúng một lần, không thêm id khác.

Chỉ trả về một object JSON, không giải thích:
{"ranking": [<id>, <id>, ...]}"""

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


def parse_ranking(reply: str, valid_ids: list[int]) -> list[int] | None:
    """Đọc danh sách id từ câu trả lời của LLM, chỉ giữ id hợp lệ và bỏ trùng.

    Trả về None nếu không đọc được gì (JSON hỏng, thiếu trường, không có id hợp lệ nào), để
    nơi gọi quay về bảng xếp hạng cũ. Id hợp lệ nhưng LLM bỏ sót sẽ được bổ sung theo thứ tự
    của `valid_ids` (thứ tự cũ), nên kết quả luôn là một hoán vị của `valid_ids`.
    """
    match = _JSON_OBJECT.search(reply or "")
    try:
        data = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        return None
    raw = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None

    valid = set(valid_ids)
    ranking: list[int] = []
    for item in raw:
        # bool là con của int trong Python; số thực/chuỗi số ("12") không phải id hợp lệ
        is_id = isinstance(item, int) and not isinstance(item, bool)
        if is_id and item in valid and item not in ranking:
            ranking.append(item)
    if not ranking:
        return None
    return ranking + [d for d in valid_ids if d not in ranking]


class ListwiseReranker:
    """Bọc một retriever có plan/search_plan (RerankedRetriever); xếp lại top `n_rerank` bằng LLM.

    n_rerank: số ứng viên đưa cho LLM (15: đủ phủ top-10 mà prompt vẫn ngắn, ít thiên vị vị trí).
    weights:  trọng số RRF (bảng cũ, bảng LLM). Mặc định (1, 1), không chỉnh theo train.
    """

    def __init__(
        self,
        base,
        llm: LLMClient | None = None,
        model: str | None = None,
        n_rerank: int = 15,
        weights: tuple[float, float] = (1.0, 1.0),
        k: int = 60,
    ):
        self.base = base
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.n_rerank = n_rerank
        self.weights = list(weights)
        self.k = k

    def prepare(self) -> None:
        if hasattr(self.base, "prepare"):
            self.base.prepare()

    def index(self, corpus: list[Document]) -> None:
        self.docs = {d.id: d for d in corpus}
        self.base.index(corpus)

    def plan(self, query: str) -> QueryPlan:
        return self.base.plan(query)

    def build_messages(self, plan: QueryPlan, ids: list[int]) -> list[dict]:
        lines = [f"CÂU HỎI: {plan.original}"]
        if plan.rewrite.strip() != plan.original.strip():
            lines.append(f"Diễn giải: {plan.rewrite}")
        if plan.subqueries:
            lines.append("Các ý cần tra: " + " | ".join(plan.subqueries))
        passages = "\n\n".join(
            f"[id={i}] {self.docs[i].club} | {self.docs[i].topic}\n{self.docs[i].text}" for i in ids
        )
        user = "\n".join(lines) + f"\n\nCÁC ĐOẠN ({len(ids)}):\n{passages}"
        return [
            {"role": "system", "content": RANK_PROMPT},
            {"role": "user", "content": user},
        ]

    def search_plan(self, plan: QueryPlan, top_n: int = 10) -> list[tuple[int, float]]:
        base = self.base.search_plan(plan, top_n=max(top_n, self.n_rerank))
        ids = [d for d, _ in base[: self.n_rerank]]
        if len(ids) < 2:
            return base[:top_n]

        try:
            reply = self.llm.chat(self.build_messages(plan, ids), model=self.model)
        except Exception:  # lỗi mạng/hết hạn mức: giữ nguyên bảng cũ
            return base[:top_n]
        ranking = parse_ranking(reply, ids)
        if ranking is None:
            return base[:top_n]

        # Các đoạn ngoài top n_rerank (nếu top_n > n_rerank) nối vào cuối, giữ thứ tự cũ.
        fused = rrf([ids, ranking], self.weights, self.k)
        tail = [(d, s) for d, s in base[self.n_rerank :]]
        return (fused + tail)[:top_n]

    def search(self, query: str, top_n: int = 10) -> list[tuple[int, float]]:
        return self.search_plan(self.plan(query), top_n)
