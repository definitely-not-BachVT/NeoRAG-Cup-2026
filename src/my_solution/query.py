"""Bước 5: dùng LLM viết lại câu hỏi trước khi truy xuất (query rewriting + decomposition).

Hai điểm yếu của retrieval mà quy tắc trong text.py không sửa được:
    1. Câu không dấu, teencode: bge-m3 hiểu kém, nhóm "không dấu" chỉ đạt 65.4 điểm.
       -> LLM khôi phục dấu, giải nghĩa tiếng lóng, gọi CLB bằng tên chính thức.
    2. Câu cần nhiều đoạn ("Trong 3 CLB thuộc S2B, CLB nào lâu đời nhất?"): cả top-5 bị các
       đoạn chung của S2B chiếm hết, không còn chỗ cho đoạn của từng CLB.
       -> LLM tách thành câu hỏi con, mỗi câu con được truy xuất riêng rồi gộp lại.

Mỗi câu hỏi tốn đúng 1 lời gọi LLM, kết quả được cache trong .cache/pipeline.sqlite.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from neorag.llm import LLMClient

REWRITE_PROMPT = """Bạn viết lại câu hỏi để hệ thống TÌM KIẾM tài liệu về các câu lạc bộ của Học viện Công nghệ Bưu chính Viễn thông (PTIT). Bạn KHÔNG trả lời câu hỏi.

Tên chính thức của các CLB:
- CLB Lập Trình PTIT (ProPTIT): CLB lập trình, code.
- CLB Multimedia PTIT (MPC): CLB đa phương tiện, thiết kế, truyền thông, media.
- CLB Nhà Sáng Tạo Game PTIT (GCC): CLB làm game.
- Cộng đồng S2B: cộng đồng chung của cả ba CLB trên.

Nhiệm vụ:
1. "rewrite": viết lại câu hỏi thành tiếng Việt CÓ DẤU, đầy đủ, rõ nghĩa. Giải nghĩa viết tắt và tiếng lóng (ko = không, dk = đăng ký, tv = thành viên, hv = Học viện...). Gọi CLB bằng tên chính thức. Giữ nguyên mọi ý, mọi con số, ngày tháng, tên riêng của câu gốc; không thêm thông tin mới.
2. "subqueries": nếu câu hỏi gồm nhiều ý cần tra cứu riêng (so sánh nhiều CLB, hỏi về "cả 3 CLB", hoặc hỏi hai việc khác nhau), tách thành các câu hỏi con. Mỗi câu con tự đủ nghĩa và nêu tên chính thức của CLB liên quan. Tối đa 4 câu. Câu hỏi chỉ có một ý thì để danh sách rỗng.

Chỉ trả về một object JSON, không giải thích:
{"rewrite": "...", "subqueries": ["...", "..."]}"""

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)
MAX_SUBQUERIES = 4


@dataclass
class QueryPlan:
    """Kết quả viết lại một câu hỏi.

    original:   câu hỏi gốc của người dùng (vẫn được dùng để truy xuất và sinh câu trả lời)
    rewrite:    bản có dấu, đầy đủ, tên CLB chính thức
    subqueries: các câu hỏi con (rỗng nếu câu hỏi chỉ có một ý)
    stepback:   câu hỏi khái quát hơn (stepback.py), chỉ dùng để tìm ứng viên; None = không có
    """

    original: str
    rewrite: str
    subqueries: list[str] = field(default_factory=list)
    stepback: str | None = None

    @property
    def queries(self) -> list[str]:
        """Mọi câu dùng để truy xuất, bỏ trùng, giữ thứ tự (câu khái quát đứng cuối)."""
        extra = [self.stepback] if self.stepback else []
        return list(dict.fromkeys([self.original, self.rewrite, *self.subqueries, *extra]))


def parse_plan(original: str, reply: str) -> QueryPlan:
    """Đọc JSON từ câu trả lời của LLM. Hỏng ở đâu thì lùi về câu gốc ở đó, không bao giờ lỗi.

    LLM đôi khi bọc JSON trong ```json ... ``` hoặc thêm chữ trước/sau, nên lấy đoạn từ dấu
    { đầu tiên đến dấu } cuối cùng rồi mới json.loads.
    """
    match = _JSON_OBJECT.search(reply or "")
    try:
        data = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}

    rewrite = data.get("rewrite")
    rewrite = rewrite.strip() if isinstance(rewrite, str) and rewrite.strip() else original

    subqueries = data.get("subqueries")
    if not isinstance(subqueries, list):
        subqueries = []
    subqueries = [s.strip() for s in subqueries if isinstance(s, str) and s.strip()]
    return QueryPlan(original, rewrite, subqueries[:MAX_SUBQUERIES])


class QueryRewriter:
    def __init__(self, llm: LLMClient | None = None, model: str | None = None):
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")

    def plan(self, query: str) -> QueryPlan:
        messages = [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": query},
        ]
        try:
            reply = self.llm.chat(messages, model=self.model)
        except Exception:  # lỗi mạng/hết hạn mức: vẫn truy xuất được bằng câu gốc
            reply = ""
        return parse_plan(query, reply)
