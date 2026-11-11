"""Step-Back: sinh một câu hỏi KHÁI QUÁT hơn để tìm thêm ứng viên (retrieval).

Câu hỏi nói về tình huống cá nhân ("em ko biet ve, chi gioi noi chuyen thi vao ban nao") còn tài
liệu viết theo chủ đề ("Cơ cấu các ban"). Câu khái quát ("CLB Multimedia PTIT có những ban nào,
mỗi ban làm gì?") khớp thẳng với chủ đề đó, nên kéo được đoạn mà câu cụ thể không với tới.

Chỉ dùng để TÌM ỨNG VIÊN (thêm một bảng xếp hạng vào RRF của MultiQueryRetriever). Không đưa
vào bước chấm của cross-encoder (câu chung chung kéo đoạn "Giới thiệu chung" lên khi lấy max)
và không đưa vào prompt của A4.

Lời gọi LLM riêng, không nhét vào JSON viết lại (query.py): đổi prompt viết lại sẽ làm bản
viết lại và câu con của mọi câu bị sinh lại, tức đổi hai thứ cùng lúc. Kết quả được cache
trong .cache/pipeline.sqlite. Lỗi mạng hoặc JSON hỏng thì không có câu khái quát (retrieval
chạy như cũ).
"""

from __future__ import annotations

import json
import os
import re

from my_solution.query import QueryPlan
from neorag.llm import LLMClient
from neorag.text import normalize

STEPBACK_PROMPT = """Bạn giúp hệ thống TÌM KIẾM tài liệu về các câu lạc bộ thuộc Học viện Công nghệ Bưu chính Viễn thông (PTIT): CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC), CLB Nhà Sáng Tạo Game PTIT (GCC) và cộng đồng S2B. Bạn KHÔNG trả lời câu hỏi.

Cho một câu hỏi cụ thể (kèm bản viết lại có dấu), hãy viết MỘT câu hỏi KHÁI QUÁT hơn về chủ đề mà tài liệu chứa câu trả lời sẽ được viết. Các chủ đề thường gặp: giới thiệu chung, quyền lợi, cơ cấu và các ban, chương trình đào tạo, điều kiện tham gia, quy trình tuyển thành viên, phí và quỹ, nội quy và nghĩa vụ, sự kiện, giải thưởng, ban chủ nhiệm, liên hệ.

Quy tắc:
1. Giữ đúng CLB mà câu hỏi nhắc tới, gọi bằng tên chính thức. Câu hỏi không nhắc CLB nào thì không nêu CLB.
2. Bỏ chi tiết riêng của tình huống người hỏi (hoàn cảnh cá nhân, tên công cụ cụ thể, mốc ngày cụ thể); giữ chủ đề.
3. Không trả lời, không thêm thông tin mới.
4. Đúng một câu hỏi, tiếng Việt có dấu.

Chỉ trả về một object JSON, không giải thích:
{"stepback": "..."}"""

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)
MAX_CHARS = 300


def parse_stepback(reply: str, plan: QueryPlan) -> str | None:
    """Đọc câu khái quát từ câu trả lời của LLM; None nếu hỏng, rỗng, quá dài hoặc trùng câu đã có."""
    match = _JSON_OBJECT.search(reply or "")
    try:
        data = json.loads(match.group()) if match else None
    except json.JSONDecodeError:
        return None
    text = data.get("stepback") if isinstance(data, dict) else None
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        return None
    existing = {normalize(q) for q in plan.queries}
    return None if normalize(text) in existing else text


class StepBack:
    def __init__(self, llm: LLMClient | None = None, model: str | None = None):
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")

    def generate(self, plan: QueryPlan) -> str | None:
        messages = [
            {"role": "system", "content": STEPBACK_PROMPT},
            {
                "role": "user",
                "content": f"Câu hỏi: {plan.original}\nBản viết lại: {plan.rewrite}",
            },
        ]
        try:
            reply = self.llm.chat(messages, model=self.model)
        except Exception:  # lỗi mạng/hết hạn mức: truy xuất như cũ
            return None
        return parse_stepback(reply, plan)
