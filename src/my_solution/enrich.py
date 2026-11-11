"""A3 – Làm giàu tài liệu lúc index (doc2query).

Vấn đề: retrieval so khớp CÂU HỎI với ĐOẠN VĂN, hai loại văn bản rất khác nhau. Câu hỏi viết
không dấu, văn nói ("clb lam game cua hv co cho nao rieng ko"), còn đoạn văn viết trang trọng.
Tệ hơn, một thông tin có thể nằm lẫn trong đoạn có chủ đề khác: "GCC dành cho sinh viên mọi
ngành" nằm trong đoạn "Lịch sử & thành lập", nên câu hỏi "ngành Marketing có vào được không"
khó tìm ra đoạn đó (lỗi public-0040, nhật ký mục 6.5b).

Cách làm (doc2query): lúc index, cho LLM đọc từng đoạn và viết ra các câu hỏi mà đoạn đó trả
lời được, bằng nhiều cách hỏi (chuẩn, văn nói, không dấu), cùng các từ khóa. Lúc truy xuất, câu
hỏi của người dùng được so với cả đoạn văn lẫn các câu hỏi sinh sẵn: hỏi-với-hỏi dễ khớp hơn.

Phần làm giàu CHỈ dùng để TÌM KIẾM (BM25, dense). Reranker và prompt sinh câu trả lời vẫn dùng
văn bản gốc: Groundedness được chấm trên văn bản gốc, không được để câu hỏi do LLM sinh lọt vào.

Chi phí: 1 lời gọi LLM cho mỗi đoạn (118 đoạn), có cache, nên chỉ tốn một lần.

Lưu trữ: nội dung làm giàu được ghi RIÊNG vào artifacts/doc2query.jsonl, không trộn vào
data/corpus.jsonl. Mỗi dòng gắn với mã hash nội dung của đoạn: đoạn nào đổi nội dung thì tự sinh
lại, đoạn không đổi thì đọc lại từ file (chạy lại cho kết quả giống hệt, không gọi API).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from neorag.data import Document
from neorag.llm import LLMClient

ENRICH_PROMPT = """Bạn giúp xây chỉ mục TÌM KIẾM cho tài liệu về các câu lạc bộ của Học viện Công nghệ Bưu chính Viễn thông (PTIT). Bạn nhận một đoạn tài liệu, kèm tên CLB và chủ đề của đoạn.

Nhiệm vụ:
1. "questions": viết 6 câu hỏi khác nhau mà ĐOẠN NÀY trả lời được trực tiếp, sao cho các câu hỏi bao phủ MỌI thông tin trong đoạn, kể cả các chi tiết phụ nằm lẫn trong đoạn. Đa dạng cách hỏi như sinh viên thật: 2 câu viết chuẩn có dấu; 2 câu văn nói, diễn đạt khác với từ ngữ trong đoạn; 2 câu viết không dấu kiểu tin nhắn, có viết tắt (ko, dc, dk, tv, clb...). Gọi CLB bằng nhiều cách khác nhau: tên đầy đủ, tên viết tắt, hoặc cách gọi thân mật theo lĩnh vực của CLB.
2. "keywords": tối đa 10 từ khóa hoặc cụm từ mà người hỏi có thể dùng để tìm đoạn này, gồm cả từ đồng nghĩa không xuất hiện trong đoạn.

Chỉ hỏi về thông tin CÓ trong đoạn. Không thêm thông tin mới, không trả lời câu hỏi.
Chỉ trả về một object JSON, không giải thích:
{"questions": ["...", "..."], "keywords": ["...", "..."]}"""

_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)
MAX_QUESTIONS = 8
MAX_KEYWORDS = 12


@dataclass
class Enrichment:
    questions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Văn bản bổ sung cho BM25: các câu hỏi + từ khóa."""
        return " | ".join([*self.questions, *self.keywords])


def _strings(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()][:limit]


def parse_enrichment(reply: str) -> Enrichment:
    """Đọc JSON từ câu trả lời của LLM. Hỏng thì trả về Enrichment rỗng, không bao giờ lỗi."""
    match = _JSON_OBJECT.search(reply or "")
    try:
        data = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        return Enrichment()
    return Enrichment(
        _strings(data.get("questions"), MAX_QUESTIONS), _strings(data.get("keywords"), MAX_KEYWORDS)
    )


def doc_hash(doc: Document) -> str:
    """Dấu vân tay nội dung của đoạn: đổi club/topic/text là đổi hash."""
    raw = "\x1f".join([doc.club, doc.topic, doc.text])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DocEnricher:
    """Sinh Enrichment cho từng đoạn. Lỗi ở đoạn nào thì đoạn đó không được làm giàu.

    store_path: file JSONL lưu riêng nội dung làm giàu (None = không lưu file, chỉ dùng cache).
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        model: str | None = None,
        max_workers: int = 4,
        store_path: str | Path | None = "artifacts/doc2query.jsonl",
    ):
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.model = model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.max_workers = max_workers  # gateway cho tối đa 8 request đồng thời
        self.store_path = Path(store_path) if store_path else None

    def _load_store(self) -> dict[int, dict]:
        if not self.store_path or not self.store_path.exists():
            return {}
        with open(self.store_path, encoding="utf-8") as f:
            return {row["id"]: row for row in map(json.loads, filter(str.strip, f))}

    def _save_store(self, corpus: list[Document], result: dict[int, Enrichment]) -> None:
        """Ghi các đoạn đã làm giàu thành công (đoạn lỗi không ghi, để lần sau sinh lại)."""
        if not self.store_path:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for doc in sorted(corpus, key=lambda d: d.id):
                e = result.get(doc.id)
                if e and (e.questions or e.keywords):
                    row = {
                        "id": doc.id,
                        "hash": doc_hash(doc),
                        "club": doc.club,
                        "topic": doc.topic,
                        "questions": e.questions,
                        "keywords": e.keywords,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def enrich_one(self, doc: Document) -> Enrichment:
        messages = [
            {"role": "system", "content": ENRICH_PROMPT},
            {
                "role": "user",
                "content": f"CLB: {doc.club}\nChủ đề: {doc.topic}\nĐoạn tài liệu:\n{doc.text}",
            },
        ]
        try:
            return parse_enrichment(self.llm.chat(messages, model=self.model))
        except Exception:  # lỗi mạng/hết hạn mức: đoạn này dùng văn bản gốc như trước
            return Enrichment()

    def enrich(self, corpus: list[Document]) -> dict[int, Enrichment]:
        stored = self._load_store()
        result: dict[int, Enrichment] = {}
        todo = []
        for doc in corpus:
            row = stored.get(doc.id)
            if row and row.get("hash") == doc_hash(doc):  # nội dung đoạn không đổi: dùng lại
                result[doc.id] = Enrichment(row.get("questions", []), row.get("keywords", []))
            else:
                todo.append(doc)
        if todo:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                generated = list(pool.map(self.enrich_one, todo))
            result.update(zip((d.id for d in todo), generated, strict=True))
            self._save_store(corpus, result)
        return result
