"""Chạy pipeline có ghi vết từng bước, cho chatbot (app/chat.py). KHÔNG dùng khi chấm điểm.

`run_traced` làm đúng những việc của `MyPipeline.run` (plan -> search_plan -> build_messages ->
chat -> clean_answer), nhưng:
  - đo thời gian từng lớp retriever (bọc tạm `search_plan` của mỗi lớp, trừ thời gian lớp con);
  - báo kết quả từng bước qua `on_update` để giao diện hiện dần;
  - trả về bản viết lại, câu hỏi con và các đoạn đã chọn để hiển thị.

Pipeline không phải MyPipeline (ví dụ baseline của BTC) thì chỉ đo tổng thời gian.
"""

from __future__ import annotations

import difflib
import html
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from my_solution.prompts import build_messages
from my_solution.text import strip_accents

_lock = threading.Lock()  # bọc tạm search_plan nên không cho hai lượt chạy đè nhau

# Tên hiển thị theo lớp; lớp lạ dùng tên lớp.
STAGE_NAMES = {
    "MultiQueryRetriever": "Tìm ứng viên",
    "RerankedRetriever": "Chấm lại bằng cross-encoder",
    "ListwiseReranker": "LLM xếp lại",
    "CorpusRanker": "LLM đọc toàn bộ tài liệu",
}
REWRITE = "Viết lại câu hỏi"
ANSWER = "Viết câu trả lời"


@dataclass
class Stage:
    name: str
    seconds: float | None = None  # None = chưa chạy xong
    note: str = ""  # ví dụ "quá hạn, dùng kết quả bước trước"


@dataclass
class Trace:
    question: str
    rewrite: str = ""
    subqueries: list[str] = field(default_factory=list)
    hits: list[tuple[int, float]] = field(default_factory=list)  # (id đoạn, điểm) đã xếp hạng
    n_context: int = 0  # số đoạn đầu được đưa vào prompt
    answer: str = ""
    stages: list[Stage] = field(default_factory=list)
    fallback: bool = False  # A5 quá hạn/lỗi, dùng kết quả A4
    traced: bool = True  # False: pipeline không hỗ trợ ghi vết từng bước

    @property
    def total(self) -> float:
        return sum(s.seconds or 0.0 for s in self.stages)


def _chain(retriever) -> list:
    """Các lớp bọc nhau từ ngoài vào trong (qua thuộc tính `.base` hoặc `.first_stage`), dừng ở lớp không còn `.base`."""
    layers = []
    node = retriever
    while node is not None and hasattr(node, "search_plan"):
        layers.append(node)
        node = getattr(node, "base", None) or getattr(node, "first_stage", None)
    return layers


def stage_name(layer) -> str:
    return STAGE_NAMES.get(type(layer).__name__, type(layer).__name__)


def run_traced(pipeline, query: str, on_update: Callable[[Trace], None] | None = None) -> Trace:
    """Trả lời một câu hỏi và ghi vết. Lỗi của pipeline được ném lên cho giao diện xử lý."""
    trace = Trace(question=query)

    def emit() -> None:
        if on_update:
            on_update(trace)

    retriever = getattr(pipeline, "retriever", None)
    if not (hasattr(retriever, "plan") and hasattr(retriever, "search_plan")):
        t0 = time.perf_counter()
        out = pipeline.run(query)
        trace.traced = False
        trace.answer = out.answer
        trace.hits = [(int(i), 0.0) for i in out.retrieved_ids]
        trace.n_context = len(trace.hits)
        trace.stages = [Stage("Trả lời", time.perf_counter() - t0)]
        emit()
        return trace

    layers = _chain(retriever)
    trace.stages = (
        [Stage(REWRITE)] + [Stage(stage_name(x)) for x in reversed(layers)] + [Stage(ANSWER)]
    )
    emit()

    t0 = time.perf_counter()
    plan = retriever.plan(query)
    trace.stages[0].seconds = time.perf_counter() - t0
    trace.rewrite, trace.subqueries = plan.rewrite, list(plan.subqueries)
    emit()

    # Thời gian bao gồm cả lớp con; lớp ngoài trừ đi lớp trong để ra thời gian riêng.
    inclusive: dict[int, float] = {}
    originals = {id(x): x.search_plan for x in layers}
    ranker = next((x for x in layers if type(x).__name__ == "CorpusRanker"), None)
    fallbacks_before = getattr(ranker, "fallbacks", 0)

    def wrap(layer):
        original = originals[id(layer)]

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                inclusive[id(layer)] = inclusive.get(id(layer), 0.0) + time.perf_counter() - start

        return timed

    top_n = getattr(pipeline, "n_retrieved", 10)
    with _lock:
        for layer in layers:
            layer.search_plan = wrap(layer)
        try:
            hits = retriever.search_plan(plan, top_n=top_n)
        finally:
            for layer in layers:
                layer.__dict__.pop("search_plan", None)  # bỏ bản bọc, trả về phương thức lớp

    for i, layer in enumerate(layers):  # layers[0] là lớp ngoài cùng
        inner = inclusive.get(id(layers[i + 1]), 0.0) if i + 1 < len(layers) else 0.0
        stage = trace.stages[len(layers) - i]  # đảo thứ tự: lớp trong chạy trước
        stage.seconds = max(0.0, inclusive.get(id(layer), 0.0) - inner)
    if ranker is not None and getattr(ranker, "fallbacks", 0) > fallbacks_before:
        trace.fallback = True
        next(
            s for s in trace.stages if s.name == STAGE_NAMES["CorpusRanker"]
        ).note = "quá hạn hoặc lỗi, dùng kết quả bước trước"
    trace.hits = [(int(i), float(s)) for i, s in hits]  # kiểu thuần Python để lưu JSON được
    trace.n_context = min(len(hits), pipeline.n_context)
    emit()

    from my_solution.pipeline import clean_answer  # tránh vòng import khi nạp module

    docs = [pipeline.docs[doc_id] for doc_id, _ in hits]
    messages = build_messages(
        query,
        docs[: pipeline.n_context],
        date.today().strftime("%d/%m/%Y"),
        rewrite=plan.rewrite,
        version=pipeline.prompt_version,
    )
    t0 = time.perf_counter()
    trace.answer = clean_answer(pipeline.llm.chat(messages, model=pipeline.llm_model))
    trace.stages[-1].seconds = time.perf_counter() - t0
    emit()
    return trace


# ----------------------------------------------------------------------------------------------
# Tô dấu: so câu gốc với bản viết lại
# ----------------------------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return text.split()


def mark_rewrite(original: str, rewrite: str) -> str:
    """HTML của bản viết lại, đánh dấu những gì pipeline đã làm với câu gốc.

    - `<mark class="dau">`: từ được khôi phục dấu (câu gốc viết không dấu, bản viết lại có dấu);
    - `<mark class="giainghia">`: từ được viết lại khác hẳn (viết tắt được giải nghĩa: ko -> không).
    Dấu câu dính vào từ được tách ra trước khi so, nên "ko," khớp với "không,".
    """

    def core(w: str) -> str:
        return strip_accents(w).lower().strip(".,;:!?()\"'")

    orig = _words(original)
    new = _words(rewrite)
    sm = difflib.SequenceMatcher(
        a=[core(w) for w in orig], b=[core(w) for w in new], autojunk=False
    )
    out: list[str] = []
    for op, a0, _a1, b0, b1 in sm.get_opcodes():
        for offset, w in enumerate(new[b0:b1]):
            safe = html.escape(w)
            if op == "equal":
                src = orig[a0 + offset]  # từ tương ứng trong câu gốc (cùng số từ ở khối "equal")
                if strip_accents(w) != w and strip_accents(src) == src:
                    safe = f'<mark class="dau">{safe}</mark>'
            elif op in ("replace", "insert"):
                safe = f'<mark class="giainghia">{safe}</mark>'
            out.append(safe)
    return " ".join(out)
