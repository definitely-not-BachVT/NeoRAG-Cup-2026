"""Chatbot hỏi đáp cộng đồng S2B (Demo Product). Chạy: python -m streamlit run app/chat.py

Bố cục: lịch sử (thanh bên) | hội thoại | quá trình trả lời của câu đang chọn (bản viết lại có tô
dấu, câu hỏi con, thời gian từng bước, nguồn). Pipeline mặc định là giải pháp của đội; đổi bằng
biến môi trường NEORAG_PIPELINE="module.path:ClassName" (pipeline khác vẫn chạy, chỉ không có
chi tiết từng bước).

Chạy trên máy của bạn (localhost). Không mở ra internet công khai: key gateway có ngân sách chung
và mọi câu hỏi đều tiêu ngân sách đó. Key chỉ nằm trong .env phía máy chủ, không gửi xuống trình duyệt.
"""

from __future__ import annotations

import logging
import os

import streamlit as st
import streamlit.components.v1 as components

from my_solution.chat_view import CSS, brand_html, escape_md, masthead_html, trace_html
from my_solution.history import History, day_label, entry_to_trace
from my_solution.trace import Trace, run_traced
from neorag.data import load_corpus
from neorag.pipelines import load_pipeline

PIPELINE = os.getenv("NEORAG_PIPELINE", "my_solution.pipeline:MyPipeline")
log = logging.getLogger("chat")

# Câu hỏi mẫu lấy từ tập train (không dấu, viết tắt), để người xem thử ngay.
EXAMPLES = [
    "e ve tay con yeu, ve hinh khoi voi dang nguoi chua chuan, vao clb game ben ve co day may cai co ban nay ko",
    "clb multimedia co phong rieng de chup anh quay phim ko",
    "multimedia ptit ten tieng anh la gi, mpc la viet tat cua gi",
]
MAX_QUESTION_CHARS = 600  # câu quá dài tốn token ở cả 4 lời gọi LLM và không cần cho câu hỏi thật
LOAD_ERROR_TEXT = (
    "Không khởi động được pipeline. Kiểm tra OPENAI_API_KEY trong file .env, kết nối mạng và "
    "bộ nhớ GPU (cần khoảng 3 GB trống cho reranker), rồi tải lại trang."
)
ERROR_TEXT = (
    "Chưa trả lời được câu này. Kiểm tra kết nối mạng và khóa OPENAI_API_KEY trong file .env "
    "rồi hỏi lại."
)

st.set_page_config(page_title="Hỏi đáp cộng đồng S2B", page_icon="💬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Đang nạp model và xây chỉ mục (lần đầu có thể mất vài phút)...")
def get_pipeline():
    corpus = load_corpus()
    pipeline = load_pipeline(PIPELINE)
    pipeline.prepare()
    pipeline.index(corpus)
    return pipeline, {d.id: d for d in corpus}


try:
    pipeline, docs = get_pipeline()
except (
    Exception
):  # thiếu key, hết bộ nhớ GPU, không tải được model: thông báo gọn thay vì traceback
    log.exception("Không nạp được pipeline %s", PIPELINE)
    st.markdown(f'<div class="warn">{LOAD_ERROR_TEXT}</div>', unsafe_allow_html=True)
    st.stop()
history = History()

ss = st.session_state
ss.setdefault("turns", [])  # các mục đang hiển thị (mỗi mục là một lần hỏi đáp đã lưu)
ss.setdefault("focus", None)  # chỉ số mục có "quá trình" đang xem; None = mục cuối
ss.setdefault("pending", None)  # câu hỏi mẫu vừa bấm
ss.setdefault("error", None)
ss.setdefault("scroll_top", False)  # cuộn lên đầu trang sau khi có câu trả lời mới hoặc mở lịch sử


def open_entry(entry: dict) -> None:
    ss.turns, ss.focus, ss.error, ss.scroll_top = [entry], 0, None, True


def new_chat() -> None:
    ss.turns, ss.focus, ss.error = [], None, None


def delete_entry(entry_id: str) -> None:
    history.delete(entry_id)
    ss.turns = [t for t in ss.turns if t["id"] != entry_id]
    ss.focus = None


def ask_example(text: str) -> None:
    ss.pending = text


def focus_turn(index: int) -> None:
    ss.focus = index


# ---------------------------------------------------------------- thanh bên: lịch sử
with st.sidebar:
    st.markdown(brand_html(), unsafe_allow_html=True)
    st.button("＋ Hỏi mới", on_click=new_chat, use_container_width=True, type="primary")
    saved = history.load()
    if not saved:
        st.caption("Các câu bạn đã hỏi sẽ hiện ở đây.")
    last_label = None
    for entry in saved:
        label = day_label(entry["ts"])
        if label != last_label:
            st.markdown(f'<div class="day">{label}</div>', unsafe_allow_html=True)
            last_label = label
        left, right = st.columns([6, 1], vertical_alignment="center")
        title = entry["question"] if len(entry["question"]) <= 48 else entry["question"][:47] + "…"
        left.button(
            escape_md(title),
            key=f"open-{entry['id']}",
            on_click=open_entry,
            args=(entry,),
            use_container_width=True,
        )
        right.button(
            "✕", key=f"del-{entry['id']}", on_click=delete_entry, args=(entry["id"],), help="Xóa"
        )

# ---------------------------------------------------------------- thân trang
typed = st.chat_input("Hỏi về ProPTIT, Multimedia hoặc GCC...")
question = (typed or ss.pending or "").strip() or None
ss.pending = None
if question and len(question) > MAX_QUESTION_CHARS:
    ss.error = (
        f"Câu hỏi dài {len(question)} ký tự, tối đa {MAX_QUESTION_CHARS}. Hãy rút gọn rồi hỏi lại."
    )
    question = None
elif question:
    ss.error = None

# đầu trang: bản đầy đủ khi chưa có câu hỏi, bản gọn khi đã có hội thoại để nhường chỗ cho câu trả lời
st.markdown(masthead_html(compact=bool(ss.turns or question)), unsafe_allow_html=True)

chat_col, trace_col = st.columns([3, 2], gap="large")

with chat_col:
    if not ss.turns and not question:
        st.markdown(
            '<p class="hint">Gõ câu hỏi ở ô bên dưới, hoặc bấm một câu mẫu để thử ngay:</p>',
            unsafe_allow_html=True,
        )
        for i, example in enumerate(EXAMPLES):
            st.button(
                example,
                key=f"ex-{i}",
                on_click=ask_example,
                args=(example,),
                use_container_width=True,
            )
    for i, turn in enumerate(ss.turns):
        with st.chat_message("user"):
            st.markdown(escape_md(turn["question"]))
        with st.chat_message("assistant"):
            st.markdown(escape_md(turn["answer"]))
            if turn.get("unsaved"):
                st.caption(
                    "Chưa lưu được vào lịch sử (không ghi được file), câu này sẽ mất khi tải lại trang."
                )
            if len(ss.turns) > 1 and i != (ss.focus if ss.focus is not None else len(ss.turns) - 1):
                st.button(
                    "Xem quá trình trả lời",
                    key=f"focus-{turn['id']}",
                    on_click=focus_turn,
                    args=(i,),
                )
    if ss.error:
        st.markdown(f'<div class="warn">{ss.error}</div>', unsafe_allow_html=True)
    answer_slot = st.container()

trace_slot = trace_col.empty()

if question:
    with answer_slot:
        with st.chat_message("user"):
            st.markdown(escape_md(question))
        with st.chat_message("assistant"):
            status = st.empty()
            status.markdown('<p class="hint">Đang tìm câu trả lời…</p>', unsafe_allow_html=True)

            def show(trace: Trace) -> None:
                trace_slot.markdown(trace_html(trace, docs, running=True), unsafe_allow_html=True)

            try:
                result = run_traced(pipeline, question, on_update=show)
            except (
                Exception
            ):  # lỗi mạng, hết hạn mức, ...: thông báo gọn, chi tiết ghi vào log máy chủ
                log.exception("Không trả lời được: %r", question)
                result = None
            if result is None:
                ss.error = ERROR_TEXT  # còn hiện ở lần chạy sau cho đến khi người dùng hỏi tiếp
                status.markdown(f'<div class="warn">{ERROR_TEXT}</div>', unsafe_allow_html=True)
                trace_slot.empty()
            else:
                status.markdown(escape_md(result.answer))
    if result is not None:
        ss.turns.append(history.add(result))
        ss.focus = None
        ss.scroll_top = True
        trace_slot.markdown(trace_html(result, docs), unsafe_allow_html=True)
elif ss.turns:
    index = ss.focus if ss.focus is not None else len(ss.turns) - 1
    trace_slot.markdown(trace_html(entry_to_trace(ss.turns[index]), docs), unsafe_allow_html=True)

if ss.scroll_top:
    ss.scroll_top = False
    # Streamlit tự cuộn xuống cuối khi có ô nhập chat, làm câu hỏi và câu trả lời trôi lên trên.
    components.html(
        "<script>const up = () => { const m = window.parent.document.querySelector("
        "'main.stMain, [data-testid=\"stAppScrollToBottomContainer\"]');"
        " if (m) m.scrollTo({top: 0}); }; up(); setTimeout(up, 250); setTimeout(up, 800);</script>",
        height=0,
    )
