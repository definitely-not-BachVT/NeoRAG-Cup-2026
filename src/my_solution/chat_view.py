"""HTML và CSS cho chatbot (app/chat.py): đầu trang, thanh các bước, bản viết lại có tô dấu, nguồn.

Họa tiết xuyên suốt là bốn màu CLB (ProPTIT, Multimedia, GCC, S2B): dải màu ở đầu trang, ô vuông
ở thanh bên, vạch và nhãn của từng nguồn.

Mọi chuỗi từ người dùng, từ LLM hoặc từ tài liệu đều đi qua html.escape trước khi chèn vào HTML,
vì Streamlit hiển thị bằng unsafe_allow_html.
"""

from __future__ import annotations

import html
import re

from my_solution.trace import Stage, Trace, mark_rewrite
from neorag.data import Document

# (từ khóa trong tên CLB, màu, tên ngắn)
CLUBS = [
    ("Lập Trình", "#2F5DA8", "ProPTIT"),
    ("Multimedia", "#9B2C7F", "MPC"),
    ("Game", "#1E8E5A", "GCC"),
    ("S2B", "#5B6472", "S2B"),
]
DEFAULT_COLOR = "#5B6472"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400..700&display=swap');
:root {
  --ink:#14213D; --ink2:#2B3A5E; --paper:#F2F4F7; --card:#FFFFFF; --red:#C8102E;
  --line:#DDE2EA; --muted:#5B6472; --c1:#2F5DA8; --c2:#9B2C7F; --c3:#1E8E5A; --c4:#5B6472;
}
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stChatInput"] {
  font-family: 'Source Sans 3', 'Segoe UI', Arial, sans-serif !important;
  font-feature-settings: "tnum" 1;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding-top: 1.4rem; padding-bottom: 6rem; }

/* ---- đầu trang: bốn màu CLB làm dải trên cùng ---- */
.mast { background: var(--ink); color: #fff; border-radius: 16px; overflow: hidden; margin: 0 0 1.4rem; }
.mast .strip { display: flex; height: 6px; }
.mast .strip i { flex: 1; }
.mast .body { padding: 1.7rem 2rem 1.6rem; }
.mast h1 { font-size: 2.35rem; line-height: 1.12; font-weight: 700; letter-spacing: -0.015em; margin: 0 0 .55rem; color: #fff; padding: 0; }
.mast p { margin: 0; color: #C5CEE0; max-width: 58ch; line-height: 1.55; font-size: 1.05rem; }
.mast .chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.15rem; }
.chip { display: inline-flex; align-items: center; gap: .45rem; padding: .22rem .7rem .22rem .55rem; border-radius: 999px;
  font-size: .9rem; font-weight: 600; background: rgba(255,255,255,.1); color: #fff; }
.chip i { width: .6rem; height: .6rem; border-radius: 50%; background: var(--c); }
.mast.slim .body { padding: .85rem 1.4rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
.mast.slim h1 { font-size: 1.35rem; margin: 0; }
.mast.slim .chips { margin: 0; }
.mast.slim .chip { font-size: .82rem; }

/* ---- thanh bên ---- */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .65rem; margin: .2rem 0 1rem; }
.brand .sq { display: grid; grid-template-columns: 1fr 1fr; gap: 3px; width: 30px; height: 30px; }
.brand .sq i { border-radius: 3px; }
.brand b { font-size: 1.12rem; font-weight: 700; color: var(--ink); line-height: 1.1; }
.brand span { display: block; font-size: .82rem; color: var(--muted); font-weight: 400; }
.day { color: var(--muted); font-size: .85rem; font-weight: 600; margin: 1rem 0 .15rem; }
[data-testid="stSidebar"] .stButton button { border: 0; background: transparent; justify-content: flex-start; text-align: left;
  font-weight: 400; color: var(--ink); padding: .4rem .6rem; border-radius: 8px; min-height: 0; }
[data-testid="stSidebar"] .stButton button:hover { background: #fff; color: var(--ink); }
[data-testid="stSidebar"] .stButton button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: var(--red); color: #fff; justify-content: center; font-weight: 600; padding: .55rem; }
[data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: #A60D26; color: #fff; }
[data-testid="stSidebar"] [data-testid="stColumn"]:last-child .stButton button { color: var(--muted); justify-content: center; padding: .4rem; }
[data-testid="stSidebar"] [data-testid="stColumn"]:last-child .stButton button:hover { color: var(--red); }

/* ---- hội thoại ---- */
[data-testid="stChatMessage"] { border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: .7rem; background: var(--card);
  border: 1px solid var(--line); box-shadow: 0 1px 2px rgba(20,33,61,.04); gap: 0; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessage"] p { margin: 0 0 1rem; line-height: 1.6; font-size: 1.04rem; }
[data-testid="stChatMessage"] { align-items: center; }
[data-testid="stChatMessageContent"] { padding: 0; margin: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--ink); border-color: var(--ink); width: 82%; margin-left: auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { color: #fff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border-left: 4px solid var(--red); }
.hint { color: var(--muted); line-height: 1.55; margin: 0 0 .7rem; }
.stMain .stButton button { text-align: left; white-space: normal; height: auto;
  padding: .75rem 1rem; border-radius: 12px; border: 1px solid var(--line); background: var(--card); color: var(--ink); line-height: 1.45; }
.stMain .stButton button > div { justify-content: flex-start; width: 100%; }
.stMain .stButton button p { text-align: left; white-space: normal; }
.stMain .stButton button:hover { border-color: var(--red); color: var(--ink); background: #fff; }
.stMain .stButton button:focus-visible { outline: 2px solid var(--red); outline-offset: 2px; }
[data-testid="stChatInput"] { border-radius: 14px; border: 1px solid var(--line); background: var(--card); box-shadow: 0 2px 8px rgba(20,33,61,.06); }
[data-testid="stChatInput"]:focus-within { border-color: var(--red); box-shadow: 0 0 0 3px rgba(200,16,46,.12); }

/* ---- cột "Quá trình trả lời" ---- */
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.25rem 1.2rem;
  box-shadow: 0 1px 2px rgba(20,33,61,.04); }
.panel h3 { font-size: 1.12rem; font-weight: 700; margin: 0 0 .15rem; color: var(--ink); padding: 0; }
.panel .total { color: var(--muted); font-size: .92rem; margin: 0 0 .75rem; }
.rail { margin: 0 0 1rem; }
.step { display: grid; grid-template-columns: 26px 1fr auto; column-gap: .65rem; align-items: center; padding: .38rem 0; }
.step .dot { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: .75rem; font-weight: 700;
  background: var(--line); color: var(--muted); }
.step .n { font-size: .95rem; color: var(--ink); line-height: 1.25; }
.step .t { font-size: .95rem; font-weight: 700; color: var(--ink); }
.step .bar { grid-column: 2 / 4; height: 4px; border-radius: 2px; background: #EAEDF2; margin-top: .3rem; overflow: hidden; }
.step .bar i { display: block; height: 100%; background: var(--ink); border-radius: 2px; }
.step.done .dot { background: var(--ink); color: #fff; }
.step.run .dot { background: var(--red); color: #fff; animation: pulse 1.2s ease-out infinite; }
.step.run .bar i { width: 100% !important; background: repeating-linear-gradient(90deg, var(--red) 0 10px, #F2B8C0 10px 20px);
  background-size: 20px 100%; animation: slide .9s linear infinite; }
.step.wait .n, .step.wait .t { color: var(--muted); }
@keyframes slide { from { background-position: 0 0; } to { background-position: 20px 0; } }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(200,16,46,.45); } 100% { box-shadow: 0 0 0 9px rgba(200,16,46,0); } }
@media (prefers-reduced-motion: reduce) { .step.run .dot, .step.run .bar i { animation: none; } }
.q { margin: 0 0 .85rem; line-height: 1.5; }
.q .k { display: block; color: var(--muted); font-size: .86rem; margin-bottom: .1rem; }
.q mark.dau { background: rgba(200,16,46,.13); color: var(--red); padding: 0 .1em; border-radius: 3px; font-weight: 700; }
.q mark.giainghia { background: transparent; color: var(--ink); border-bottom: 1.5px dotted var(--red); }
.legend { color: var(--muted); font-size: .82rem; margin: -.4rem 0 1rem; }
.warn { border-left: 3px solid var(--red); background: #FCEDEF; border-radius: 0 8px 8px 0; padding: .5rem .8rem; margin: 0 0 1rem; font-size: .93rem; }
ul.subq { margin: .1rem 0 .9rem 1.1rem; padding: 0; line-height: 1.5; }
h4.sec { font-size: 1rem; font-weight: 700; margin: 1.1rem 0 .5rem; color: var(--ink); padding: 0; }
.src { border: 1px solid var(--line); border-left: 4px solid var(--c); border-radius: 10px; padding: .5rem .8rem; margin-bottom: .45rem; background: #fff; }
.src summary { cursor: pointer; list-style: none; line-height: 1.4; display: flex; flex-wrap: wrap; align-items: center; gap: .15rem .55rem; }
.src summary::-webkit-details-marker { display: none; }
.src summary:focus-visible { outline: 2px solid var(--red); outline-offset: 3px; }
.src .club { background: var(--c); color: #fff; border-radius: 6px; padding: .02rem .5rem; font-size: .82rem; font-weight: 700; }
.src .topic { font-weight: 600; color: var(--ink); }
.src .tag { color: var(--muted); font-size: .83rem; }
.src p { margin: .5rem 0 .15rem; line-height: 1.6; font-size: .96rem; color: var(--ink2); }
.src[open] { box-shadow: 0 2px 8px rgba(20,33,61,.07); }
.src.spare { opacity: .6; }
.empty-panel { color: var(--muted); line-height: 1.55; }
@media (max-width: 760px) {
  .mast .body { padding: 1.2rem 1.1rem; } .mast h1 { font-size: 1.7rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { width: 94%; margin-left: auto; }
}
</style>
"""


def _club(name: str) -> tuple[str, str]:
    """(màu, tên ngắn) của một CLB theo tên đầy đủ."""
    for key, color, short in CLUBS:
        if key in name:
            return color, short
    return DEFAULT_COLOR, name


def club_color(club: str) -> str:
    return _club(club)[0]


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!$|<>~])")


def escape_md(text: str) -> str:
    """Thoát ký tự markdown/LaTeX để hiển thị nguyên văn (không tạo link, ảnh, công thức từ `$`)."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def masthead_html(compact: bool) -> str:
    """Đầu trang: dải bốn màu CLB, tiêu đề và nhãn các CLB. `compact` khi đã có hội thoại."""
    strip = "".join(f'<i style="background:{c}"></i>' for _, c, _ in CLUBS)
    chips = "".join(
        f'<span class="chip" style="--c:{c}"><i></i>{html.escape(short)}</span>'
        for _, c, short in CLUBS
    )
    if compact:
        return (
            f'<div class="mast slim"><div class="strip">{strip}</div><div class="body">'
            f'<h1>Hỏi đáp cộng đồng S2B</h1><div class="chips">{chips}</div></div></div>'
        )
    return (
        f'<div class="mast"><div class="strip">{strip}</div><div class="body">'
        "<h1>Hỏi đáp cộng đồng S2B</h1>"
        "<p>Thông tin về CLB Lập Trình PTIT, CLB Multimedia PTIT và CLB Nhà Sáng Tạo Game PTIT. "
        "Viết không dấu hay viết tắt đều được, mỗi câu hỏi được trả lời độc lập.</p>"
        f'<div class="chips">{chips}</div></div></div>'
    )


def brand_html() -> str:
    squares = "".join(f'<i style="background:{c}"></i>' for _, c, _ in CLUBS)
    return (
        f'<div class="brand"><div class="sq">{squares}</div>'
        "<div><b>Hỏi đáp S2B</b><span>Lịch sử trò chuyện</span></div></div>"
    )


def _seconds(value: float | None) -> str:
    return "…" if value is None else f"{value:.1f} s"


def rail_html(stages: list[Stage], running: bool) -> str:
    """Các bước dạng danh sách dọc, mỗi bước có thanh dài theo thời gian (thấy bước nào chậm nhất).

    Khi `running`, bước đầu tiên chưa có thời gian được tô là đang chạy.
    """
    active = next((i for i, s in enumerate(stages) if s.seconds is None), None) if running else None
    longest = max((s.seconds or 0.0 for s in stages), default=0.0) or 1.0
    parts = []
    for i, stage in enumerate(stages):
        state = "done" if stage.seconds is not None else ("run" if i == active else "wait")
        width = 0.0 if stage.seconds is None else max(2.0, 100 * stage.seconds / longest)
        shown = _seconds(stage.seconds) if state != "wait" else ""
        parts.append(
            f'<div class="step {state}"><span class="dot">{i + 1}</span>'
            f'<span class="n">{html.escape(stage.name)}</span><span class="t">{shown}</span>'
            f'<span class="bar"><i style="width:{width:.0f}%"></i></span></div>'
        )
    return f'<div class="rail">{"".join(parts)}</div>'


def sources_html(trace: Trace, docs: dict[int, Document]) -> str:
    rows = []
    for rank, (doc_id, _score) in enumerate(trace.hits, 1):
        doc = docs.get(doc_id)
        if doc is None:
            continue
        color, short = _club(doc.club)
        spare = rank > trace.n_context
        tag = "dự phòng" if spare else f"đưa vào câu trả lời, hạng {rank}"
        rows.append(
            f'<details class="src{" spare" if spare else ""}" style="--c:{color}">'
            f'<summary><span class="club">{html.escape(short)}</span>'
            f'<span class="topic">{html.escape(doc.topic)}</span>'
            f'<span class="tag">[{doc.id}] {tag}</span></summary>'
            f"<p>{html.escape(doc.text)}</p></details>"
        )
    return "".join(rows)


def trace_html(trace: Trace, docs: dict[int, Document], running: bool = False) -> str:
    """Toàn bộ cột "Quá trình trả lời"."""
    out = ['<div class="panel"><h3>Quá trình trả lời</h3>']
    if trace.traced and trace.stages:
        if not running:
            out.append(f'<div class="total">Tổng {trace.total:.1f} s</div>')
        else:
            out.append('<div class="total">Đang xử lý…</div>')
        out.append(rail_html(trace.stages, running))
    out.append(f'<div class="q"><span class="k">Câu gốc</span>{html.escape(trace.question)}</div>')
    if trace.rewrite:
        out.append(
            '<div class="q"><span class="k">Viết lại</span>'
            f"{mark_rewrite(trace.question, trace.rewrite)}</div>"
            '<div class="legend">Chữ đỏ: dấu được khôi phục. Gạch chấm: từ được viết đầy đủ.</div>'
        )
    if trace.subqueries:
        items = "".join(f"<li>{html.escape(q)}</li>" for q in trace.subqueries)
        out.append(
            '<div class="q"><span class="k">Tách thành các câu hỏi con</span></div>'
            f'<ul class="subq">{items}</ul>'
        )
    for stage in trace.stages:
        if stage.note:
            out.append(
                f'<div class="warn">{html.escape(stage.name)}: {html.escape(stage.note)}.</div>'
            )
    if trace.hits:
        out.append('<h4 class="sec">Nguồn tham khảo</h4>' + sources_html(trace, docs))
    out.append("</div>")
    return "".join(out)
