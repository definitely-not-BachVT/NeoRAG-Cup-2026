"""Lịch sử hỏi đáp của chatbot, lưu thành một file JSON cục bộ (mặc định .cache/chat_history.json).

File nằm trong .cache/ (đã .gitignore) nên không bị commit. Mở lại một mục lịch sử chỉ đọc lại
bản đã lưu (câu trả lời, bản viết lại, nguồn, thời gian), không gọi lại pipeline hay API.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from my_solution.trace import Stage, Trace

DEFAULT_PATH = Path(".cache/chat_history.json")
_lock = (
    threading.Lock()
)  # nhiều phiên trình duyệt cùng một tiến trình: đọc-sửa-ghi không được đè nhau
MAX_ENTRIES = 200  # giữ file nhỏ; mục cũ nhất bị bỏ trước


def trace_to_entry(trace: Trace, now: datetime | None = None) -> dict:
    entry = asdict(trace)
    entry["id"] = uuid.uuid4().hex[:12]
    entry["ts"] = (now or datetime.now()).isoformat(timespec="seconds")
    return entry


def entry_to_trace(entry: dict) -> Trace:
    stages = [
        Stage(name=str(s["name"]), seconds=s.get("seconds"), note=str(s.get("note", "")))
        for s in entry.get("stages", [])
        if isinstance(s, dict) and "name" in s
    ]
    return Trace(
        question=entry["question"],
        rewrite=entry.get("rewrite", ""),
        subqueries=list(entry.get("subqueries", [])),
        hits=[(int(i), float(s)) for i, s in entry.get("hits", [])],
        n_context=entry.get("n_context", 0),
        answer=entry.get("answer", ""),
        stages=stages,
        fallback=entry.get("fallback", False),
        traced=entry.get("traced", True),
    )


class History:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)

    def load(self) -> list[dict]:
        """Các mục, mới nhất trước. File hỏng hoặc chưa có thì trả về danh sách rỗng."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        entries = []
        for e in data:
            if isinstance(e, dict) and isinstance(e.get("question"), str) and e.get("id"):
                e.setdefault("answer", "")
                e["ts"] = (
                    e["ts"] if isinstance(e.get("ts"), str) else ""
                )  # mục cũ thiếu giờ vẫn mở được
                entries.append(e)
        return sorted(entries, key=lambda e: e["ts"], reverse=True)

    def _save(self, entries: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)  # ghi nguyên tử: tắt máy giữa chừng không làm hỏng file cũ

    def add(self, trace: Trace) -> dict:
        """Lưu một mục. Nếu ghi file lỗi (khóa file, ổ đĩa đầy) vẫn trả về mục để hiển thị."""
        entry = trace_to_entry(trace)
        with _lock:
            try:
                self._save([entry, *self.load()][:MAX_ENTRIES])
            except OSError:
                entry["unsaved"] = True
        return entry

    def delete(self, entry_id: str) -> None:
        # không xóa được thì mục vẫn còn, không sập app
        with _lock, contextlib.suppress(OSError):
            self._save([e for e in self.load() if e["id"] != entry_id])

    def clear(self) -> None:
        with _lock:
            self._save([])


def day_label(ts: str, today: datetime | None = None) -> str:
    """Nhóm theo ngày: "Hôm nay", "Hôm qua", còn lại là ngày/tháng."""
    today = today or datetime.now()
    try:
        when = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return "Trước đó"
    if when.date() == today.date():
        return "Hôm nay"
    if when.date() == (today - timedelta(days=1)).date():
        return "Hôm qua"
    return when.strftime("%d/%m/%Y")
