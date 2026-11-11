"""Kiểm tra file predictions trước khi nộp (chạy được cả ngày Private Test, không gọi API).

    python scripts/check_submission.py private_test            # outputs/private_test_pred.jsonl
    python scripts/check_submission.py public_test --pred outputs/public_test_v9.jsonl

Báo lỗi (thoát mã 1) nếu: thiếu/thừa câu hỏi, id đoạn không có trong corpus hoặc trùng, không đủ
10 id, câu trả lời rỗng. Cảnh báo (không chặn) nếu có câu dùng câu dự phòng. In thêm thời gian trả lời và số câu nghi ngờ để
quyết định có chạy lại không (ví dụ nhiều câu mất hơn 60 giây).
"""

from __future__ import annotations

import argparse
import sys
from statistics import median

from my_solution.pipeline import FALLBACK_ANSWER
from neorag.data import DATA_DIR, load_corpus, read_jsonl

OUTPUT_DIR = DATA_DIR.parent / "outputs"
N_IDS = 10  # số id nộp cho mỗi câu (configs/eval.toml: max_retrieved)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", choices=["train", "public_test", "private_test"])
    parser.add_argument("--pred", help="file predictions (mặc định outputs/<bộ>_pred.jsonl)")
    args = parser.parse_args()

    pred_path = args.pred or OUTPUT_DIR / f"{args.split}_pred.jsonl"
    questions = [r["id"] for r in read_jsonl(DATA_DIR / args.split / "questions.jsonl")]
    preds = {r["id"]: r for r in read_jsonl(pred_path)}
    valid_ids = {d.id for d in load_corpus()}

    errors: list[str] = []
    if set(preds) != set(questions):
        errors.append(
            f"id câu hỏi không khớp: thiếu {len(set(questions) - set(preds))}, "
            f"thừa {len(set(preds) - set(questions))}"
        )
    short, fallback = [], []
    for qid, row in preds.items():
        ids = row.get("retrieved_ids", [])
        if len(ids) != len(set(ids)) or not set(ids) <= valid_ids:
            errors.append(f"{qid}: id đoạn trùng hoặc không có trong corpus")
        if len(ids) < N_IDS:
            short.append(qid)
        answer = (row.get("answer") or "").strip()
        if not answer:
            errors.append(f"{qid}: câu trả lời rỗng")
        elif answer == FALLBACK_ANSWER:
            fallback.append(qid)  # LLM trả rỗng nên pipeline dùng câu dự phòng: nên chạy lại
    if short:
        errors.append(f"{len(short)} câu có ít hơn {N_IDS} id (ví dụ {short[0]})")

    latencies = [r["latency_ms"] / 1000 for r in preds.values() if "latency_ms" in r]
    print(f"{pred_path}: {len(preds)}/{len(questions)} câu")
    if latencies:
        slow = sum(x > 60 for x in latencies)
        print(
            f"thời gian mỗi câu: median {median(latencies):.1f}s, lớn nhất {max(latencies):.1f}s, "
            f"{slow} câu quá 60s, tổng {sum(latencies) / 60:.1f} phút"
        )
    if fallback:
        print(f"CẢNH BÁO: {len(fallback)} câu dùng câu trả lời dự phòng: {', '.join(fallback[:5])}")
    if errors:
        print("\nKHÔNG nộp, có lỗi:")
        print("\n".join(f"  - {e}" for e in errors[:20]))
        return 1
    print("Hợp lệ. Có thể nộp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
