"""Phân tích lỗi CÂU TRẢ LỜI trên tập train: sai ở đâu, vì sao. Chạy offline, không gọi API.

eval_retrieval.py cho biết retrieval tốt đến đâu. Script này trả lời câu hỏi tiếp theo:
khi câu trả lời thiếu key fact hoặc ROUGE thấp, lỗi nằm ở bước nào?

    retrieval_miss   đoạn chứa đáp án không nằm trong n_context đoạn đưa vào prompt
                     -> model không thể trả lời đúng; sửa ở retrieval (bước 5, 6)
    missing_fact     đoạn đúng ĐÃ có trong prompt, key fact có trong đoạn đó, nhưng câu trả lời
                     thiếu -> model bỏ sót hoặc viết lại khác đi; sửa prompt (bước 7)
    fact_not_literal key fact không xuất hiện nguyên văn trong các đoạn đưa vào prompt
                     (đáp án mẫu tự suy ra/đổi định dạng) -> khó sửa, chấp nhận
    false_abstain    câu có đáp án nhưng hệ thống lại nói "không có thông tin"
    missed_abstain   câu KHÔNG có đáp án nhưng hệ thống không từ chối
    too_long         câu trả lời dài hơn 2 lần đáp án mẫu -> ROUGE-L và các metric chấm từng câu
                     đều bị kéo xuống

    python scripts/analyze_predictions.py                                   # outputs/train_pred.jsonl
    python scripts/analyze_predictions.py --show 10                         # in 10 câu tệ nhất
    python scripts/analyze_predictions.py --show 5 --only missing_fact      # chỉ một loại lỗi
    python scripts/analyze_predictions.py --predictions outputs/train_v2.jsonl \\
        --baseline outputs/train_pred.jsonl                                 # so sánh hai lần chạy

Chỉ số judge (groundedness, ...) được lấy thêm từ outputs/eval_<tên file>/per_query.jsonl
nếu lần chạy đó có chấm judge.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from my_solution.text import flatten
from neorag.data import DATA_DIR, load_corpus, read_jsonl
from neorag.metrics.lexical import rouge_l, string_presence

# Cách nói "không có thông tin", ở dạng phẳng (không dấu) để khớp mọi cách viết.
ABSTAIN_PHRASES = [
    "khong co thong tin",
    "chua co thong tin",
    "khong du thong tin",
    "khong neu",
    "khong de cap",
    "khong noi",
    "khong cung cap",
]
DIAGNOSES = [
    "retrieval_miss",
    "missing_fact",
    "fact_not_literal",
    "false_abstain",
    "missed_abstain",
    "too_long",
]
JUDGE_METRICS = ["groundedness", "response_relevancy", "noise_sensitivity", "abstention"]


def abstains(answer: str) -> bool:
    flat = flatten(answer)
    return any(p in flat for p in ABSTAIN_PHRASES)


def analyze(gold: dict, pred: dict, docs: dict, n_context: int, judge: dict | None) -> dict:
    """Chấm và chẩn đoán MỘT câu hỏi. Trả về dict gồm metric + danh sách lỗi."""
    answer, reference = pred["answer"], gold["answer"]
    context_ids = pred["retrieved_ids"][:n_context]
    context = " ".join(docs[i].text for i in context_ids if i in docs)
    facts = gold.get("key_facts", [])

    row = {
        "id": gold["id"],
        "type": gold.get("type", ""),
        "query": gold["query"],
        "answer": answer,
        "reference": reference,
        "answerable": gold.get("answerable", True),
        "len_ratio": len(answer) / max(len(reference), 1),
        "missing_facts": [f for f in facts if string_presence(answer, [f]) == 0],
        "not_in_context": sorted(set(gold.get("relevant_ids", [])) - set(context_ids)),
        "diagnoses": [],
    }
    if row["answerable"]:
        row["string_presence"] = string_presence(answer, facts) if facts else None
        row["rouge_l"] = rouge_l(answer, reference)
    for metric in JUDGE_METRICS:
        if judge and metric in judge:
            row[metric] = judge[metric]

    d = row["diagnoses"]
    if not row["answerable"]:
        if not abstains(answer):
            d.append("missed_abstain")
        return row
    if row["not_in_context"]:
        d.append("retrieval_miss")
    if abstains(answer) and row["missing_facts"]:
        d.append("false_abstain")
    for fact in row["missing_facts"]:
        if string_presence(context, [fact]) == 1:
            d.append("missing_fact")  # fact nằm ngay trong prompt mà model không viết ra
        elif not row["not_in_context"]:
            d.append("fact_not_literal")  # đủ đoạn đúng nhưng fact không có nguyên văn
        # còn lại: fact thiếu vì đoạn chứa nó bị trượt -> đã tính vào retrieval_miss
    if row["len_ratio"] > 2:
        d.append("too_long")
    row["diagnoses"] = list(dict.fromkeys(d))  # bỏ trùng, giữ thứ tự
    return row


def load_rows(pred_path: Path, golds: dict, docs: dict, n_context: int) -> list[dict]:
    judge_path = pred_path.parent / f"eval_{pred_path.stem}" / "per_query.jsonl"
    judged = {r["id"]: r for r in read_jsonl(judge_path)} if judge_path.exists() else {}
    preds = read_jsonl(pred_path)
    return [
        analyze(golds[p["id"]], p, docs, n_context, judged.get(p["id"]))
        for p in preds
        if p["id"] in golds
    ]


def summarize(rows: list[dict]) -> dict:
    """Trung bình metric của một nhóm câu (bỏ qua câu không có metric đó)."""
    out = {"n": len(rows)}
    for m in ["string_presence", "rouge_l", *JUDGE_METRICS]:
        values = [r[m] for r in rows if r.get(m) is not None]
        out[m] = mean(values) * 100 if values else None
    out["len_ratio"] = mean(r["len_ratio"] for r in rows)
    out["errors"] = Counter(d for r in rows for d in r["diagnoses"])
    return out


def fmt(value: float | None) -> str:
    return "   -" if value is None else f"{value:5.1f}"


def print_table(groups: dict[str, list[dict]], baseline: dict[str, list[dict]] | None) -> None:
    print(
        f"{'nhóm':<22} {'n':>3}  {'SP':>5} {'ROUGE':>5} {'dài/mẫu':>7}  "
        + "  ".join(f"{d[:12]:>12}" for d in DIAGNOSES)
    )
    for key in sorted(groups, key=lambda k: (k != "TẤT CẢ", k)):
        s = summarize(groups[key])
        line = (
            f"{key:<22} {s['n']:>3}  {fmt(s['string_presence'])} {fmt(s['rouge_l'])} "
            f"{s['len_ratio']:7.2f}  " + "  ".join(f"{s['errors'][d]:>12}" for d in DIAGNOSES)
        )
        if baseline and key in baseline:
            b = summarize(baseline[key])
            delta = [
                (s[m] or 0) - (b[m] or 0)
                for m in ("string_presence", "rouge_l")
                if s[m] is not None and b[m] is not None
            ]
            line += "   Δ SP/ROUGE " + "/".join(f"{x:+.1f}" for x in delta)
        print(line)

    judged = summarize(groups["TẤT CẢ"])
    if any(judged[m] is not None for m in JUDGE_METRICS):
        print("\nJudge: " + "  ".join(f"{m} {fmt(judged[m])}" for m in JUDGE_METRICS))


def group(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list, {"TẤT CẢ": rows})
    for r in rows:
        groups[r["type"] or "(trống)"].append(r)
    return groups


def badness(row: dict) -> float:
    """Câu càng tệ điểm càng cao: thiếu key fact nặng nhất, rồi đến ROUGE thấp."""
    if not row["answerable"]:
        return 2.0 if row["diagnoses"] else 0.0
    sp = row["string_presence"] if row["string_presence"] is not None else 1.0
    return (1 - sp) * 2 + (1 - row["rouge_l"])


def show(rows: list[dict], n: int, docs: dict) -> None:
    for r in sorted(rows, key=badness, reverse=True)[:n]:
        print(f"\n[{r['id']}] ({r['type']})  lỗi: {', '.join(r['diagnoses']) or 'không'}")
        print(f"  HỎI : {r['query']}")
        print(f"  ĐÁP : {r['answer']}")
        print(f"  MẪU : {r['reference']}")
        if r["missing_facts"]:
            print(f"  thiếu key fact: {r['missing_facts']}")
        if r["not_in_context"]:
            names = [f"{i} ({docs[i].topic})" for i in r["not_in_context"]]
            print(f"  đoạn đúng không có trong prompt: {names}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", default="outputs/train_pred.jsonl")
    parser.add_argument("--baseline", help="file predictions của lần chạy trước để so sánh")
    parser.add_argument("--questions", default=str(DATA_DIR / "train" / "questions.jsonl"))
    parser.add_argument("--n-context", type=int, default=8, help="khớp MyPipeline.n_context")
    parser.add_argument("--show", type=int, default=0, help="in N câu tệ nhất")
    parser.add_argument("--only", choices=DIAGNOSES, help="chỉ in các câu có loại lỗi này")
    args = parser.parse_args()

    docs = {d.id: d for d in load_corpus()}
    golds = {g["id"]: g for g in read_jsonl(args.questions)}
    rows = load_rows(Path(args.predictions), golds, docs, args.n_context)
    baseline = None
    if args.baseline:
        baseline = group(load_rows(Path(args.baseline), golds, docs, args.n_context))

    print(f"Predictions: {args.predictions}  ({len(rows)} câu)\n")
    print_table(group(rows), baseline)

    if args.show:
        selected = [r for r in rows if not args.only or args.only in r["diagnoses"]]
        show(selected, args.show, docs)


if __name__ == "__main__":
    main()
