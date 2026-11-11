"""Xử lý văn bản tiếng Việt cho bước truy xuất.

Vấn đề cần giải quyết: câu hỏi của người dùng thường viết không dấu, viết tắt, gọi CLB bằng
biệt danh ("clb lam game cua hv"), trong khi tài liệu viết có dấu, đầy đủ, dùng tên chính thức
("CLB Nhà Sáng Tạo Game PTIT ... Học viện"). Module này đưa cả hai về cùng một dạng để so khớp.

Luồng xử lý:
    văn bản gốc
      -> flatten()               chữ thường, bỏ dấu, bỏ dấu câu     "Đa phương tiện!" -> "da phuong tien"
      -> expand_abbreviations()  giải nghĩa viết tắt theo từng token "hv" -> "hoc vien"
      -> tokenize()              tách token + ghép bigram cho BM25
    detect_clubs()  nhận diện CLB được nhắc tới trong câu hỏi
    expand_query()  câu hỏi gốc + bản đã giải nghĩa + tên chính thức của CLB
"""

from __future__ import annotations

import re
import unicodedata

from neorag.text import normalize  # NFC + chữ thường + thống nhất vị trí dấu thanh (hoà = hòa)

# ----------------------------------------------------------------------------------------------
# 1. Tên CLB: viết ĐÚNG như trường "club" trong data/corpus.jsonl để so sánh trực tiếp được.
# ----------------------------------------------------------------------------------------------

PROPTIT = "CLB Lập Trình PTIT"
MULTIMEDIA = "CLB Multimedia PTIT"
GCC = "CLB Nhà Sáng Tạo Game PTIT (GCC)"
S2B = "Cộng đồng S2B"
CLUBS = [PROPTIT, MULTIMEDIA, GCC, S2B]


# ----------------------------------------------------------------------------------------------
# 2. Chuẩn hóa: bỏ dấu và làm phẳng văn bản
# ----------------------------------------------------------------------------------------------


def strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt: "Đa phương tiện" -> "Da phuong tien".

    Cách làm: tách mỗi chữ có dấu thành chữ gốc + các dấu rời (dạng NFD),
    rồi bỏ các dấu rời (Unicode category "Mn" = Mark, nonspacing).
        "ệ" --NFD--> "e" + dấu mũ + dấu nặng --bỏ Mn--> "e"
    Riêng "đ" là một chữ cái độc lập (không phải "d" + dấu) nên phải thay tay.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)  # mọi ký tự không phải chữ/số/khoảng trắng


def flatten(text: str) -> str:
    """Dạng "phẳng" dùng cho mọi phép so khớp.

    "Tuyển thành viên ĐỢT 2026 (Gen 14)!" -> "tuyen thanh vien dot 2026 gen 14"

    Thứ tự: normalize (NFC, chữ thường) -> bỏ dấu -> thay dấu câu bằng khoảng trắng
    -> gộp khoảng trắng. Thay dấu câu bằng KHOẢNG TRẮNG (không xóa hẳn) để "vay,thi"
    thành "vay thi" chứ không dính thành "vaythi".
    """
    text = strip_accents(normalize(text))
    return " ".join(_NON_WORD.sub(" ", text).split())


# ----------------------------------------------------------------------------------------------
# 3. Viết tắt / teencode
# ----------------------------------------------------------------------------------------------

# Khóa và giá trị đều ở dạng phẳng (không dấu, chữ thường). Tra theo TỪNG TOKEN, không thay
# chuỗi con: nếu thay chuỗi con thì "k" trong "k26" hay "tv" trong "tvien" sẽ bị thay nhầm.
ABBREVIATIONS: dict[str, str] = {
    # đại từ, hư từ hay viết tắt
    "e": "em",
    "mk": "minh",
    "ko": "khong", "k": "khong", "kh": "khong", "hok": "khong", "hong": "khong",
    "dc": "duoc",
    "dk": "dang ky", "dky": "dang ky", "dki": "dang ky",
    "j": "gi",
    "v": "vay", "z": "vay",
    "bn": "bao nhieu",
    "ntn": "nhu the nao",
    # danh từ trong ngữ cảnh CLB / trường học
    "hv": "hoc vien",
    "sv": "sinh vien",
    "tv": "thanh vien",
    "bcn": "ban chu nhiem",
    "cn": "chu nhiem",
    "nk": "nhiem ky",
    "dpt": "da phuong tien",
    "khmt": "khoa hoc may tinh",
    "cntt": "cong nghe thong tin",
    "attt": "an toan thong tin",
    "fb": "facebook",
    # tháng: "t8" -> "thang 8"
    **{f"t{i}": f"thang {i}" for i in range(1, 13)},
}  # fmt: skip


def expand_abbreviations(flat: str) -> str:
    """Giải nghĩa viết tắt trên chuỗi ĐÃ flatten, thay theo từng token.

    "link dk cua mm" -> "link dang ky cua mm"
    "e muon vao clb ma van di lam them dc ko" -> "em muon vao clb ma van di lam them duoc khong"
    """
    return " ".join(ABBREVIATIONS.get(tok, tok) for tok in flat.split())


# ----------------------------------------------------------------------------------------------
# 4. Nhận diện CLB
# ----------------------------------------------------------------------------------------------

# Các cách gọi tên từng CLB, ở dạng phẳng SAU KHI đã expand_abbreviations
# (vì vậy viết "hoc vien" chứ không viết "hv"). So khớp nguyên cụm, có khoảng trắng hai đầu,
# nên "mpc" không khớp nhầm với "mpcx" và "clb game" không khớp với "clb gamer".
CLUB_ALIASES: dict[str, list[str]] = {
    PROPTIT: [
        "proptit", "pro ptit", "programming ptit",
        "clb lap trinh", "cau lac bo lap trinh", "lap trinh ptit",
        "clb code", "clb lt",
    ],
    MULTIMEDIA: [
        "multimedia", "mpc",
        "clb mm", "ben mm", "cua mm", "o mm", "vao mm",
        "da phuong tien", "clb media",
    ],
    GCC: [
        "gcc", "game creators", "nha sang tao game",
        "clb game", "clb lam game", "cau lac bo game", "cau lac bo lam game",
    ],
    S2B: [
        "s2b", "3 clb", "ba clb", "ca 3 clb", "cong dong chung",
    ],
}  # fmt: skip


def detect_clubs(query: str) -> list[str]:
    """Các CLB được nhắc tới trong câu hỏi, theo thứ tự trong CLUBS.

    "clb lam game cua hv co cho nao rieng ko"        -> [GCC]
    "nhom lam game ben clb lap trinh ptit ... clb game" -> [PROPTIT, GCC]
    "Học lập trình có cần giỏi toán không?"          -> []   (không chắc chắn thì không đoán)

    Trả về danh sách vì một câu có thể nhắc nhiều CLB (câu so sánh).
    """
    padded = f" {expand_abbreviations(flatten(query))} "
    return [club for club in CLUBS if any(f" {alias} " in padded for alias in CLUB_ALIASES[club])]


# ----------------------------------------------------------------------------------------------
# 5. Tách token cho BM25 và mở rộng câu hỏi
# ----------------------------------------------------------------------------------------------


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    """Token cho BM25: âm tiết đã flatten + giải nghĩa viết tắt, kèm bigram âm tiết.

    "Tuyển thành viên" -> ["tuyen", "thanh", "vien", "tuyen_thanh", "thanh_vien"]

    Tiếng Việt viết cách theo ÂM TIẾT, nhưng từ thường gồm 2 âm tiết. Chỉ dùng âm tiết thì
    "thanh vien" và "thanh lap" cùng khớp chữ "thanh". Bigram "thanh_vien" giữ lại được
    đơn vị từ mà không cần công cụ tách từ (pyvi, underthesea).
    """
    unigrams = expand_abbreviations(flatten(text)).split()
    if not bigrams:
        return unigrams
    pairs = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:], strict=False)]
    return unigrams + pairs


def expand_query(query: str) -> str:
    """Câu hỏi mở rộng = câu gốc + bản đã giải nghĩa viết tắt + tên chính thức của CLB.

    "clb da phuong tien tuyen tv ko"
      -> "clb da phuong tien tuyen tv ko | clb da phuong tien tuyen thanh vien khong
          | CLB Multimedia PTIT"

    - Giữ câu gốc: embedding vẫn thấy được cách diễn đạt nguyên bản.
    - Thêm bản đã giải nghĩa: BM25 khớp được "thanh vien" trong tài liệu.
    - Thêm tên chính thức: đoạn văn nào cũng ghi tên CLB đầy đủ, nên tên chính thức giúp
      cả BM25 lẫn embedding kéo đúng CLB lên. S2B không thêm, vì câu hỏi về S2B thường cần
      đoạn của từng CLB chứ không chỉ đoạn chung của S2B.
    """
    expanded = expand_abbreviations(flatten(query))
    clubs = [c for c in detect_clubs(query) if c != S2B]
    return " | ".join([query, expanded, *clubs])
