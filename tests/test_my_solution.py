"""Unit test cho src/my_solution. Chạy hoàn toàn offline: LLM, embedding và reranker đều là bản giả.

    pytest tests/test_my_solution.py -q

Mỗi test kiểm tra MỘT hành vi đã được thiết kế có chủ đích (ghi trong docstring của code),
để khi sửa code sau này mà vô tình phá hành vi đó thì test báo ngay.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeEmbedder
from my_solution.corpus_rank import CorpusRanker, parse_corpus_ranking
from my_solution.enrich import DocEnricher, Enrichment, parse_enrichment
from my_solution.llm_rerank import ListwiseReranker, parse_ranking
from my_solution.pipeline import FALLBACK_ANSWER, MAX_ANSWER_CHARS, MyPipeline, clean_answer
from my_solution.prompts import SYSTEM_PROMPTS, build_messages
from my_solution.query import REWRITE_PROMPT, QueryPlan, QueryRewriter, parse_plan
from my_solution.rerank import RerankedRetriever
from my_solution.retrievers import (
    BM25,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    MultiQueryRetriever,
    rrf,
)
from my_solution.stepback import STEPBACK_PROMPT, StepBack, parse_stepback
from my_solution.text import (
    GCC,
    MULTIMEDIA,
    PROPTIT,
    S2B,
    detect_clubs,
    expand_abbreviations,
    expand_query,
    flatten,
    strip_accents,
    tokenize,
)
from my_solution.trace import mark_rewrite, run_traced
from neorag.data import Document

# ----------------------------------------------------------------------------------------------
# Đồ giả (fake) dùng chung
# ----------------------------------------------------------------------------------------------


class FakeGateway:
    """Thay cho LLMClient: chat trả lời theo hàm `responder`, embed dùng FakeEmbedder."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda messages: "")
        self.chats: list[list[dict]] = []
        self._embedder = FakeEmbedder()

    def chat(self, messages, model, temperature=None):
        self.chats.append(messages)
        return self.responder(messages)

    def embed(self, texts, model, batch_size=32):
        return self._embedder.embed(texts).tolist()


class FakeReranker:
    """Thay cho CrossEncoderReranker: điểm = số âm tiết chung giữa câu hỏi và đoạn văn."""

    def __init__(self):
        self.queries: list[str] = []

    def load(self):
        return None

    def scores(self, query, docs):
        self.queries.append(query)
        q = set(flatten(query).split())
        return [len(q & set(flatten(d.text).split())) for d in docs]


@pytest.fixture
def clubs_corpus() -> list[Document]:
    return [
        Document(1, PROPTIT, "Giới thiệu", "CLB Lập Trình PTIT thành lập ngày 09/10/2011."),
        Document(2, MULTIMEDIA, "Giới thiệu", "CLB Multimedia PTIT thành lập ngày 10/11/2012."),
        Document(3, GCC, "Giới thiệu", "CLB Nhà Sáng Tạo Game PTIT ra đời năm 2023."),
        Document(4, S2B, "Giới thiệu", "S2B là cộng đồng chung của 3 CLB, gồm nhiều CLB."),
        Document(5, PROPTIT, "Tuyển thành viên", "Đăng ký tuyển thành viên qua link form."),
    ]


def hybrid(fake: FakeGateway) -> HybridRetriever:
    return HybridRetriever(dense=DenseRetriever(llm=fake, model="fake"))


# ----------------------------------------------------------------------------------------------
# text.py
# ----------------------------------------------------------------------------------------------


def test_strip_accents_handles_d_bar():
    assert strip_accents("Đa phương tiện") == "Da phuong tien"


def test_flatten_replaces_punctuation_with_space():
    assert flatten("Tuyển thành viên ĐỢT 2026 (Gen 14)!") == "tuyen thanh vien dot 2026 gen 14"
    assert flatten("vay,thi") == "vay thi"  # không dính thành "vaythi"


def test_expand_abbreviations_is_token_wise():
    assert expand_abbreviations("link dk cua mm") == "link dang ky cua mm"
    assert expand_abbreviations("k26 k") == "k26 khong"  # "k" trong "k26" giữ nguyên


@pytest.mark.parametrize(
    "query, expected",
    [
        ("clb lam game cua hv co cho nao rieng ko", [GCC]),
        ("cho e xin mail clb mm ptit voi a", [MULTIMEDIA]),
        ("ProPTIT với GCC khác nhau thế nào?", [PROPTIT, GCC]),
        ("Trong 3 CLB thuộc S2B, CLB nào lâu đời nhất?", [S2B]),
        ("Học lập trình có cần giỏi toán không?", []),  # không chắc thì không đoán
        ("mmm hay qua", []),  # so khớp nguyên cụm, "mmm" không phải "mm"
    ],
)
def test_detect_clubs(query, expected):
    assert detect_clubs(query) == expected


def test_tokenize_adds_bigrams():
    assert tokenize("Tuyển thành viên") == [
        "tuyen", "thanh", "vien", "tuyen_thanh", "thanh_vien",
    ]  # fmt: skip
    assert tokenize("Tuyển thành viên", bigrams=False) == ["tuyen", "thanh", "vien"]


def test_expand_query_adds_official_name_but_not_s2b():
    assert expand_query("clb mm tuyen tv ko").endswith(f"| {MULTIMEDIA}")
    assert S2B not in expand_query("cong dong s2b co may clb")


# ----------------------------------------------------------------------------------------------
# retrievers.py
# ----------------------------------------------------------------------------------------------


def test_bm25_rare_term_wins_and_idf_is_positive():
    bm25 = BM25().fit([["clb", "tiktok"], ["clb", "youtube"], ["clb", "facebook"]])
    scores = bm25.scores(["tiktok"])
    assert scores[0] > 0 and scores[1] == scores[2] == 0
    assert bm25.idf["clb"] > 0  # token có ở mọi đoạn vẫn không âm (biến thể Lucene)


def test_bm25_retriever_matches_unaccented_query(clubs_corpus):
    retriever = BM25Retriever()
    retriever.index(clubs_corpus)
    hits = retriever.search("clb lap trinh ptit thanh lap ngay nao", top_n=3)
    assert hits[0][0] == 1
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_rrf_rewards_agreement_between_rankings():
    fused = [doc for doc, _ in rrf([[1, 2, 3], [4, 2, 5]])]
    assert fused[0] == 2  # hạng 2 ở CẢ HAI bảng (2/62) thắng hạng 1 ở MỘT bảng (1/61)
    assert len(fused) == len(set(fused)) == 5  # không mất, không trùng doc nào


def test_rrf_weights():
    assert rrf([[1], [2]], weights=[2.0, 1.0])[0][0] == 1
    assert rrf([[1], [2]], weights=[1.0, 2.0])[0][0] == 2


def test_hybrid_returns_unique_ids(clubs_corpus):
    retriever = hybrid(FakeGateway())
    retriever.index(clubs_corpus)
    ids = [doc for doc, _ in retriever.search("multimedia thanh lap khi nao", top_n=10)]
    assert len(ids) == len(set(ids)) == len(clubs_corpus)


def test_multi_query_covers_every_subquery(clubs_corpus):
    retriever = MultiQueryRetriever(hybrid(FakeGateway()), rewriter=None)
    retriever.index(clubs_corpus)
    plan = QueryPlan(
        "3 clb s2b clb nao lau doi nhat",
        "Trong 3 CLB thuộc S2B, CLB nào lâu đời nhất?",
        [
            "CLB Lập Trình PTIT thành lập năm nào?",
            "CLB Multimedia PTIT thành lập năm nào?",
            "CLB Nhà Sáng Tạo Game PTIT ra đời năm nào?",
        ],
    )
    top4 = {doc for doc, _ in retriever.search_plan(plan, top_n=4)}
    assert {1, 2, 3} <= top4  # đoạn của từng CLB đều có mặt, không bị đoạn S2B lấn át


def test_multi_query_without_rewriter_uses_original_query():
    retriever = MultiQueryRetriever(rewriter=None, base=HybridRetriever(dense=object()))
    assert retriever.plan("abc").queries == ["abc"]


# ----------------------------------------------------------------------------------------------
# query.py
# ----------------------------------------------------------------------------------------------


def test_parse_plan_reads_fenced_json_and_drops_junk():
    reply = '```json\n{"rewrite": "Câu có dấu?", "subqueries": ["A?", "", 3, "B?"]}\n```'
    assert parse_plan("q", reply) == QueryPlan("q", "Câu có dấu?", ["A?", "B?"])


@pytest.mark.parametrize(
    "reply",
    ["", "xin lỗi, tôi không hiểu", "{không phải json}", "[1, 2]", '{"rewrite": ""}'],
)
def test_parse_plan_falls_back_to_original(reply):
    assert parse_plan("q", reply) == QueryPlan("q", "q", [])


def test_parse_plan_caps_subqueries():
    reply = json.dumps({"rewrite": "r", "subqueries": [f"s{i}?" for i in range(10)]})
    assert len(parse_plan("q", reply).subqueries) == 4


def test_query_plan_queries_are_deduplicated_in_order():
    assert QueryPlan("a", "a", ["b", "a", "c"]).queries == ["a", "b", "c"]


def test_rewriter_survives_llm_error():
    def broken(messages):
        raise RuntimeError("gateway down")

    plan = QueryRewriter(llm=FakeGateway(broken), model="fake").plan("clb mm o dau")
    assert plan == QueryPlan("clb mm o dau", "clb mm o dau", [])


# ----------------------------------------------------------------------------------------------
# rerank.py
# ----------------------------------------------------------------------------------------------


def test_reranker_scores_with_rewrite_and_subqueries(clubs_corpus):
    reranker = FakeReranker()
    retriever = RerankedRetriever(
        MultiQueryRetriever(hybrid(FakeGateway()), rewriter=None), reranker=reranker
    )
    retriever.index(clubs_corpus)
    plan = QueryPlan("goc khong dau", "Câu viết lại?", ["Câu con 1?", "Câu con 2?"])
    hits = retriever.search_plan(plan, top_n=10)

    assert reranker.queries == ["Câu viết lại?", "Câu con 1?", "Câu con 2?"]  # không dùng câu gốc
    ids = [doc for doc, _ in hits]
    assert len(ids) == len(set(ids)) == len(clubs_corpus)


# ----------------------------------------------------------------------------------------------
# pipeline.py
# ----------------------------------------------------------------------------------------------


def test_clean_answer():
    assert clean_answer("**CLB** thành lập\n\nnăm 2011.") == "CLB thành lập năm 2011."
    assert clean_answer("  \n ") == FALLBACK_ANSWER
    long = clean_answer("chữ " * 1000)
    assert 0 < len(long) <= MAX_ANSWER_CHARS and not long.endswith(" ")


def _responder(messages):
    if messages[0]["content"] == REWRITE_PROMPT:
        return '{"rewrite": "CLB Multimedia PTIT thành lập khi nào?", "subqueries": []}'
    return "**CLB Multimedia PTIT** thành lập ngày 10/11/2012."


def test_pipeline_end_to_end_offline(clubs_corpus):
    fake = FakeGateway(_responder)
    retriever = RerankedRetriever(
        MultiQueryRetriever(hybrid(fake), rewriter=QueryRewriter(llm=fake, model="fake")),
        reranker=FakeReranker(),
    )
    pipeline = MyPipeline(n_context=3, llm=fake, llm_model="fake", retriever=retriever)
    pipeline.prepare()
    pipeline.index(clubs_corpus)
    out = pipeline.run("clb mm thanh lap khi nao")

    assert out.answer == "CLB Multimedia PTIT thành lập ngày 10/11/2012."  # đã bỏ markdown
    assert out.retrieved_ids[0] == 2
    assert len(out.retrieved_ids) == len(set(out.retrieved_ids)) == len(clubs_corpus)
    assert len(fake.chats) == 2  # đúng 1 lời gọi viết lại + 1 lời gọi sinh câu trả lời
    prompt = fake.chats[1][1]["content"]
    assert "Diễn giải câu hỏi: CLB Multimedia PTIT thành lập khi nào?" in prompt
    assert prompt.count("[") == 3  # chỉ n_context đoạn được đưa vào prompt


def test_pipeline_constructs_without_arguments(monkeypatch, tmp_path):
    """load_pipeline gọi MyPipeline() không tham số; khởi tạo không được gọi mạng hay nạp model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # file cache .cache/ được tạo trong thư mục tạm
    pipeline = MyPipeline()
    assert isinstance(pipeline.retriever, CorpusRanker)  # A5 bật mặc định, bọc A4
    assert isinstance(pipeline.retriever.base, ListwiseReranker)
    assert isinstance(pipeline.retriever.base.base, RerankedRetriever)
    assert (
        pipeline.retriever.base.base.reranker._model is None
    )  # model nặng chỉ nạp trong prepare()


def test_prompt_versions(clubs_corpus):
    """v1 (bản đã nộp) vẫn dùng được để so sánh; phiên bản lạ bị từ chối ngay khi khởi tạo."""
    v1 = build_messages("q", clubs_corpus[:1], "03/11/2026", version="v1")
    v2 = build_messages("q", clubs_corpus[:1], "03/11/2026", version="v2")
    assert v1[0]["content"] != v2[0]["content"] and v1[1] == v2[1]  # chỉ system prompt khác
    for version in SYSTEM_PROMPTS:  # không còn chỗ {..} nào chưa được điền
        assert "{" not in build_messages("q", [], "x", version=version)[0]["content"]
    with pytest.raises(ValueError):
        MyPipeline(prompt_version="v9", llm=FakeGateway(), retriever=object())


# ----------------------------------------------------------------------------------------------
# enrich.py (A3 – doc2query)
# ----------------------------------------------------------------------------------------------


def test_parse_enrichment_reads_json_and_survives_junk():
    reply = '```json\n{"questions": ["Hỏi 1?", "", 5, "Hỏi 2?"], "keywords": ["a", "b"]}\n```'
    e = parse_enrichment(reply)
    assert e.questions == ["Hỏi 1?", "Hỏi 2?"] and e.keywords == ["a", "b"]
    assert e.text == "Hỏi 1? | Hỏi 2? | a | b"
    for junk in ["", "không phải json", "[1, 2]", '{"questions": "abc"}']:
        assert parse_enrichment(junk) == Enrichment()


def test_enricher_survives_llm_error(clubs_corpus):
    def broken(messages):
        raise RuntimeError("gateway down")

    enriched = DocEnricher(llm=FakeGateway(broken), model="fake").enrich(clubs_corpus)
    assert set(enriched) == {d.id for d in clubs_corpus}
    assert all(e == Enrichment() for e in enriched.values())  # lỗi thì đoạn dùng văn bản gốc


class FixedEnricher:
    """Enricher giả: gắn câu hỏi cho đúng một đoạn."""

    def __init__(self, doc_id, questions):
        self.doc_id, self.questions = doc_id, questions

    def enrich(self, corpus):
        return {d.id: Enrichment(self.questions if d.id == self.doc_id else []) for d in corpus}


def test_enrichment_lets_paraphrased_question_find_doc(clubs_corpus):
    query = "zzz qqq xxx"  # không khớp từ nào trong các đoạn gốc
    plain = hybrid(FakeGateway())
    plain.index(clubs_corpus)
    assert plain.bm25.search(query, top_n=1)[0][1] == 0  # BM25 không tìm được gì

    enriched = HybridRetriever(
        dense=DenseRetriever(llm=FakeGateway(), model="fake"),
        enricher=FixedEnricher(5, ["zzz qqq xxx"]),
    )
    enriched.index(clubs_corpus)
    assert enriched.bm25.search(query, top_n=1)[0][0] == 5  # câu hỏi sinh sẵn khớp nguyên văn
    assert enriched.dense.search(query, top_n=1)[0][0] == 5  # vector câu hỏi kéo đoạn 5 lên


def test_enricher_stores_separately_and_reuses(clubs_corpus, tmp_path):
    store = tmp_path / "doc2query.jsonl"
    reply = '{"questions": ["Hỏi?"], "keywords": ["k"]}'
    fake = FakeGateway(lambda m: reply)
    first = DocEnricher(llm=fake, model="fake", store_path=store).enrich(clubs_corpus)
    assert store.exists() and len(fake.chats) == len(clubs_corpus)
    assert first[1] == Enrichment(["Hỏi?"], ["k"])

    again = DocEnricher(llm=fake, model="fake", store_path=store).enrich(clubs_corpus)
    assert again == first and len(fake.chats) == len(clubs_corpus)  # đọc file, không gọi LLM

    changed = [Document(1, clubs_corpus[0].club, "Khác", "Nội dung mới."), *clubs_corpus[1:]]
    DocEnricher(llm=fake, model="fake", store_path=store).enrich(changed)
    assert len(fake.chats) == len(clubs_corpus) + 1  # chỉ đoạn đổi nội dung được sinh lại


# ----------------------------------------------------------------------------------------------
# llm_rerank.py (A4)
# ----------------------------------------------------------------------------------------------


class StubBase:
    """Retriever giả trả về thứ tự cố định, thay cho RerankedRetriever."""

    def __init__(self, order):
        self.order = order
        self.docs = {}

    def index(self, corpus):
        self.docs = {d.id: d for d in corpus}

    def plan(self, query):
        return QueryPlan(query, query, [])

    def search_plan(self, plan, top_n=10):
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(self.order)][:top_n]


def _listwise(clubs_corpus, responder, order=(1, 2, 3, 4, 5), **kw):
    fake = FakeGateway(responder)
    ranker = ListwiseReranker(StubBase(list(order)), llm=fake, model="fake", **kw)
    ranker.index(clubs_corpus)
    return ranker, fake


def test_parse_ranking_keeps_valid_ids_and_fills_missing():
    valid = [1, 2, 3, 4]
    assert parse_ranking('{"ranking": [3, 1, 2, 4]}', valid) == [3, 1, 2, 4]
    # id lạ, trùng, chuỗi, bool bị bỏ; id bị sót được bổ sung theo thứ tự cũ
    assert parse_ranking('{"ranking": [3, 99, 3, "1", true, 4]}', valid) == [3, 4, 1, 2]
    assert parse_ranking('```json\n{"ranking": [2]}\n```', valid) == [2, 1, 3, 4]


@pytest.mark.parametrize(
    "reply", ["", "không phải json", "{}", '{"ranking": "1,2"}', '{"ranking": [99]}']
)
def test_parse_ranking_returns_none_on_junk(reply):
    assert parse_ranking(reply, [1, 2, 3]) is None


def test_listwise_promotes_doc_llm_prefers(clubs_corpus):
    ranker, fake = _listwise(clubs_corpus, lambda m: '{"ranking": [5, 4, 3, 2, 1]}', weights=(1, 3))
    ids = [d for d, _ in ranker.search("câu hỏi", top_n=5)]
    assert ids[0] == 5 and sorted(ids) == [1, 2, 3, 4, 5]
    assert len(fake.chats) == 1  # đúng 1 lời gọi LLM mỗi câu


def test_listwise_fuses_with_old_ranking(clubs_corpus):
    """Trọng số (1, 1): hai đoạn đầu đổi chỗ cho nhau chỉ làm hòa; các đoạn sau giữ thứ tự cũ."""
    ranker, _ = _listwise(clubs_corpus, lambda m: '{"ranking": [2, 1, 3, 4, 5]}')
    ids = [d for d, _ in ranker.search("câu hỏi", top_n=5)]
    assert set(ids[:2]) == {1, 2} and ids[2:] == [3, 4, 5]


@pytest.mark.parametrize("reply", ["", "hỏng {", '{"ranking": []}'])
def test_listwise_falls_back_on_bad_reply(clubs_corpus, reply):
    ranker, _ = _listwise(clubs_corpus, lambda m: reply)
    assert [d for d, _ in ranker.search("câu hỏi", top_n=5)] == [1, 2, 3, 4, 5]


def test_listwise_survives_llm_error(clubs_corpus):
    def broken(messages):
        raise RuntimeError("gateway down")

    ranker, _ = _listwise(clubs_corpus, broken)
    assert [d for d, _ in ranker.search("câu hỏi", top_n=5)] == [1, 2, 3, 4, 5]


def test_listwise_only_reranks_top_n_rerank_and_keeps_tail(clubs_corpus):
    ranker, fake = _listwise(
        clubs_corpus, lambda m: '{"ranking": [3, 2, 1]}', n_rerank=3, weights=(0, 1)
    )
    ids = [d for d, _ in ranker.search("câu hỏi", top_n=5)]
    assert ids == [3, 2, 1, 4, 5]  # LLM chỉ thấy 3 đoạn đầu; đoạn 4, 5 giữ nguyên ở cuối
    assert "[id=4]" not in fake.chats[0][1]["content"]


def test_listwise_prompt_has_queries_and_no_date(clubs_corpus):
    ranker, fake = _listwise(clubs_corpus, lambda m: "")
    plan = QueryPlan("goc khong dau", "Câu viết lại?", ["Câu con 1?"])
    ranker.search_plan(plan, top_n=5)
    system, user = (m["content"] for m in fake.chats[0])
    assert "Diễn giải: Câu viết lại?" in user and "Câu con 1?" in user
    assert "Hôm nay" not in system + user  # ngày nằm trong khóa cache


def test_pipeline_llm_rerank_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    assert isinstance(MyPipeline(corpus_rank=False).retriever, ListwiseReranker)  # mặc định bật
    plain = MyPipeline(llm_rerank=False, corpus_rank=False).retriever
    assert isinstance(plain, RerankedRetriever)


# ----------------------------------------------------------------------------------------------
# stepback.py (Step-Back)
# ----------------------------------------------------------------------------------------------


def test_query_plan_includes_stepback_last_and_dedups():
    plan = QueryPlan("goc", "Viết lại?", ["Con 1?"], stepback="Khái quát?")
    assert plan.queries == ["goc", "Viết lại?", "Con 1?", "Khái quát?"]
    assert QueryPlan("goc", "goc", [], stepback="goc").queries == ["goc"]
    assert QueryPlan("goc", "goc").stepback is None


@pytest.mark.parametrize(
    "reply, expected",
    [
        (
            '{"stepback": "CLB Multimedia PTIT có những ban nào?"}',
            "CLB Multimedia PTIT có những ban nào?",
        ),
        (
            '```json\n{"stepback": " Quyền lợi của thành viên là gì? "}\n```',
            "Quyền lợi của thành viên là gì?",
        ),
        ("", None),
        ("không phải json", None),
        ('{"stepback": 5}', None),
        ('{"stepback": "   "}', None),
        ('{"stepback": "' + "x" * 400 + '"}', None),
        ('{"stepback": "viết lại?"}', None),  # trùng bản viết lại (không phân biệt hoa/thường)
    ],
)
def test_parse_stepback(reply, expected):
    plan = QueryPlan("goc", "Viết lại?", [])
    assert parse_stepback(reply, plan) == expected


def test_stepback_survives_llm_error():
    def broken(messages):
        raise RuntimeError("gateway down")

    assert StepBack(llm=FakeGateway(broken), model="fake").generate(QueryPlan("a", "b?")) is None


def _sb_retriever(clubs_corpus, reply, with_subqueries=False, weight=1.0):
    def responder(messages):
        if messages[0]["content"] == REWRITE_PROMPT:
            subs = '["Câu con?"]' if with_subqueries else "[]"
            return (
                '{"rewrite": "CLB Multimedia PTIT thành lập khi nào?", "subqueries": ' + subs + "}"
            )
        assert messages[0]["content"] == STEPBACK_PROMPT
        return reply

    fake = FakeGateway(responder)
    retriever = MultiQueryRetriever(
        hybrid(fake),
        rewriter=QueryRewriter(llm=fake, model="fake"),
        stepback=StepBack(llm=fake, model="fake"),
        stepback_weight=weight,
    )
    retriever.index(clubs_corpus)
    return retriever, fake


def test_stepback_added_only_when_no_subqueries(clubs_corpus):
    reply = '{"stepback": "CLB Multimedia PTIT hoạt động ra sao?"}'
    retriever, fake = _sb_retriever(clubs_corpus, reply)
    plan = retriever.plan("clb mm thanh lap khi nao")
    assert plan.stepback == "CLB Multimedia PTIT hoạt động ra sao?"
    assert plan.queries[-1] == plan.stepback

    retriever, fake = _sb_retriever(clubs_corpus, reply, with_subqueries=True)
    plan = retriever.plan("clb mm thanh lap khi nao")
    assert plan.stepback is None and plan.subqueries == ["Câu con?"]
    assert len(fake.chats) == 1  # câu có câu con: không gọi thêm LLM cho step-back


def test_stepback_off_by_default_makes_no_extra_call(clubs_corpus):
    fake = FakeGateway(lambda m: '{"rewrite": "r?", "subqueries": []}')
    retriever = MultiQueryRetriever(hybrid(fake), rewriter=QueryRewriter(llm=fake, model="fake"))
    retriever.index(clubs_corpus)
    assert retriever.plan("clb mm").stepback is None and len(fake.chats) == 1


def test_stepback_weight_changes_fusion(clubs_corpus):
    reply = '{"stepback": "Tuyển thành viên qua link form như thế nào?"}'
    full, _ = _sb_retriever(clubs_corpus, reply, weight=1.0)
    zero, _ = _sb_retriever(clubs_corpus, reply, weight=0.0)
    plan = full.plan("clb mm thanh lap khi nao")
    off = MultiQueryRetriever(hybrid(FakeGateway()), rewriter=None)
    off.index(clubs_corpus)
    base = off.search_plan(QueryPlan(plan.original, plan.rewrite, [], None), top_n=5)
    # trọng số 0: câu khái quát không ảnh hưởng thứ hạng
    assert zero.search_plan(plan, top_n=5) == base or [d for d, _ in zero.search_plan(plan, 5)] == [
        d for d, _ in base
    ]


def test_stepback_not_used_by_reranker_or_listwise(clubs_corpus):
    reranker = FakeReranker()
    base, _ = _sb_retriever(clubs_corpus, '{"stepback": "Chủ đề chung của CLB?"}')
    retriever = RerankedRetriever(base, reranker=reranker)
    retriever.index(clubs_corpus)
    plan = retriever.plan("clb mm thanh lap khi nao")
    assert plan.stepback
    retriever.search_plan(plan, top_n=5)
    assert plan.stepback not in reranker.queries  # không dùng để chấm cross-encoder

    fake = FakeGateway(lambda m: "")
    ranker = ListwiseReranker(retriever, llm=fake, model="fake")
    ranker.index(clubs_corpus)
    ranker.search_plan(plan, top_n=5)
    assert plan.stepback not in fake.chats[0][1]["content"]  # không đưa vào prompt của A4


def test_dense_local_model_bypasses_gateway(monkeypatch, tmp_path, clubs_corpus):
    """local_model: embedding chạy trên máy, không gọi LLMClient.embed (không tốn gateway)."""
    import numpy as np

    class FakeLocal:
        def __init__(self, name):
            self.name = name

        def embed(self, texts):
            rng = np.random.default_rng(len(texts))
            v = rng.normal(size=(len(texts), 8)).astype(np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    class NoGateway:
        def embed(self, *a, **k):
            raise AssertionError("không được gọi gateway")

    monkeypatch.setattr("my_solution.retrievers.LocalEmbedder", FakeLocal)
    dense = DenseRetriever(llm=NoGateway(), local_model="some/model")
    assert dense.local.name == "some/model"
    dense.index(clubs_corpus)
    assert len(dense.search("câu hỏi", top_n=3)) == 3
    assert DenseRetriever(llm=NoGateway()).local is None  # mặc định vẫn là bge-m3 qua gateway


def test_pipeline_stepback_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    assert MyPipeline().retriever.base.base.first_stage.stepback is None  # mặc định tắt
    assert MyPipeline(stepback=True).retriever.base.base.first_stage.stepback is not None


def test_prompt_v3_differs_from_v2_only_in_rule_8():
    v2, v3 = SYSTEM_PROMPTS["v2"], SYSTEM_PROMPTS["v3"]
    diff = [(a, b) for a, b in zip(v2.split("\n"), v3.split("\n"), strict=True) if a != b]
    assert len(v2.split("\n")) == len(v3.split("\n")) and len(diff) == 1
    assert diff[0][0].startswith("8. ") and diff[0][1].startswith("8. ")
    assert "HỎI TRỰC TIẾP" in v3 and "HỎI TRỰC TIẾP" not in v2
    assert (
        MyPipeline(prompt_version="v3", llm=FakeGateway(), retriever=object()).prompt_version
        == "v3"
    )


# ----------------------------------------------------------------------------------------------
# corpus_rank.py (A5)
# ----------------------------------------------------------------------------------------------


def test_parse_corpus_ranking_filters_dedups_and_caps():
    valid = {1, 2, 3, 4, 5}
    assert parse_corpus_ranking('{"ranking": [3, 1, 99, 3, "2", true, 5]}', valid) == [3, 1, 5]
    assert parse_corpus_ranking('```json\n{"ranking": [2, 4]}\n```', valid) == [2, 4]
    assert parse_corpus_ranking('{"ranking": [1, 2, 3, 4, 5]}', valid, top_k=2) == [1, 2]


@pytest.mark.parametrize(
    "reply", ["", "không phải json", "{}", '{"ranking": "1,2"}', '{"ranking": [99]}']
)
def test_parse_corpus_ranking_returns_none_on_junk(reply):
    assert parse_corpus_ranking(reply, {1, 2, 3}) is None


def _corpus_ranker(clubs_corpus, responder, order=(1, 2, 3, 4, 5), **kw):
    fake = FakeGateway(responder)
    ranker = CorpusRanker(StubBase(list(order)), llm=fake, model="fake", **kw)
    ranker.index(clubs_corpus)
    return ranker, fake


def test_corpus_ranker_prompt_has_whole_corpus_and_no_date(clubs_corpus):
    ranker, fake = _corpus_ranker(clubs_corpus, lambda m: "")
    plan = QueryPlan("goc khong dau", "Câu viết lại?", ["Câu con 1?"])
    ranker.search_plan(plan, top_n=5)
    system, user = (m["content"] for m in fake.chats[0])
    assert all(f"[id={d.id}]" in system for d in clubs_corpus)  # đủ mọi đoạn
    assert "Diễn giải: Câu viết lại?" in user and "Câu con 1?" in user
    assert "Hôm nay" not in system + user  # ngày nằm trong khóa cache
    assert len(fake.chats) == 1  # đúng 1 lời gọi mỗi câu


def test_corpus_ranker_can_bring_in_doc_outside_base_list(clubs_corpus):
    """A5 đưa vào đoạn 5 mà bảng cũ (chỉ có 1-3) không có. Đoạn 2 có ở cả hai bảng nên vẫn đứng đầu
    (4/62 so với 3/61 của đoạn 5); đoạn 5 vượt các đoạn chỉ có ở bảng cũ (1 và 3)."""
    ranker, _ = _corpus_ranker(
        clubs_corpus, lambda m: '{"ranking": [5, 2]}', order=(1, 2, 3), weights=(1, 3)
    )
    ids = [d for d, _ in ranker.search("câu hỏi", top_n=5)]
    assert ids == [2, 5, 1, 3]


def test_corpus_ranker_zero_weight_keeps_base_order(clubs_corpus):
    ranker, _ = _corpus_ranker(clubs_corpus, lambda m: '{"ranking": [5, 4]}', weights=(1, 0))
    assert [d for d, _ in ranker.search("câu hỏi", top_n=3)] == [1, 2, 3]


@pytest.mark.parametrize("reply", ["", "hỏng {", '{"ranking": [99]}'])
def test_corpus_ranker_falls_back_on_bad_reply(clubs_corpus, reply):
    ranker, _ = _corpus_ranker(clubs_corpus, lambda m: reply)
    assert [d for d, _ in ranker.search("câu hỏi", top_n=5)] == [1, 2, 3, 4, 5]


def test_corpus_ranker_survives_llm_error(clubs_corpus):
    def broken(messages):
        raise RuntimeError("gateway down")

    ranker, _ = _corpus_ranker(clubs_corpus, broken)
    assert [d for d, _ in ranker.search("câu hỏi", top_n=5)] == [1, 2, 3, 4, 5]
    assert (ranker.calls, ranker.fallbacks) == (1, 1)  # lùi về A4 được đếm


def test_pipeline_corpus_rank_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    off = MyPipeline(corpus_rank=False).retriever
    assert isinstance(off, ListwiseReranker)  # tắt: A4 ngoài cùng
    on = MyPipeline().retriever  # mặc định bật
    assert isinstance(on, CorpusRanker) and isinstance(on.base, ListwiseReranker)


def test_pipeline_a5_uses_short_timeout_client(monkeypatch, tmp_path):
    """A5 dùng client riêng: timeout ngắn, không thử lại (quá hạn thì lùi về A4)."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    ranker = MyPipeline().retriever
    assert ranker.llm._client.timeout == 20.0 and ranker.llm._client.max_retries == 0
    other = MyPipeline().llm._client  # các lời gọi khác giữ nguyên cấu hình mặc định
    assert other.timeout == 60.0 and other.max_retries == 5


# ----------------------------------------------------------------------------------------------
# trace.py (ghi vết cho chatbot)
# ----------------------------------------------------------------------------------------------


def _traced_pipeline(clubs_corpus, a5_responder=None):
    fake = FakeGateway(_responder)
    retriever = RerankedRetriever(
        MultiQueryRetriever(hybrid(fake), rewriter=QueryRewriter(llm=fake, model="fake")),
        reranker=FakeReranker(),
    )
    if a5_responder is not None:
        retriever = CorpusRanker(retriever, llm=FakeGateway(a5_responder), model="fake")
    pipeline = MyPipeline(n_context=3, llm=fake, llm_model="fake", retriever=retriever)
    pipeline.prepare()
    pipeline.index(clubs_corpus)
    return pipeline, fake


def test_run_traced_matches_pipeline_run(clubs_corpus):
    """Chatbot phải trả lời đúng như pipeline thi: cùng id, cùng câu trả lời, cùng prompt."""
    pipeline, fake = _traced_pipeline(clubs_corpus)
    out = pipeline.run("clb mm thanh lap khi nao")
    trace = run_traced(pipeline, "clb mm thanh lap khi nao")
    assert trace.answer == out.answer
    assert [i for i, _ in trace.hits] == out.retrieved_ids
    assert fake.chats[-1] == fake.chats[1]  # prompt sinh câu trả lời giống hệt
    assert trace.rewrite == "CLB Multimedia PTIT thành lập khi nào?"
    assert trace.n_context == 3
    names = [s.name for s in trace.stages]
    assert names == [
        "Viết lại câu hỏi",
        "Tìm ứng viên",
        "Chấm lại bằng cross-encoder",
        "Viết câu trả lời",
    ]
    assert all(s.seconds is not None and s.seconds >= 0 for s in trace.stages)
    # đã trả search_plan về phương thức của lớp (không để lại bản bọc trên đối tượng)
    assert "search_plan" not in vars(pipeline.retriever)
    assert "search_plan" not in vars(pipeline.retriever.first_stage)


def test_run_traced_reports_progress_and_a5_fallback(clubs_corpus):
    def broken(messages):
        raise TimeoutError("quá hạn")

    pipeline, _ = _traced_pipeline(clubs_corpus, a5_responder=broken)
    seen = []
    trace = run_traced(
        pipeline, "clb mm thanh lap khi nao", on_update=lambda t: seen.append(t.rewrite)
    )
    assert trace.fallback is True
    a5 = next(s for s in trace.stages if s.name == "LLM đọc toàn bộ tài liệu")
    assert "dùng kết quả bước trước" in a5.note
    assert seen[0] == "" and seen[-1] == trace.rewrite  # câu viết lại hiện trước khi có câu trả lời
    assert [s.name for s in trace.stages][-1] == "Viết câu trả lời"


def test_run_traced_without_trace_support_still_answers():
    class Plain:
        def run(self, query):
            from neorag.pipelines.base import RAGOutput

            return RAGOutput(answer="ok", retrieved_ids=[3, 1])

    trace = run_traced(Plain(), "hỏi")
    assert trace.traced is False and trace.answer == "ok" and [i for i, _ in trace.hits] == [3, 1]


def test_mark_rewrite_marks_restored_accents_and_expansions():
    out = mark_rewrite("e ko co hoa tay", "Em không có hoa tay")
    assert '<mark class="dau">có</mark>' in out  # co -> có: khôi phục dấu
    assert '<mark class="giainghia">không</mark>' in out  # ko -> không: giải nghĩa viết tắt
    assert "hoa tay" in out and "<mark" not in out.split("có")[1]  # từ không đổi thì không tô
    assert mark_rewrite("a", "<b>") == '<mark class="giainghia">&lt;b&gt;</mark>'  # thoát HTML


# ----------------------------------------------------------------------------------------------
# history.py
# ----------------------------------------------------------------------------------------------


def test_history_roundtrip_delete_and_corrupt_file(tmp_path):
    from datetime import datetime

    from my_solution.history import History, day_label, entry_to_trace
    from my_solution.trace import Stage, Trace

    hist = History(tmp_path / "sub" / "h.json")
    assert hist.load() == []  # chưa có file
    t = Trace(
        question="e ko co hoa tay",
        rewrite="Em không có hoa tay",
        hits=[(94, 0.5)],
        n_context=1,
        answer="Có.",
        stages=[Stage("Viết lại câu hỏi", 1.2), Stage("LLM đọc toàn bộ tài liệu", 2.0, "ghi chú")],
        fallback=True,
    )
    first = hist.add(t)
    second = hist.add(Trace(question="câu hai"))
    loaded = hist.load()
    assert {e["id"] for e in loaded} == {first["id"], second["id"]}
    back = entry_to_trace(next(e for e in loaded if e["id"] == first["id"]))
    assert back == t  # lưu rồi đọc lại không mất thông tin (kể cả tiếng Việt có dấu)
    hist.delete(second["id"])
    assert [e["id"] for e in hist.load()] == [first["id"]]
    hist.path.write_text("{hỏng", encoding="utf-8")
    assert hist.load() == []  # file hỏng không làm sập chatbot

    now = datetime(2026, 11, 6, 12, 0)
    assert day_label("2026-11-06T08:00:00", now) == "Hôm nay"
    assert day_label("2026-11-05T23:59:00", now) == "Hôm qua"
    assert day_label("2026-11-01T10:00:00", now) == "01/11/2026"


# ----------------------------------------------------------------------------------------------
# chat_view.py
# ----------------------------------------------------------------------------------------------


def test_chat_view_escapes_untrusted_text_and_marks_sources():
    from my_solution.chat_view import club_color, rail_html, trace_html
    from my_solution.trace import Stage, Trace

    docs = {
        1: Document(1, "CLB Multimedia PTIT", "Cơ cấu", "<script>alert(1)</script> Ban Nội dung")
    }
    t = Trace(
        question='<img src=x onerror="a()">',
        rewrite="CLB Multimedia PTIT có ban nào?",
        subqueries=["<b>x</b>"],
        hits=[(1, 0.9), (1, 0.8)],
        n_context=1,
        stages=[Stage("Viết lại câu hỏi", 1.0), Stage("LLM đọc toàn bộ tài liệu", 2.0, "quá hạn")],
    )
    out = trace_html(t, docs)
    assert "<script>" not in out and "<img" not in out and "<b>x</b>" not in out
    assert "&lt;script&gt;" in out and "đưa vào câu trả lời, hạng 1" in out and "dự phòng" in out
    assert "quá hạn" in out and "Tổng 3.0 s" in out
    assert club_color("CLB Multimedia PTIT") == "#9B2C7F" and club_color("lạ") == "#5B6472"
    running = rail_html([Stage("a", 1.0), Stage("b"), Stage("c")], running=True)
    assert running.count("step done") == 1
    assert running.count("step run") == 1 and running.count("step wait") == 1


def test_run_traced_full_chain_has_all_stages_in_order(clubs_corpus):
    fake = FakeGateway(_responder)
    retriever = RerankedRetriever(
        MultiQueryRetriever(hybrid(fake), rewriter=QueryRewriter(llm=fake, model="fake")),
        reranker=FakeReranker(),
    )
    retriever = ListwiseReranker(
        retriever, llm=FakeGateway(lambda m: '{"ranking": [2, 1]}'), model="fake"
    )
    retriever = CorpusRanker(retriever, llm=FakeGateway(lambda m: '{"ranking": [2]}'), model="fake")
    pipeline = MyPipeline(n_context=3, llm=fake, llm_model="fake", retriever=retriever)
    pipeline.prepare()
    pipeline.index(clubs_corpus)
    trace = run_traced(pipeline, "clb mm thanh lap khi nao")
    assert [s.name for s in trace.stages] == [
        "Viết lại câu hỏi",
        "Tìm ứng viên",
        "Chấm lại bằng cross-encoder",
        "LLM xếp lại",
        "LLM đọc toàn bộ tài liệu",
        "Viết câu trả lời",
    ]
    assert trace.fallback is False and all(s.seconds is not None for s in trace.stages)
    assert [i for i, _ in trace.hits] == [
        i for i, _ in pipeline.retriever.search("clb mm thanh lap khi nao")
    ]
    assert all(
        "search_plan" not in vars(layer)
        for layer in (retriever, retriever.base, retriever.base.base)
    )


def test_escape_md_keeps_text_literal():
    from my_solution.chat_view import escape_md

    out = escape_md("giá $5 ![x](http://e/x.png) **đậm** a_b")
    assert out == "giá \\$5 \\!\\[x\\]\\(http://e/x.png\\) \\*\\*đậm\\*\\* a\\_b"
    assert escape_md("Có ban Nội dung.") == "Có ban Nội dung."  # chữ tiếng Việt thường không đổi


def test_history_survives_bad_entries_and_failed_save(tmp_path, monkeypatch):
    import json

    from my_solution.history import History, entry_to_trace
    from my_solution.trace import Trace

    hist = History(tmp_path / "h.json")
    hist.path.write_text(
        json.dumps(
            [
                {"id": "a", "question": "thiếu giờ và câu trả lời"},
                {
                    "id": "b",
                    "question": "stage hỏng",
                    "ts": "2026-11-06T10:00:00",
                    "stages": [1, {"x": 1}, {"name": "Tìm"}],
                },
                {"id": "c", "question": 5},  # sai kiểu
                {"question": "thiếu id"},
                "rác",
            ]
        ),
        encoding="utf-8",
    )
    loaded = hist.load()
    assert {e["id"] for e in loaded} == {"a", "b"} and all(isinstance(e["ts"], str) for e in loaded)
    assert [s.name for s in entry_to_trace(next(e for e in loaded if e["id"] == "b")).stages] == [
        "Tìm"
    ]
    assert entry_to_trace(next(e for e in loaded if e["id"] == "a")).answer == ""

    hist.path.write_text('{"không": "phải danh sách"}', encoding="utf-8")
    assert hist.load() == []

    def locked(self, entries):
        raise PermissionError("file đang bị khóa")

    monkeypatch.setattr(History, "_save", locked)
    entry = hist.add(Trace(question="q", answer="a"))  # không ném lỗi, vẫn hiển thị được
    assert entry["unsaved"] is True and entry["question"] == "q"
    hist.delete("a")  # không ném lỗi
