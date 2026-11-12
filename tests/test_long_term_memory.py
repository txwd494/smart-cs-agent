"""test_long_term_memory — 2.2 冒烟断言的固化（分块 / 二元组 /
假嵌入索引 / 持久化 / 降级），全程零网络（嵌入走 conftest 假实现）
"""

import shutil

from memory.bigram_search import _overlap_score
from memory.long_term import LongTermMemory, _chunk_text
from tests.conftest import BrokenEmbeddings, FakeEmbeddings

TMP = "_test_pytest_faiss"


# ── 分块确定性 ──────────────────────────────────────────────────

def test_chunk_paragraph_split():
    chunks = _chunk_text("第一段内容。\n\n第二段内容。\n\n第三段内容。", chunk_size=10, overlap=4)
    assert len(chunks) >= 2


def test_chunk_decimal_not_split_at_boundary():
    # 边界锚点：按英文句点切会把 3.5% 切进块边界
    text = "收益率为3.5%-5.2%，期限为6个月至3年。" * 8
    chunks = _chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        assert not (a[-1] in "0123456789." and b[0] in "0123456789%.")


def test_chunk_overlap_prefix():
    # 后块以前块尾部开头（无空白文本上精确成立）
    text = "甲乙丙丁戊己庚辛壬子丑寅卯辰巳午未申酉戌亥" * 20
    chunks = _chunk_text(text, chunk_size=20, overlap=6)
    for a, b in zip(chunks, chunks[1:]):
        assert b.startswith(a[-6:])


def test_chunk_hard_cut():
    chunks = _chunk_text("无句读连续文本" * 200, chunk_size=50, overlap=20)
    assert len(chunks) >= 2 and all(len(c) <= 50 for c in chunks)


def test_chunk_deterministic():
    text = "收益率为3.5%-5.2%，期限为6个月至3年。\n\n退款政策说明。"
    assert _chunk_text(text) == _chunk_text(text)


# ── 二元组评分（降级路径的语义） ────────────────────────────────

def test_overlap_score_identical():
    assert _overlap_score("理财产品", "理财产品") == 1.0


def test_overlap_score_disjoint():
    assert _overlap_score("甲乙丙", "子丑寅") == 0.0


def test_overlap_score_partial():
    score = _overlap_score("理财收益", "理财产品收益率")
    assert 0.0 < score < 1.0


def test_overlap_score_empty():
    assert _overlap_score("", "内容") == 0.0
    assert _overlap_score("查询", "") == 0.0


# ── 假嵌入索引（真实 FAISS，零网络） ────────────────────────────

async def test_fake_embeddings_index_and_search(fake_embeddings):
    ltm = LongTermMemory(embeddings=fake_embeddings)
    await ltm.add_document("理财产品A年化收益率为3.5%", source="product_faq.txt")
    await ltm.add_document("退款政策为7天无理由", source="refund_policy.txt")
    assert ltm._index is not None and ltm._index.ntotal == 2
    assert ltm._index.d == fake_embeddings.dim  # 维度探测（2.2 差异 2）
    # 确定性：假嵌入无语义（哈希向量），只断言同查询两次结果稳定一致
    r1 = await ltm.search("任意查询", top_k=1)
    r2 = await ltm.search("任意查询", top_k=1)
    assert r1[0]["source"] == r2[0]["source"]
    assert r1[0]["score"] == r2[0]["score"]


# ── 持久化（零网络：加载注入抛异常 stub 证明无嵌入调用） ────────

async def test_persistence_roundtrip(fake_embeddings):
    shutil.rmtree(TMP, ignore_errors=True)
    ltm = LongTermMemory(index_path=f"{TMP}/idx", embeddings=fake_embeddings)
    await ltm.add_document("内容一", source="a.txt")
    await ltm.add_document("内容二", source="b.txt")
    ltm.save()

    ltm2 = LongTermMemory(index_path=f"{TMP}/idx", embeddings=BrokenEmbeddings())
    assert len(ltm2._documents) == 2
    assert ltm2._index is not None and ltm2._index.ntotal == 2
    assert ltm2._index.d == fake_embeddings.dim
    shutil.rmtree(TMP, ignore_errors=True)


# ── 读写失败的对称策略 ──────────────────────────────────────────

async def test_add_failure_no_residue():
    # add 嵌入失败抛异常且零残留（2.2 自查修正的先嵌入后入库顺序）
    ltm = LongTermMemory(embeddings=BrokenEmbeddings())
    try:
        await ltm.add_document("测试", source="t.txt")
        raise AssertionError("应抛异常")
    except RuntimeError:
        pass
    assert ltm._documents == []


async def test_search_fallback_to_bigram(fake_embeddings):
    # 索引时可用、查询时故障 → 降级二元组不拒服务
    ltm = LongTermMemory(embeddings=fake_embeddings)
    await ltm.add_document("理财产品A年化收益率3.5%-5.2%", source="product_faq.txt")
    ltm._embeddings = BrokenEmbeddings()
    r = await ltm.search("理财产品A的收益率是多少", top_k=3)
    assert r and r[0]["source"] == "product_faq.txt"
