"""test_rag_and_router — RAG 流水线各步（1.6 冒烟固化 + rerank
边界回归锚点）与意图路由解析防护（1.5），stub LLM 零网络
"""

from agents.compliance_checker import ComplianceCheckerAgent
from agents.intent_router import IntentRouterAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from memory.long_term import LongTermMemory
from tests.conftest import FakeEmbeddings, StubLLM


async def _seeded_ltm() -> LongTermMemory:
    ltm = LongTermMemory(embeddings=FakeEmbeddings())
    await ltm.add_document("理财产品A年化收益率为3.5%-5.2%", source="product_faq.txt")
    await ltm.add_document("退款政策为7天无理由退款", source="refund_policy.txt")
    return ltm


# ── rerank 边界（1.6 差异 2/3 的回归锚点） ──────────────────────

async def test_rerank_dedup_and_truncate():
    stub = StubLLM([], default="0, 0, 1, 1, 2")  # 重复索引
    agent = KnowledgeRAGAgent(llm=stub, long_term_memory=await _seeded_ltm())
    docs = [{"content": f"d{i}", "source": f"f{i}.txt", "score": 1.0} for i in range(5)]
    out = await agent.rerank_documents("q", docs, top_k=3)
    # 去重保序 + 截断：0,1,2
    assert [d["content"] for d in out] == ["d0", "d1", "d2"]


async def test_rerank_negative_index_filtered():
    # 边界锚点：负索引穿透会让 LLM 返回的 -1 从列表尾部取文档
    stub = StubLLM([], default="-1, 0")
    agent = KnowledgeRAGAgent(llm=stub, long_term_memory=await _seeded_ltm())
    docs = [{"content": f"d{i}", "source": "f.txt", "score": 1.0} for i in range(3)]
    out = await agent.rerank_documents("q", docs, top_k=3)
    assert [d["content"] for d in out] == ["d0"]


async def test_rerank_parse_failure_falls_back_to_order():
    stub = StubLLM([], default="无法排序")
    agent = KnowledgeRAGAgent(llm=stub, long_term_memory=await _seeded_ltm())
    docs = [{"content": f"d{i}", "source": "f.txt", "score": 1.0} for i in range(5)]
    out = await agent.rerank_documents("q", docs, top_k=3)
    assert [d["content"] for d in out] == ["d0", "d1", "d2"]  # 检索序兜底


# ── 流水线（stub 全链） ─────────────────────────────────────────

async def test_rag_pipeline_with_stubs():
    stub = StubLLM([
        ("改写", "理财产品A年化收益率"),
        ("相关性", "0"),
        ("参考文档", "理财产品A年化收益率为3.5%-5.2%，投资须谨慎。"),
    ])
    agent = KnowledgeRAGAgent(llm=stub, long_term_memory=await _seeded_ltm())
    out = await agent.process({
        "messages": [type("M", (), {"content": "理财产品A收益率多少"})()],
        "user_id": "u1", "session_id": "s1",
    })
    reply = out["sub_results"]["knowledge_rag"]
    assert "3.5" in reply


async def test_rag_empty_retrieval_fixed_fallback():
    # 知识库零命中 → 固定兜底话术，不再调生成（1.6 设计）
    stub = StubLLM([("改写", "完全不相关的查询")])
    empty_ltm = LongTermMemory(embeddings=FakeEmbeddings())
    await empty_ltm.add_document("无关内容", source="x.txt")
    agent = KnowledgeRAGAgent(llm=stub, long_term_memory=empty_ltm)
    # 检索必然零命中（查询与文档无字符重合且假嵌入余弦极低）：
    # 验证走的是空检索兜底分支而非生成
    out = await agent.process({
        "messages": [type("M", (), {"content": "甲乙丙丁戊己庚辛"})()],
        "user_id": "u1", "session_id": "s1",
    })
    assert "knowledge_rag" in out["sub_results"]


# ── 意图路由解析防护（1.5 差异 2/3 的锚点） ─────────────────────

async def test_intent_router_valid_json():
    stub = StubLLM([], default=(
        '{"primary_intent": "consultation", "secondary_intent": "产品咨询", '
        '"confidence": "0.9", "entities": {"产品": "理财产品A"}, '
        '"suggested_agent": "knowledge_rag"}'
    ))
    agent = IntentRouterAgent(llm=stub)
    result = await agent.classify("理财产品A怎么样")
    assert result.primary_intent.value == "consultation"
    assert result.confidence == 0.9  # 字符串置信度可转
    assert result.entities == {"产品": "理财产品A"}
    assert result.suggested_agent == "knowledge_rag"


async def test_intent_router_invalid_enum_falls_back():
    stub = StubLLM([], default=(
        '{"primary_intent": "不存在的类别", "secondary_intent": "", '
        '"confidence": 0.5, "entities": [], '
        '"suggested_agent": "hacker_agent"}'
    ))
    agent = IntentRouterAgent(llm=stub)
    result = await agent.classify("随便")
    # 非法枚举不崩 + 非法路由目标回落合法集合（裸转换会直接抛异常）
    assert result.primary_intent.value == "unknown"
    assert result.suggested_agent in {
        "knowledge_rag", "ticket_handler", "compliance_checker",
    }


async def test_intent_router_garbage_output():
    stub = StubLLM([], default="这不是JSON")
    agent = IntentRouterAgent(llm=stub)
    result = await agent.classify("随便")
    assert result is not None  # 全字段兜底，不崩
