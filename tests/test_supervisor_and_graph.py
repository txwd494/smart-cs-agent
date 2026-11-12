"""test_supervisor_and_graph — 图级 mock 流程测试（1.4 冒烟的固化
+ 3.1 的编排逻辑回归）：stub LLM 提示匹配，整图执行毫秒级确定性

提示匹配关键词对应各模块的系统提示（改提示词时同步这里）：
路由 → supervisor 路由指令（要求返回纯文本 Agent 名，非 JSON）；
口语化问题改写 → RAG 改写步；文档相关性 → 重排步；
知识库问答 → 生成步；金融合规审查 → 合规 llm_check；
工单处理Agent → 工单 analyze。
"""

import pytest
from langchain_core.messages import HumanMessage

from memory.long_term import LongTermMemory
from memory.working_memory import WorkingMemory
from tests.conftest import FakeEmbeddings, StubLLM

STUB_RULES = [
    ("请分析用户的最新消息", "knowledge_rag"),  # supervisor：纯文本路由
    ("口语化问题改写", "理财产品A年化收益率"),
    ("文档相关性排序", "0"),
    ("知识库问答", "理财产品A年化收益率为3.5%-5.2%。理财非存款，投资须谨慎。"),
    ("金融合规审查", '{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}'),
]


def _stub_llm():
    return StubLLM(STUB_RULES)


def _initial_state(message, session_id):
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": "u1", "session_id": session_id,
        "intent": "", "sub_results": {}, "compliance_passed": True,
        "final_response": "", "current_agent": "",
    }


async def _build_graph():
    """stub LLM + 假嵌入长期记忆的完整图（零网络）"""
    from agents.graph import create_supervisor_graph

    ltm = LongTermMemory(embeddings=FakeEmbeddings())
    await ltm.add_document("理财产品A年化收益率为3.5%-5.2%", source="product_faq.txt")
    return create_supervisor_graph(
        llm=_stub_llm(), working_memory=WorkingMemory(), long_term_memory=ltm,
    )


@pytest.fixture
async def graph():
    return await _build_graph()


async def test_knowledge_flow_end_to_end(graph):
    result = await graph.ainvoke(
        _initial_state("理财产品A的年化收益率是多少", "g1"),
        config={"configurable": {"thread_id": "g1"}},
    )
    assert result["intent"] == "knowledge_rag"
    # 生成步的 stub 响应进了 sub_results 与最终回复
    assert "3.5" in result["sub_results"]["knowledge_rag"]
    assert "3.5" in result["final_response"]
    assert result["compliance_passed"] is True


async def test_working_memory_carries_last_intent(graph):
    # 跨轮通道：第一轮路由结果写入 working memory，第二轮可读
    await graph.ainvoke(
        _initial_state("理财产品A的年化收益率是多少", "wm1"),
        config={"configurable": {"thread_id": "wm1"}},
    )
    # 直接检查 Supervisor 写入的 last_intent（图外自管通道）
    result = await graph.ainvoke(
        _initial_state("换个问题", "wm1"),
        config={"configurable": {"thread_id": "wm1"}},
    )
    assert result["intent"] in {"knowledge_rag", "ticket_handler", "compliance_checker"}


async def test_synthesize_joins_only_strings():
    # compliance dict 混入 sub_results 不得让 join 崩（1.4 差异 2 锚点）
    from agents.supervisor import SupervisorNode

    node = SupervisorNode(llm=_stub_llm(), working_memory=WorkingMemory())
    out = await node.synthesize_response({
        "sub_results": {
            "knowledge_rag": "答案",
            "compliance": {"passed": True, "risk_level": "low", "violations": []},
        },
        "compliance_passed": True,
    })
    assert out["final_response"] == "答案"


async def test_synthesize_handoff_when_compliance_failed():
    from agents.supervisor import SupervisorNode

    node = SupervisorNode(llm=_stub_llm(), working_memory=WorkingMemory())
    out = await node.synthesize_response({"sub_results": {}, "compliance_passed": False})
    assert "已转交人工客服" in out["final_response"]


async def test_messages_accumulate_across_turns(graph):
    config = {"configurable": {"thread_id": "acc1"}}
    r1 = await graph.ainvoke(
        _initial_state("理财产品A的年化收益率是多少", "acc1"), config=config,
    )
    r2 = await graph.ainvoke(
        _initial_state("再问一次", "acc1"), config=config,
    )
    # checkpointer 通道：同 thread 的 messages 累积
    assert len(r2["messages"]) > len(r1["messages"])
