"""
agents.graph — Supervisor 编排的 StateGraph 工厂

拓扑：

    entry → supervisor_route
    supervisor_route ──(条件边: route_to_agent)──→ knowledge_rag | ticket_handler | compliance_check
    knowledge_rag    ──(固定边)──→ compliance_check
    ticket_handler   ──(固定边)──→ compliance_check
    compliance_check ──→ synthesize ──→ END

拓扑含义：知识/工单两条内容生产路径经固定边必过合规审查再进汇总，
这是「所有回复必须过审」在图结构上的表达。直接路由 compliance_check
的路径汇入同一节点，该路径审查用户消息本身（产出的 sub_results
为空，不查用户消息等于该路径空转，见 compliance_checker 的
process）。

子 Agent：
- knowledge_rag：RAG 四步流水线，检索走 LongTermMemory
  （FAISS + 真实 embedding，嵌入不可用降级二元组）
- ticket_handler：分析 / 建单 / 查询，进程内 TicketStore，
  store 可从外部注入共享（图与 MCP 工具用同一张单）
- compliance_check：规则引擎 + LLM 深度审查，两阶段串联
  （规则失败短路、通过再上 LLM，process 走 full_check）

checkpointer：MemorySaver（进程内），thread_id 由调用方经 config
传入（= session_id）。跨轮通道因此完整：messages 经 checkpointer
累积，last_intent 经 working_memory 携带。

装配设计：
- LLM 兜底从 .env 构造：换模型只改环境变量不改代码
- 图工厂参数即依赖注入面：long_term_memory 与 ticket_store 由
  API 层构造注入；short_term_memory 不进图——对话历史写透发生在
  API 层（main.py），图内跨轮通道是 messages（checkpointer）与
  working_memory
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.compliance_checker import ComplianceCheckerAgent
from agents.knowledge_rag import KnowledgeRAGAgent
from agents.state import AgentState
from agents.supervisor import SupervisorNode, route_to_agent
from agents.ticket_handler import TicketHandlerAgent, TicketStore
from memory.long_term import LongTermMemory
from memory.working_memory import WorkingMemory


def _default_llm() -> ChatOpenAI:
    """
    兜底 LLM：从环境变量构造（.env 由入口加载，本模块不做副作用）。

    temperature=0 保证路由决策可复现（冒烟测试依赖这一点做断言）。
    """
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "DeepSeek-V4-Pro-0813"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://llmapi.paratera.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def create_supervisor_graph(
    llm: ChatOpenAI | None = None,
    working_memory: WorkingMemory | None = None,
    long_term_memory: LongTermMemory | None = None,
    ticket_store: TicketStore | None = None,
    enable_checkpointing: bool = True,
) -> CompiledStateGraph:
    """
    构建 Supervisor 编排的多 Agent StateGraph（系统核心入口）。

    Args:
        llm: Supervisor 决策用的 LLM，未传则从 .env 构造
        working_memory: 工作记忆，未传则新建（生产由 main.py 建单例注入）
        long_term_memory: 长期记忆（知识检索 Agent 的检索后端），
                          未传则新建（2.2 起为 FAISS + 真实 embedding）
        ticket_store: 工单存储。未传则由工单 Agent 自建（独立于
                      MCP）；传入则图与 MCP 工具共享同一张单
                      （2.3，生产由 main.py 建单例注入）
        enable_checkpointing: 是否启用 MemorySaver（按 thread 存档，
                              消息跨轮累积依赖它）
    """
    if llm is None:
        llm = _default_llm()
    if working_memory is None:
        working_memory = WorkingMemory()
    if long_term_memory is None:
        long_term_memory = LongTermMemory()
    if ticket_store is None:
        ticket_store = TicketStore()

    supervisor = SupervisorNode(llm, working_memory)
    knowledge_agent = KnowledgeRAGAgent(llm, long_term_memory)
    ticket_agent = TicketHandlerAgent(llm, ticket_store=ticket_store)
    compliance_agent = ComplianceCheckerAgent(llm)

    graph = StateGraph(AgentState)

    graph.add_node("supervisor_route", supervisor.route_decision)
    graph.add_node("knowledge_rag", knowledge_agent.process)
    graph.add_node("ticket_handler", ticket_agent.process)
    graph.add_node("compliance_check", compliance_agent.process)
    graph.add_node("synthesize", supervisor.synthesize_response)

    graph.set_entry_point("supervisor_route")

    # 条件边：路由函数返回分支键（合法意图名），path_map 翻译成节点名。
    # 意图名与节点名不一致的 compliance_checker → compliance_check 在此吸收
    graph.add_conditional_edges(
        "supervisor_route",
        route_to_agent,
        {
            "knowledge_rag": "knowledge_rag",
            "ticket_handler": "ticket_handler",
            "compliance_checker": "compliance_check",
        },
    )

    # 固定边：内容生产路径必过合规审查
    graph.add_edge("knowledge_rag", "compliance_check")
    graph.add_edge("ticket_handler", "compliance_check")
    graph.add_edge("compliance_check", "synthesize")
    graph.add_edge("synthesize", END)

    checkpointer = MemorySaver() if enable_checkpointing else None
    return graph.compile(checkpointer=checkpointer)
