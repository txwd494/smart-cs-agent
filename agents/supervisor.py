"""
agents.supervisor — Supervisor 编排节点（路由决策 + 结果汇总）

SupervisorNode 承担图里两个节点，是整个系统的中央协调者：
- route_decision（supervisor_route 节点）：用 LLM 分析意图，决定路由。
  每轮读 working memory 的上下文注入 prompt，决策后写回 last_intent，
  下一轮由此携带上一轮的路由结果（该信息的唯一载体，语义与机制见
  memory/working_memory.py 的模块 docstring）
- synthesize_response（synthesize 节点）：汇总子 Agent 结果生成最终
  回复，并把回复以 AIMessage 追加进 messages——这是对话历史的第二个
  写入点（第一个是端点写入用户消息）

route_decision 的 prompt 四段结构：
    [角色系统提示] + [工作记忆上下文(SystemMessage)] + [*messages] + [决策指令]
决策指令限定只返回三个合法值之一。LLM 输出经 strip/lower 规范化后再
校验一次，非法值回落 knowledge_rag；条件边一侧的 route_to_agent 是
同一校验的第二道兜底（双保险）。

关键设计决策（四处）：
1. 节点返回只含变更键（delta 约定）。全量回写 messages 会触发
   add_messages 的按 ID 替换，净效果为零，纯开销
2. synthesize 只拼接非空字符串并做类型过滤。compliance 节点会往
   sub_results 塞一个 dict，无过滤的 str.join 遇到非 str 抛
   TypeError，知识/工单路由的正常路径会因此崩掉
3. 系统提示只列三个可路由子 Agent，与图的实际节点一一对应：
   决策指令允许返回图中不存在的节点是路由错误的来源
4. 条件边的键翻译集中一处。langgraph 的 path_map 以路由函数
   返回值为键查表，路由函数必须返回 path_map 里存在的键；意图名
   与节点名不一致时（compliance_checker 对 compliance_check），
   翻译集中在 path_map 一处完成，路由函数只返回合法意图名
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from memory.working_memory import WorkingMemory
from tracing.otel_config import trace_agent_call


# 系统提示只列三个可路由的子 Agent，与图的实际节点一一对应
SUPERVISOR_SYSTEM_PROMPT = """你是一个智能客服系统的Supervisor（主管编排Agent）。
你的职责是：
1. 分析用户意图，决定分发给哪个子Agent处理
2. 汇总子Agent的处理结果，生成最终回复
3. 确保所有回复都经过合规审查

可用的子Agent：
- knowledge_rag: 知识库检索和回答
- ticket_handler: 工单创建和查询
- compliance_checker: 合规审查和敏感词检测

根据用户消息，决定下一步路由到哪个Agent。
"""

# 合法意图集合：route_decision 的校验依据
VALID_INTENTS = {"knowledge_rag", "ticket_handler", "compliance_checker"}


class SupervisorNode:
    """Supervisor 决策节点（route_decision）与汇总节点（synthesize_response）"""

    def __init__(self, llm: ChatOpenAI, working_memory: WorkingMemory):
        self.llm = llm
        self.working_memory = working_memory

    @trace_agent_call("supervisor")
    async def route_decision(self, state: AgentState) -> dict[str, Any]:
        """分析用户意图，决定路由（supervisor_route 节点）"""
        messages = state.get("messages", [])
        session_id = state.get("session_id", "default")

        # 先读上一轮的路由结果（session 的合并快照）。
        # 读在写之前：本轮的决策尚未产生，读到的是上一轮的值
        context = self.working_memory.get_context(session_id)

        routing_prompt = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            SystemMessage(content=f"当前工作记忆上下文: {context}"),
            *messages,
            HumanMessage(content=(
                "请分析用户的最新消息，返回应该路由到的Agent名称。"
                "只返回以下之一: knowledge_rag, ticket_handler, compliance_checker"
            )),
        ]

        response = await self.llm.ainvoke(routing_prompt)
        intent = response.content.strip().lower()

        # 第一道校验：LLM 输出规范化后不在合法集合内则回落。
        # 条件边一侧的 route_to_agent 是第二道
        if intent not in VALID_INTENTS:
            intent = "knowledge_rag"

        # 决策产生后写回，供下一轮读取（跨轮存活由此保证）
        self.working_memory.update(session_id, {"last_intent": intent})

        # 只返回变更键（delta 约定）。LangGraph 节点返回增量更新，
        # 未提到的字段保持原值
        return {
            "intent": intent,
            "current_agent": "supervisor",
        }

    @trace_agent_call("supervisor_synthesize")
    async def synthesize_response(self, state: AgentState) -> dict[str, Any]:
        """汇总子 Agent 结果，生成最终回复（synthesize 节点）"""
        sub_results = state.get("sub_results", {})
        compliance_passed = state.get("compliance_passed", True)

        if not compliance_passed:
            # 合规未过：不透出子 Agent 原文，走固定转人工话术
            final_response = (
                "抱歉，您的请求涉及敏感内容，已转交人工客服处理。"
                "工单编号已自动生成，请留意后续通知。"
            )
        else:
            # 只拼接非空字符串：compliance 节点塞进 sub_results 的
            # dict 混入 join 会抛 TypeError；合规详情仍可从 state 的
            # sub_results 里读，只是不进最终回复
            result_parts = [
                result for result in sub_results.values()
                if isinstance(result, str) and result
            ]
            final_response = (
                "\n\n".join(result_parts) if result_parts
                else "抱歉，暂时无法处理您的请求，请稍后重试。"
            )

        # final_response 以 AIMessage 追加进 messages（第二个写入点），
        # 下一轮的路由 prompt 与子 Agent 都能看到历史回复
        return {
            "final_response": final_response,
            "messages": [AIMessage(content=final_response)],
        }


def route_to_agent(state: AgentState) -> str:
    """
    条件边路由函数：校验后返回分支键（合法意图名）。

    键到节点名的翻译由图工厂的 path_map 完成（意图名与节点名不一致
    的 compliance_checker → compliance_check 在那里吸收）。注意路由
    函数返回的必须是 path_map 里存在的键，直接返回节点名而 path_map
    以意图名为键会错位 KeyError。route_decision 已校验过 intent，
    这里是第二道兜底：意外值一律回落 knowledge_rag，保证图不因路由
    函数而中断。
    """
    intent = state.get("intent", "knowledge_rag")
    return intent if intent in VALID_INTENTS else "knowledge_rag"
