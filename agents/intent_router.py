"""
agents.intent_router — 意图路由 Agent（专职用户意图分类）

定位：对单条用户消息做结构化意图分类，输出五维结果：
一级意图 / 二级意图 / 置信度 / 关键实体 / 建议路由。
与 Supervisor.route_decision 的设计分工是上下游：
- 本模块：细粒度理解（五维结构化），但只看最新一条消息
- Supervisor：会话级调度（读完整历史 + working memory），输出单标签

架构现状（重要）：本模块当前未接入图。Supervisor 的 route_decision
以单标签路由覆盖了主流程需求，专职分类器的增量价值（置信度阈值
回落、实体贯通下游）要等接线才兑现，因此整合排在后续优化（方案
已定：Supervisor 内部委托 classify，图拓扑不动）。接线前本模块
没有运行时调用方，验证走独立冒烟与 pytest。

解析与防护设计（四处，均为 LLM 输出的真实风险面）：
1. JSON 解析走共享助手 parse_llm_json（agents/llm_json.py）：
   截取首个 { 到最后一个 } 再 json.loads，markdown 围栏
   （```json ... ```）与前后解释文字天然兼容。直解析在 LLM
   带围栏输出时失败，分类结果被静默丢弃且无痕迹
2. 枚举转换纳入保护：LLM 标签漂移是常态（如返回中文「咨询」），
   IntentCategory(...) 的 ValueError 必须兜住，非法值回落
   unknown。confidence 与 entities 的类型防护同此（字符串数字
   转 float、非 dict 实体回落空）
3. suggested_agent 校验：不在合法路由目标内回落 knowledge_rag，
   合法集合复用 supervisor.VALID_INTENTS（单一事实源，避免两处
   定义漂移）。本模块未接线，不能依赖图内 route_to_agent 的兜底
4. process 按 delta 约定返回变更键
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.llm_json import parse_llm_json
from agents.supervisor import VALID_INTENTS
from tracing.otel_config import trace_agent_call


class IntentCategory(str, Enum):
    """一级意图分类"""
    CONSULTATION = "consultation"       # 咨询类
    COMPLAINT = "complaint"             # 投诉类
    TRANSACTION = "transaction"         # 交易/办理类
    ACCOUNT = "account"                 # 账户类
    COMPLIANCE = "compliance"           # 合规相关
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """意图识别结果（五维输出）"""
    primary_intent: IntentCategory      # 一级意图
    secondary_intent: str               # 二级意图（具体业务子类型）
    confidence: float                   # 置信度 0.0-1.0
    entities: dict[str, str]            # 关键实体（订单号/产品名/金额）
    suggested_agent: str                # 建议路由（合法路由目标之一）


INTENT_SYSTEM_PROMPT = """你是一个专业的意图识别Agent，负责分析用户的客服消息。

请从以下维度分析用户意图：
1. 一级意图分类: consultation(咨询), complaint(投诉), transaction(交易办理), account(账户), compliance(合规)
2. 二级意图: 具体的业务子类型
3. 置信度: 0.0-1.0
4. 关键实体: 提取订单号、产品名、金额等关键信息
5. 建议路由: knowledge_rag(知识查询), ticket_handler(工单处理), compliance_checker(合规审查)

以JSON格式返回，示例：
{
    "primary_intent": "consultation",
    "secondary_intent": "product_inquiry",
    "confidence": 0.95,
    "entities": {"product": "理财产品A"},
    "suggested_agent": "knowledge_rag"
}

金融场景特殊规则：
- 涉及资金安全、账户异常、欺诈举报 → compliance_checker
- 涉及退款、理赔、开户 → ticket_handler
- 涉及产品咨询、利率查询、政策了解 → knowledge_rag
"""


class IntentRouterAgent:
    """意图路由 Agent（当前未接入图，见模块 docstring 的架构现状）"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @trace_agent_call("intent_router")
    async def classify(self, user_message: str) -> IntentResult:
        """对单条用户消息做意图分类（一次 LLM 调用）"""
        messages = [
            SystemMessage(content=INTENT_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]

        response = await self.llm.ainvoke(messages)
        return self._to_intent_result(parse_llm_json(response.content))

    def _to_intent_result(self, data: dict[str, Any] | None) -> IntentResult:
        """
        dict → IntentResult，带完整兜底（差异点 2 / 3 的落点）。

        所有字段单独防护，任何一处值异常都回落默认值，
        保证 classify 永远返回合法 IntentResult 而不抛异常。
        """
        if data is None:
            data = {}

        # 一级意图：不在枚举内（标签漂移）回落 unknown；
        # 注意必须包在 try 里，ValueError 会直接崩
        try:
            primary = IntentCategory(data.get("primary_intent", "unknown"))
        except ValueError:
            primary = IntentCategory.UNKNOWN

        # 置信度：容忍字符串数字（"0.85"），非法回落 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # 实体：LLM 偶尔返回列表等非 dict 形态，回落空
        entities = data.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        # 建议路由：必须落在合法集合内，否则回落 knowledge_rag。
        # 复用 supervisor.VALID_INTENTS 保持单一事实源
        suggested = data.get("suggested_agent", "knowledge_rag")
        if suggested not in VALID_INTENTS:
            suggested = "knowledge_rag"

        return IntentResult(
            primary_intent=primary,
            secondary_intent=data.get("secondary_intent", "unknown"),
            confidence=confidence,
            entities=entities,
            suggested_agent=suggested,
        )

    @trace_agent_call("intent_router_process")
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        作为 Graph 节点处理状态（预留接口，当前未接线）。

        已知的结构问题：把五维分类详情以 dict 塞进 sub_results，与
        给用户的回复文本混装同一容器（compliance 节点也是这个形状，
        synthesize 靠类型过滤才不崩）。接线整合时应改为独立的
        state 字段
        或写入 working memory。
        """
        messages = state.get("messages", [])
        if not messages:
            return {}

        last_message = messages[-1].content
        intent_result = await self.classify(last_message)

        return {
            "intent": intent_result.suggested_agent,
            "sub_results": {
                **state.get("sub_results", {}),
                "intent_router": {
                    "primary": intent_result.primary_intent.value,
                    "secondary": intent_result.secondary_intent,
                    "confidence": intent_result.confidence,
                    "entities": intent_result.entities,
                },
            },
        }
