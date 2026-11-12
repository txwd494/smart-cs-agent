"""
agents.ticket_handler — 工单处理 Agent（业务办理路径的执行者）

三类路径里唯一的有状态事务：创建的工单有编号、可凭编号查回，
ticket_id 是用户与系统之间的跨轮凭据。数据流：process 读最后一条
消息与 user_id → analyze_request 一次 LLM 调用提取结构化工单信息
（action / ticket_type / priority / summary / details / ticket_id，
JSON）→ 按 action 分支执行 → TicketStore 读写 → 格式化回复进
sub_results。分析与执行分离：LLM 做理解，代码做事务。

存储说明：进程内 TicketStore，无持久化，重启即失。外部工单系统
的接入走 MCP 工具层（mcp/tools/default_tools.py 的 ticket_create
落同一个 store），本模块不直接涉及 MCP。

LLM 输出的防护设计（五处，均为工单提取的真实风险面）：
1. 解析走共享助手 parse_llm_json（agents/llm_json.py）：
   对整个响应直 json.loads 在 LLM 带围栏输出时失败，提取结果
   被静默丢弃
2. 字段校验回落：action / ticket_type / priority 对合法集合校验，
   非法值回落默认。自由字符串直接入库与回显会把 LLM 标签漂移
   （如中文「退款」）原样写进工单
3. prompt 动作只列 create / query，且 schema 带 ticket_id 字段：
   schema 允许 update 而 process 无此分支时，update 落进 else
   等于静默建一张新单（想改状态的用户被开了新单）；查询分支
   依赖 ticket_id，schema 不声明它等于靠模型自觉补字段
4. query 缺 ticket_id 时返回「请提供工单号」且不建单：
   此情形落进建单分支的话，用户问进度，收获一张新工单
5. process 按 delta 约定返回变更键

已知限制（留作后续优化）：analyze 只看最后一条消息，跨轮断裂，
第一轮建单拿到号、第二轮问进度不带号时查不到上轮工单（号码在
messages 历史里，analyze 看不见）；存储无持久化，ticket_id 的用户
预期生命周期长于进程。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.llm_json import parse_llm_json
from agents.state import AgentState
from tracing.otel_config import trace_agent_call


class TicketStatus(str, Enum):
    """工单状态（update 流转接入前，实际写入的只有 created）"""
    CREATED = "created"
    PROCESSING = "processing"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class TicketPriority(str, Enum):
    """工单优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# 合法取值集合（校验依据，与 prompt 的说明一一对应）
VALID_ACTIONS = {"create", "query"}
VALID_TICKET_TYPES = {
    "refund", "claim", "account_open", "account_change", "complaint", "general",
}
VALID_PRIORITIES = {p.value for p in TicketPriority}


TICKET_SYSTEM_PROMPT = """你是一个专业的工单处理Agent，负责处理客户的业务办理请求。

你的职责：
1. 分析用户需求，判断是否需要创建工单
2. 提取工单关键信息（类型、优先级、描述）
3. 创建工单并返回工单号
4. 查询现有工单状态

工单类型：
- refund: 退款申请
- claim: 理赔申请
- account_open: 开户申请
- account_change: 账户变更
- complaint: 投诉工单
- general: 通用工单

优先级判断规则：
- urgent: 资金安全、账户被盗
- high: 退款超时、理赔争议
- medium: 常规业务办理
- low: 信息咨询类

请以JSON格式返回工单信息：
{
    "action": "create|query",
    "ticket_type": "refund|claim|account_open|...",
    "priority": "low|medium|high|urgent",
    "summary": "工单摘要",
    "details": "详细描述",
    "ticket_id": "用户要查询工单时填入其提供的工单号，创建时留空"
}
"""


class TicketStore:
    """内存工单存储（无持久化，重启即失；生产环境应替换为数据库）"""

    def __init__(self):
        self._tickets: dict[str, dict] = {}

    def create(self, ticket_type: str, priority: str, summary: str, details: str, user_id: str) -> dict:
        """建单：编号 TK-日期-六位随机，创建与更新时间同值起步"""
        ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "type": ticket_type,
            "priority": priority,
            "status": TicketStatus.CREATED.value,
            "summary": summary,
            "details": details,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._tickets[ticket_id] = ticket
        return ticket

    def query(self, ticket_id: str) -> dict | None:
        """
        按工单号精确查询。

        返回的是存储内 dict 的引用，调用方修改会污染存储。与
        working_memory 的拷贝约定不同：本类是包内组件，消费方只有
        本 Agent 的只读格式化，从简；需要严格隔离时可参照
        working_memory 的双层拷贝做法。
        """
        return self._tickets.get(ticket_id)

    def query_by_user(self, user_id: str) -> list[dict]:
        """按用户查询其全部工单（当前无调用方，预留接口）"""
        return [t for t in self._tickets.values() if t["user_id"] == user_id]

    def update_status(self, ticket_id: str, status: str) -> dict | None:
        """
        更新工单状态（当前无调用方：action 的 update 分支未实现，
        留作后续优化；预留接口）。
        """
        ticket = self._tickets.get(ticket_id)
        if ticket:
            ticket["status"] = status
            ticket["updated_at"] = datetime.now().isoformat()
        return ticket


class TicketHandlerAgent:
    """工单处理 Agent（结构与差异见模块 docstring）"""

    def __init__(self, llm: ChatOpenAI, ticket_store: TicketStore | None = None):
        self.llm = llm
        self.ticket_store = ticket_store or TicketStore()

    @trace_agent_call("ticket_analyze")
    async def analyze_request(self, user_message: str) -> dict[str, Any]:
        """分析用户需求，提取并校验工单信息（一次 LLM 调用）"""
        messages = [
            SystemMessage(content=TICKET_SYSTEM_PROMPT),
            HumanMessage(content=f"用户消息: {user_message}"),
        ]

        response = await self.llm.ainvoke(messages)
        return self._validate(parse_llm_json(response.content), user_message)

    @staticmethod
    def _validate(data: dict[str, Any] | None, user_message: str = "") -> dict[str, Any]:
        """
        校验并补全 LLM 返回的工单信息（差异点 2 的落点）。

        任何字段异常都回落默认值，保证返回的 dict 字段完整且取值合法。
        解析失败（data 为 None）或摘要为空时用用户原文兜底，
        兜底工单仍携带上下文（降级语义）。
        """
        if data is None:
            data = {}

        action = data.get("action", "create")
        if action not in VALID_ACTIONS:
            action = "create"

        ticket_type = data.get("ticket_type", "general")
        if ticket_type not in VALID_TICKET_TYPES:
            ticket_type = "general"

        priority = data.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        summary = str(data.get("summary", "")).strip()[:100]
        if not summary:
            summary = user_message[:100]
        details = str(data.get("details", "")).strip()
        if not details:
            details = user_message

        return {
            "action": action,
            "ticket_type": ticket_type,
            "priority": priority,
            "summary": summary,
            "details": details,
            "ticket_id": str(data.get("ticket_id", "")).strip(),
        }

    @trace_agent_call("ticket_create")
    async def create_ticket(self, ticket_info: dict[str, Any], user_id: str) -> str:
        """创建工单，返回给用户看的回执文本"""
        ticket = self.ticket_store.create(
            ticket_type=ticket_info["ticket_type"],
            priority=ticket_info["priority"],
            summary=ticket_info["summary"],
            details=ticket_info["details"],
            user_id=user_id,
        )

        priority_label = {
            "low": "普通", "medium": "中等", "high": "高", "urgent": "紧急"
        }.get(ticket["priority"], "中等")

        return (
            f"工单已创建成功！\n\n"
            f"📋 工单号: {ticket['ticket_id']}\n"
            f"📝 类型: {ticket['type']}\n"
            f"⚡ 优先级: {priority_label}\n"
            f"📄 摘要: {ticket['summary']}\n"
            f"🕐 创建时间: {ticket['created_at']}\n\n"
            f"我们将尽快处理您的请求，请保存好工单号以便后续查询。"
        )

    @trace_agent_call("ticket_query")
    async def query_ticket(self, ticket_id: str) -> str:
        """按工单号查询状态，返回给用户看的查询结果文本"""
        ticket = self.ticket_store.query(ticket_id)
        if not ticket:
            return f"未找到工单号 {ticket_id}，请确认工单号是否正确。"

        status_label = {
            "created": "已创建",
            "processing": "处理中",
            "pending_review": "待审核",
            "resolved": "已解决",
            "closed": "已关闭",
            "escalated": "已升级",
        }.get(ticket["status"], ticket["status"])

        return (
            f"工单查询结果：\n\n"
            f"📋 工单号: {ticket['ticket_id']}\n"
            f"📊 状态: {status_label}\n"
            f"📝 类型: {ticket['type']}\n"
            f"📄 摘要: {ticket['summary']}\n"
            f"🕐 创建时间: {ticket['created_at']}\n"
            f"🔄 更新时间: {ticket['updated_at']}"
        )

    @trace_agent_call("ticket_handler_process")
    async def process(self, state: AgentState) -> dict[str, Any]:
        """作为 Graph 节点处理状态：分析 → 按 action 分支执行。"""
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        if not messages:
            return {}

        # 隐含假设：最后一条消息是本轮用户消息
        #（成立前提同 knowledge_rag.process，见其注释）
        last_message = messages[-1].content
        ticket_info = await self.analyze_request(last_message)

        if ticket_info["action"] == "query":
            if ticket_info["ticket_id"]:
                result = await self.query_ticket(ticket_info["ticket_id"])
            else:
                # 无号查询返回要号话术，不建单：
                # 此情形若落进建单分支，用户问进度收获一张新工单
                result = (
                    "请提供您要查询的工单号（格式如 TK-20261017-A1B2C3），"
                    "以便为您查询处理进度。"
                )
        else:
            result = await self.create_ticket(ticket_info, user_id)

        return {
            "sub_results": {
                **state.get("sub_results", {}),
                "ticket_handler": result,
            },
        }
