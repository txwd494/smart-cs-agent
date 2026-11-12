"""
mcp.tools.default_tools — 默认工具集（真实后端 + 演示桩）

四个工具（order_query / knowledge_search / ticket_create /
risk_check），后端分两类：

- 真实后端：knowledge_search 走 LongTermMemory.search（真实
  语义检索，embedding 不可用时降级二元组）；ticket_create 落
  共享 TicketStore（与图内工单 Agent 同一张单，store 单例由
  API lifespan 构造，图工厂与本模块各注入一份，图内建的单
  MCP 凭号可查，反向亦然）
- 演示桩：order_query 与 risk_check 无真实后端可接（本系统没有
  订单与风控组件），返回样例数据与固定阈值策略，description
  已标注演示用途，生产不应依赖其结果

字段映射：ticket_create 对外 schema 是工具协议侧的四字段
（title / category），存储字段是工单体系自己的（summary /
ticket_type），映射发生在 handler 内部，两边互不渗透。

校验策略与图内 Agent 的差异（有意为之）：Agent 面对的是 LLM
输出，标签漂移常见，非法值回落默认值保持可用（见
ticket_handler 的防护设计）；工具面对的是程序化调用方，
schema 即契约，非法取值明确报错。合法集合直接复用
ticket_handler 的模块级常量，单一事实源。

后端未注入时（测试或裁剪部署）：工具仍注册，调用时报配置
错误。注册与配置解耦，list_tools 的输出稳定。
"""

from __future__ import annotations

from agents.ticket_handler import (
    VALID_PRIORITIES,
    VALID_TICKET_TYPES,
    TicketStore,
)
from memory.long_term import LongTermMemory
from mcp.server import MCPToolServer


def create_default_tools(
    server: MCPToolServer,
    long_term_memory: LongTermMemory | None = None,
    ticket_store: TicketStore | None = None,
) -> MCPToolServer:
    """
    注册默认工具集。返回传入的 server（链式习惯）。

    Args:
        server: 工具服务器
        long_term_memory: 长期记忆（knowledge_search 的检索后端）
        ticket_store: 工单存储（ticket_create 的落库后端，与图共享）
    """

    @server.register(
        name="order_query",
        description="查询订单信息（演示桩：返回样例数据，无真实订单系统）",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
                "user_id": {"type": "string", "description": "用户ID"},
            },
        },
        category="order",
    )
    async def order_query(order_id: str = "", user_id: str = "") -> dict:
        # 演示桩：无论查什么都返回同一单样例数据
        return {
            "order_id": order_id or "ORD-20260401-001",
            "status": "shipped",
            "amount": 299.00,
            "product": "智能理财产品A",
            "created_at": "2026-04-01T10:00:00",
        }

    @server.register(
        name="knowledge_search",
        description="搜索企业知识库，返回语义检索的相关文档片段",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 3},
            },
            "required": ["query"],
        },
        category="knowledge",
    )
    async def knowledge_search(query: str, top_k: int = 3) -> list[dict]:
        if long_term_memory is None:
            raise RuntimeError("知识后端未配置：LongTermMemory 未注入")
        docs = await long_term_memory.search(query, top_k=top_k)
        # 返回形状：内容、来源、得分（内部 id 与块元数据不外露）
        return [
            {"content": d["content"], "source": d["source"], "score": d["score"]}
            for d in docs
        ]

    @server.register(
        name="ticket_create",
        description="创建客服工单（落共享工单存储，图内工单 Agent 凭工单号可查到同一张单）",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "工单摘要"},
                "description": {"type": "string", "description": "详细描述"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "category": {
                    "type": "string",
                    "description": "工单类型",
                    "enum": ["refund", "claim", "account_open", "account_change", "complaint", "general"],
                },
                "user_id": {"type": "string", "description": "发起方标识，默认 mcp_client"},
            },
            "required": ["title", "description"],
        },
        category="ticket",
    )
    async def ticket_create(
        title: str,
        description: str,
        priority: str = "medium",
        category: str = "general",
        user_id: str = "mcp_client",
    ) -> dict:
        if ticket_store is None:
            raise RuntimeError("工单后端未配置：TicketStore 未注入")
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"非法 priority: {priority!r}，合法值 {sorted(VALID_PRIORITIES)}")
        if category not in VALID_TICKET_TYPES:
            raise ValueError(f"非法 category: {category!r}，合法值 {sorted(VALID_TICKET_TYPES)}")
        # 字段映射（工具协议侧 schema 与工单体系自己的存储字段）
        return ticket_store.create(
            ticket_type=category,
            priority=priority,
            summary=title,
            details=description,
            user_id=user_id,
        )

    @server.register(
        name="risk_check",
        description="风控接口（演示策略：按金额阈值分级，无真实风控系统）",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "action": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["user_id", "action"],
        },
        category="compliance",
    )
    async def risk_check(user_id: str, action: str, amount: float = 0.0) -> dict:
        # 演示策略：三档阈值
        if amount > 50000:
            risk_level = "high"
        elif amount > 10000:
            risk_level = "medium"
        else:
            risk_level = "low"
        return {
            "user_id": user_id,
            "action": action,
            "risk_level": risk_level,
            "requires_manual_review": risk_level == "high",
        }

    return server
