"""test_ticket_handler — 1.7 冒烟断言的固化（store / 校验 / stub 流程）"""

import re

from agents.ticket_handler import (
    VALID_PRIORITIES,
    VALID_TICKET_TYPES,
    TicketHandlerAgent,
    TicketStore,
)
from langchain_core.messages import HumanMessage
from tests.conftest import StubLLM

TK_RE = re.compile(r"TK-\d{8}-[A-Z0-9]{6}")


# ── TicketStore 纯逻辑 ─────────────────────────────────────────

def test_store_create_and_query():
    store = TicketStore()
    ticket = store.create("refund", "high", "退款", "详情", "u1")
    assert TK_RE.fullmatch(ticket["ticket_id"])
    assert ticket["status"] == "created"
    assert store.query(ticket["ticket_id"])["type"] == "refund"


def test_store_query_missing():
    assert TicketStore().query("TK-00000000-XXXXXX") is None


def test_store_query_by_user():
    store = TicketStore()
    store.create("refund", "high", "a", "d", "u1")
    store.create("claim", "low", "b", "d", "u2")
    mine = store.query_by_user("u1")
    assert len(mine) == 1 and mine[0]["summary"] == "a"


def test_store_update_status():
    store = TicketStore()
    t = store.create("refund", "high", "a", "d", "u1")
    updated = store.update_status(t["ticket_id"], "processing")
    assert updated["status"] == "processing"
    assert store.update_status("TK-00000000-XXXXXX", "x") is None


# ── 字段校验回落（2.4 差异 2 同款防护，Agent 侧是回落非报错） ──

def test_validate_valid_set_constants():
    # 合法集合单一事实源：MCP 工具与 Agent 共用
    assert VALID_PRIORITIES == {"low", "medium", "high", "urgent"}
    assert "refund" in VALID_TICKET_TYPES


# ── stub LLM 流程 ──────────────────────────────────────────────

def _state(message, user="u1"):
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user, "session_id": "s1",
    }


async def test_process_create_branch():
    stub = StubLLM([], default=(
        '{"action": "create", "ticket_type": "refund", "priority": "high", '
        '"summary": "退款申请", "details": "订单未到账", "ticket_id": ""}'
    ))
    agent = TicketHandlerAgent(llm=stub)
    out = await agent.process(_state("帮我创建退款工单"))
    reply = out["sub_results"]["ticket_handler"]
    m = TK_RE.search(reply)
    assert m, reply
    assert agent.ticket_store.query(m.group(0))["type"] == "refund"


async def test_process_query_without_id_asks_and_no_ticket():
    stub = StubLLM([], default=(
        '{"action": "query", "ticket_type": "general", "priority": "medium", '
        '"summary": "", "details": "", "ticket_id": ""}'
    ))
    agent = TicketHandlerAgent(llm=stub)
    before = len(agent.ticket_store._tickets)
    out = await agent.process(_state("我的工单处理得怎么样了"))
    assert "工单号" in out["sub_results"]["ticket_handler"]
    # 边界锚点：无号查询不得建单
    assert len(agent.ticket_store._tickets) == before


async def test_process_query_with_id():
    stub = StubLLM([], default=(
        '{"action": "query", "ticket_type": "general", "priority": "medium", '
        '"summary": "", "details": "", "ticket_id": "TK-20260101-ABCDEF"}'
    ))
    agent = TicketHandlerAgent(llm=stub)
    ticket = agent.ticket_store.create("refund", "high", "s", "d", "u1")
    # 覆盖 stub 里的固定号：用真实号
    stub.default = (
        '{"action": "query", "ticket_type": "general", "priority": "medium", '
        f'"summary": "", "details": "", "ticket_id": "{ticket["ticket_id"]}"}}'
    )
    out = await agent.process(_state(f"查询工单 {ticket['ticket_id']}"))
    assert ticket["ticket_id"] in out["sub_results"]["ticket_handler"]
