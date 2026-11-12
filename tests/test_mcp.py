"""test_mcp — 2.3 冒烟断言的固化（协议层 / 校验 / 真实后端注入）"""

from agents.ticket_handler import TicketStore
from mcp.server import MCPToolServer
from mcp.tools.default_tools import create_default_tools
from tests.conftest import FakeEmbeddings

TOOL_NAMES = {"order_query", "knowledge_search", "ticket_create", "risk_check"}


def _server(**kwargs) -> MCPToolServer:
    return create_default_tools(MCPToolServer(), **kwargs)


# ── 发现与过滤 ──────────────────────────────────────────────────

async def test_list_tools_shape():
    tools = _server().list_tools()
    assert {t["name"] for t in tools} == TOOL_NAMES
    for t in tools:
        assert set(t) == {"name", "description", "inputSchema", "category"}


async def test_list_tools_category_filter():
    server = _server()
    assert [t["name"] for t in server.list_tools(category="knowledge")] == ["knowledge_search"]
    assert server.list_tools(category="nope") == []


# ── 校验次序 ────────────────────────────────────────────────────

async def test_unknown_tool():
    r = await _server().call_tool("no_such", {})
    assert not r.success and "not found" in r.error and "knowledge_search" in r.error


async def test_required_blocks_before_handler():
    server = MCPToolServer()
    calls = []

    @server.register(
        name="probe", description="探针",
        input_schema={"type": "object", "properties": {"must": {"type": "string"}},
                      "required": ["must"]},
    )
    async def probe(must: str):
        calls.append(must)

    r = await server.call_tool("probe", {})
    assert not r.success and "must" in r.error and calls == []
    r = await server.call_tool("probe", {"must": "x"})
    assert r.success and calls == ["x"]


async def test_backend_not_configured_raises_into_result():
    r = await _server().call_tool("knowledge_search", {"query": "x"})
    assert not r.success and "未配置" in r.error


# ── 演示桩与 risk_check 阈值 ────────────────────────────────────

async def test_order_query_stub():
    r = await _server().call_tool("order_query", {})
    assert r.success and r.result["order_id"] == "ORD-20260401-001"


async def test_risk_check_thresholds():
    server = _server()
    for amount, level in [(100, "low"), (20000, "medium"), (60000, "high")]:
        r = await server.call_tool("risk_check", {"user_id": "u", "action": "buy", "amount": amount})
        assert r.result["risk_level"] == level
        assert r.result["requires_manual_review"] == (level == "high")


async def test_risk_check_missing_required():
    r = await _server().call_tool("risk_check", {"user_id": "u"})
    assert not r.success and "action" in r.error


# ── 真实后端 ────────────────────────────────────────────────────

async def test_knowledge_search_real_backend(fake_embeddings):
    from memory.long_term import LongTermMemory

    ltm = LongTermMemory(embeddings=fake_embeddings)
    await ltm.add_document("理财产品A年化收益率3.5%", source="product_faq.txt")
    server = _server(long_term_memory=ltm)
    r = await server.call_tool("knowledge_search", {"query": "理财"})
    assert r.success and set(r.result[0]) == {"content", "source", "score"}
    assert r.result[0]["source"] == "product_faq.txt"


async def test_ticket_create_shared_store():
    store = TicketStore()
    server = _server(ticket_store=store)
    r = await server.call_tool("ticket_create", {
        "title": "退款申请", "description": "订单未到账",
        "priority": "high", "category": "refund",
    })
    assert r.success
    assert store.query(r.result["ticket_id"])["type"] == "refund"
    assert r.result["priority"] == "high"


async def test_ticket_create_invalid_enum_errors():
    server = _server(ticket_store=TicketStore())
    r = await server.call_tool("ticket_create", {"title": "t", "description": "d", "priority": "urgenttt"})
    assert not r.success and "priority" in r.error
    r = await server.call_tool("ticket_create", {"title": "t", "description": "d", "category": "foo"})
    assert not r.success and "category" in r.error


async def test_ticket_create_defaults():
    server = _server(ticket_store=TicketStore())
    r = await server.call_tool("ticket_create", {"title": "t", "description": "d"})
    assert r.result["type"] == "general" and r.result["priority"] == "medium"
    assert r.result["user_id"] == "mcp_client"


# ── JSON-RPC 协议层 ─────────────────────────────────────────────

async def test_jsonrpc_ping():
    resp = await _server().handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert resp == {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}


async def test_jsonrpc_tools_call():
    resp = await _server().handle_jsonrpc({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "order_query", "arguments": {}},
    })
    assert resp["id"] == 2 and resp["result"]["success"]


async def test_jsonrpc_unknown_method():
    resp = await _server().handle_jsonrpc({"jsonrpc": "2.0", "id": 3, "method": "foo"})
    assert resp["error"]["code"] == -32601


# ── 调用日志有界 ────────────────────────────────────────────────

async def test_log_bounded():
    server = MCPToolServer(max_log=5)
    create_default_tools(server)
    for _ in range(8):
        await server.call_tool("risk_check", {"user_id": "u", "action": "a"})
    assert len(server.get_call_log(last_n=100)) == 5
