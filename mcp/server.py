"""
mcp.server — MCP 工具协议服务端（进程内工具注册 / 发现 / 调用中心）

Model Context Protocol 的协议层实现。工具用 ToolDefinition 声明
（名称、描述、输入 schema、handler），语义上对应 MCP 的三个动作：
tools/list（发现）、tools/call（调用）、注册（服务器侧准备）。
消息格式是 JSON-RPC 2.0。

对外有两条消费路径，都在 api/main.py：
- REST 风格：GET /api/tools 与 POST /api/tools/call，直接调方法
- 协议风格：POST /api/mcp，消息体是 JSON-RPC 请求，走本模块的
  handle_jsonrpc

位置：图之外的旁路服务。图内 Agent 不在执行中调用工具（见已知
边界），MCP 面向外部客户端（运维面板、其他服务、前端调试台）。

实现要点（六处）：
1. handle_jsonrpc 经 POST /api/mcp 实际接线：协议层要有真实
   消费路径，只写不调等于死代码
2. required 校验：call_tool 入口对照 input_schema 的 required
   列表查参数，缺字段直接失败，不进 handler。schema 只做展示的
   话，参数透传 **arguments，缺参要到运行时 TypeError 才暴露，
   required 声明形同虚设
3. 调用日志有界（默认 500 条，超出丢最旧）。append-only 无上界
   会让长进程内存只增不减，get_call_log 的 last_n 只控制展示
   不控制存储
4. 计时用 time.perf_counter()（单调时钟，与 tracing 模块一致）；
   time.time() 受系统时间跳变影响
5. 默认工具接真实后端（mcp/tools/default_tools.py）：
   knowledge_search 走 LongTermMemory（真实语义检索），
   ticket_create 落共享 TicketStore（与图内工单 Agent 同一张单，
   单例由 API lifespan 构造注入两处）。工具返回假数据、与系统
   真实存储互不相通，等于两套平行体系
6. call_tool 开 OTel span：名称 mcp.tool.{工具名}（工具名进
   span 名，后端可按工具过滤），属性记 success 与 error。工具
   调用不进 AgentMetrics：tool_call_log 已承载逐条记录，重复
   聚合无增量，两套观测各自干净

已知边界：无 MCP 传输层（stdio / Streamable HTTP 均未实现，协议
消息只经 HTTP 端点直达），无 initialize 握手与能力协商，
resources / prompts 两类能力未实现；requires_auth 只是标注，无
鉴权检查；Agent 不在图执行中调用工具。做符合规范的服务器需要
官方 mcp SDK（FastMCP），留作后续升级。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from tracing.otel_config import business_span


@dataclass
class ToolDefinition:
    """工具定义：声明即契约。list_tools 对外可见，required 参与校验"""

    name: str
    description: str
    # JSON Schema（type/properties/required）。required 列表在
    # call_tool 入口实际生效（差异 2），其余键供调用方参考
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    category: str = "general"
    # 仅标注（哪些工具该被保护），无检查逻辑（见模块 docstring 已知边界）
    requires_auth: bool = False


@dataclass
class ToolCallResult:
    """工具调用结果：成功带负载，或失败带错误说明，附耗时与时间戳"""

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MCPToolServer:
    """MCP 工具服务端（架构与差异见模块 docstring）"""

    def __init__(self, max_log: int = 500):
        self._tools: dict[str, ToolDefinition] = {}
        self._call_log: list[ToolCallResult] = []
        self._max_log = max_log

    # ── 注册 ─────────────────────────────────────────────────────

    def register_tool(self, tool: ToolDefinition) -> None:
        """注册工具（同名覆盖：重新注册即更新）"""
        self._tools[tool.name] = tool

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        category: str = "general",
        requires_auth: bool = False,
    ) -> Callable:
        """工具注册装饰器：声明与实现写在同一处"""

        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            self.register_tool(ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func,
                category=category,
                requires_auth=requires_auth,
            ))
            return func

        return decorator

    # ── 发现与调用 ───────────────────────────────────────────────

    def list_tools(self, category: str | None = None) -> list[dict]:
        """工具发现（JSON-RPC 方法 tools/list）。可按 category 过滤"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
                "category": t.category,
            }
            for t in self._tools.values()
            if category is None or t.category == category
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        """
        工具调用（JSON-RPC 方法 tools/call）。

        校验次序：存在性 → required 字段 → handler 执行。前两关失败
        不进 handler。执行期异常捕获进结果而非上抛：工具层失败是
        业务结果，与协议层失败（未知方法）分工，见 handle_jsonrpc。
        整个过程包在 mcp.tool.{工具名} span 里（含校验，工具名进
        span 名，见模块 docstring 差异 6）。
        """
        arguments = arguments or {}

        with business_span(f"mcp.tool.{name}", **{"mcp.tool_name": name}) as span:
            tool = self._tools.get(name)
            if tool is None:
                result = ToolCallResult(
                    tool_name=name,
                    success=False,
                    error=f"Tool '{name}' not found. Available: {sorted(self._tools)}",
                )
                self._log(result)
                self._finish_span(span, result)
                return result

            missing = [f for f in tool.input_schema.get("required", []) if f not in arguments]
            if missing:
                result = ToolCallResult(
                    tool_name=name,
                    success=False,
                    error=f"Missing required arguments: {missing}",
                )
                self._log(result)
                self._finish_span(span, result)
                return result

            start = time.perf_counter()
            try:
                output = await tool.handler(**arguments)
                result = ToolCallResult(
                    tool_name=name,
                    success=True,
                    result=output,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as e:
                result = ToolCallResult(
                    tool_name=name,
                    success=False,
                    error=str(e),
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            self._log(result)
            self._finish_span(span, result)
            return result

    @staticmethod
    def _finish_span(span: Any, result: ToolCallResult) -> None:
        """span 收尾属性（business_span 降级时 span 为 None，静默跳过）"""
        if span is None:
            return
        span.set_attribute("mcp.success", result.success)
        if result.error:
            span.set_attribute("mcp.error", result.error)

    # ── JSON-RPC 协议层 ─────────────────────────────────────────

    async def handle_jsonrpc(self, request: dict) -> dict:
        """
        JSON-RPC 2.0 消息处理（POST /api/mcp 的接线点）。

        工具层失败（未知工具、缺参数、执行异常）落在 result 里的
        success=false；协议层失败（未知方法、消息结构非法）才是
        error 信封。这一分工与 MCP 规范一致：调用的业务结果走
        result，协议自身的错误走 error。
        """
        try:
            method = request.get("method", "")
            params = request.get("params") or {}
            req_id = request.get("id", 1)

            if method == "tools/list":
                result = self.list_tools(category=params.get("category"))
            elif method == "tools/call":
                call_result = await self.call_tool(
                    params.get("name", ""),
                    params.get("arguments") or {},
                )
                result = {
                    "success": call_result.success,
                    "result": call_result.result,
                    "error": call_result.error,
                }
            elif method == "ping":
                result = {"status": "ok"}
            else:
                return self._jsonrpc_error(req_id, -32601, f"Method not found: {method}")

            return {"jsonrpc": "2.0", "result": result, "id": req_id}

        except Exception as e:
            # request 非 dict、params 结构非法等落这里（-32603 internal error）
            req_id = request.get("id", 1) if isinstance(request, dict) else 1
            return self._jsonrpc_error(req_id, -32603, str(e))

    @staticmethod
    def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }

    # ── 观测 ─────────────────────────────────────────────────────

    def get_call_log(self, last_n: int = 100) -> list[dict]:
        """
        最近工具调用日志。不含 result 负载：日志是观测数据，
        避免大对象随调用堆积；要看负载走 call_tool 的即时返回
        """
        return [
            {
                "tool": r.tool_name,
                "success": r.success,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp,
                "error": r.error,
            }
            for r in self._call_log[-last_n:]
        ]

    def _log(self, result: ToolCallResult) -> None:
        """调用日志：有界截断，超出上限丢最旧（差异 3）"""
        self._call_log.append(result)
        if len(self._call_log) > self._max_log:
            self._call_log = self._call_log[-self._max_log:]
