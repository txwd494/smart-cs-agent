"""
api.main — FastAPI 入口（HTTP 边界）

整个系统的对外服务层。四项职责：
- 生命周期：lifespan 里做一次性初始化（tracer、图单例、知识文档
  种入），请求到达前全部就绪
- 请求契约：Pydantic 模型定义输入输出，自动字段校验
- 会话与状态组装：把 HTTP 请求翻译成图的 initial_state，
  session_id 生成或透传，并绑定为 checkpointer 的 thread_id
- 错误边界：初始化未完成 503，处理失败 500（通用文案，
  细节进服务端日志）

依赖注入链在此闭合：本模块构造 working_memory 与种好文档的
long_term_memory，传入 create_supervisor_graph，拿编译图挂到
模块全局（图工厂的参数设计见 agents/graph.py）。

端点范围（按阶段计划逐步补全）：
- 本模块：POST /api/chat、GET /health、GET /api/history/{session_id}（2.1）、
  GET /api/tools、POST /api/tools/call、POST /api/mcp（2.3，MCP 工具服务器）、
  GET /api/metrics（2.5，Agent 聚合指标 + MCP 工具调用日志）

HTTP 层设计决策（十处）：
1. 指标端点读 get_metrics() 单例：装饰器一路积累、端点读同一
   实例，自建实例没人写入等于端点永远为空
2. 短期记忆双写（写透位置：invoke 前写 user、成功后写 assistant，
   分工分析见 memory/short_term.py；失败窗口行为「有问无答」
   符合事实）
3. 500 通用文案 + 服务端日志：异常文本直接回传客户端会泄露
   内部细节
4. CORS：通配来源不带凭据。通配 origins 加 credentials=True 的
   组合违反 CORS 规范，浏览器会拒绝该响应头；本系统无 Cookie
   场景
5. initial_state 不带无消费字段（状态契约里零读写的字段不传）
6. 未实现的能力不进 docstring：SSE 流式回复留作后续，
   不预写声明
7. 端点按能力分组补全，/api/chat 之外的能力各有独立端点
8. MCP 装配三点：mcp_server 在 lifespan 构造（与图单例同一
   模式，不在模块级 import 即建）；ticket_store 单例同时注入
   图工厂与 MCP server（两处各建各的会形成两套互不相通的
   工单体系）；POST /api/mcp 端点跑 JSON-RPC 协议层（协议层
   要有真实消费路径）
9. /api/metrics 返回 agent_metrics + tool_call_log 两段
10. /api/chat 开 api.chat 业务 span：session_id / user_id /
    intent / compliance_passed 挂 span 属性，trace 可按业务字段
    检索。显式自建而非挂 fastapi 内置 telemetry 的服务端 span：
    不依赖框架 instrumentation 内部行为（静默失效不可见），
    树层级与代码职责边界对齐（fastapi.endpoint → api.chat →
    agent.*）
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agents.graph import create_supervisor_graph
from agents.ticket_handler import TicketStore
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from memory.working_memory import WorkingMemory
from mcp.server import MCPToolServer
from mcp.tools.default_tools import create_default_tools
from tracing.otel_config import business_span, get_metrics, init_tracer

load_dotenv()

logger = logging.getLogger(__name__)

# 编译图单例。lifespan 完成前为 None，/api/chat 据此返回 503；
# 这是「初始化中」的正常处理路径，非错误状态
graph = None

# MCP 工具服务器单例（2.3）。同样 lifespan 构造、503 保护
mcp_server = None

# 短期记忆单例。Redis 不可用时自动走进程内降级，
# 本机开发即降级形态，接真实 Redis 只改 .env 的 REDIS_URL
short_term_memory = ShortTermMemory(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


# ── 请求 / 响应契约 ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    session_id: str | None = None  # None 则由服务端新建


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    compliance_passed: bool


class ToolCallRequest(BaseModel):
    """POST /api/tools/call 的请求体（REST 路径的调用契约）"""

    name: str
    arguments: dict = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """一次性初始化：追踪 → 记忆单例 → 图与 MCP。teardown 释放引用。"""
    global graph, mcp_server

    # OTel 初始化：未配 OTLP 端点时自动退化为控制台导出，
    # 开发期无 Jaeger 也能跑（见 tracing/otel_config.py）
    init_tracer(
        service_name=os.getenv("OTEL_SERVICE_NAME", "smart-cs-agent"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None,
    )

    working_memory = WorkingMemory()

    # 长期记忆（2.2 真实向量检索版）：索引存在则直接加载
    # （持久化生效、重启零嵌入成本），否则从知识库目录加载并落盘
    long_term_memory = LongTermMemory(
        index_path=os.getenv("FAISS_INDEX_PATH"),
    )
    if not long_term_memory._documents:
        count = await long_term_memory.load_knowledge_base("data/knowledge_base")
        if count:
            long_term_memory.save()

    # 共享工单存储：图内工单 Agent 与 MCP 的 ticket_create 工具
    # 共用同一张单（两处各建各的会互不相通）
    ticket_store = TicketStore()

    # 图只构建一次；会话级单例（working_memory / long_term_memory /
    # ticket_store）由这里持有并注入，进程内全图共享
    graph = create_supervisor_graph(
        working_memory=working_memory,
        long_term_memory=long_term_memory,
        ticket_store=ticket_store,
    )

    # MCP 工具服务器：同一批单例注入工具后端，与图一致走 lifespan
    mcp_server = create_default_tools(
        MCPToolServer(),
        long_term_memory=long_term_memory,
        ticket_store=ticket_store,
    )

    yield

    # teardown：进程内的记忆与 checkpointer 随进程消亡，
    # 显式释放引用便于测试里重建
    graph = None
    mcp_server = None


app = FastAPI(
    title="智能客服多Agent系统",
    description="基于LangGraph的Supervisor编排多Agent智能客服系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 通配来源 + 不带凭据：规范合法的组合（本系统无 Cookie 场景；
# 通配来源加 credentials=True 会被浏览器拒绝）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """主聊天接口：请求 → initial_state → 带线程档案的图执行 → 响应。"""
    if graph is None:
        # lifespan 未完成（启动竞态）：服务未就绪，非处理错误
        raise HTTPException(status_code=503, detail="系统初始化中")

    session_id = request.session_id or str(uuid.uuid4())

    # 对话写透（短期记忆）：invoke 前写用户消息。
    # 图执行失败时该条已在档而回复无——「有问无答」符合事实，
    # 与 checkpointer 侧的表现一致
    await short_term_memory.add_message(session_id, "user", request.message)

    # HTTP 请求到图状态的边界翻译。
    # 注意 intent / sub_results 等字段每轮从空值起步：无 reducer 的
    # 字段被输入合并覆盖，跨轮只靠 messages（checkpointer）与
    # working_memory（last_intent）两条通道，见 agents/state.py
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": session_id,
        "intent": "",
        "sub_results": {},
        "compliance_passed": True,
        "final_response": "",
        "current_agent": "",
    }
    config = {"configurable": {"thread_id": session_id}}

    try:
        # api.chat 业务 span：覆盖图执行，业务属性挂 span。
        # 异常传播路径：ainvoke 抛出 → with 退出（span 记录异常）→
        # 这里捕获 → 500。span 能看到原始异常，客户端看不到
        with business_span(
            "api.chat",
            **{"chat.session_id": session_id, "chat.user_id": request.user_id},
        ) as span:
            result = await graph.ainvoke(initial_state, config=config)
            if span is not None:
                span.set_attribute("chat.intent", result.get("intent", "unknown"))
                span.set_attribute(
                    "chat.compliance_passed", result.get("compliance_passed", True)
                )
    except Exception:
        # 不把异常文本回传客户端，服务端留全量日志
        logger.exception("graph 执行失败 session_id=%s", session_id)
        raise HTTPException(status_code=500, detail="处理失败，请稍后重试")

    final_response = result.get("final_response", "系统处理异常，请稍后重试")

    # 写透另一半：图成功后才写回复
    await short_term_memory.add_message(session_id, "assistant", final_response)

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        intent=result.get("intent", "unknown"),
        compliance_passed=result.get("compliance_passed", True),
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """查询指定会话的对话历史（短期记忆的查询出口）"""
    history = await short_term_memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.get("/api/tools")
async def list_tools():
    """MCP 工具发现（REST 路径，直调 list_tools）"""
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="系统初始化中")
    return {"tools": mcp_server.list_tools()}


@app.post("/api/tools/call")
async def call_tool(request: ToolCallRequest):
    """
    MCP 工具调用（REST 路径，直调 call_tool）。

    失败语义：工具层的失败（未知工具、缺参数、执行异常）落在
    200 响应的 success=false 里；协议层错误不在此路径出现
    """
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="系统初始化中")
    result = await mcp_server.call_tool(request.name, request.arguments)
    return {
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


@app.post("/api/mcp")
async def mcp_endpoint(request: dict):
    """
    MCP 的 JSON-RPC 2.0 协议入口。

    消息体形如 {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
    "params": {...}}，支持 tools/list、tools/call、ping。协议层
    失败返回 error 信封（-32601 方法不存在 / -32603 内部错误），
    工具层失败在 result 里带 success=false
    """
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="系统初始化中")
    return await mcp_server.handle_jsonrpc(request)


@app.get("/api/metrics")
async def metrics_endpoint():
    """
    系统指标（2.5）：Agent 聚合指标 + MCP 工具调用日志。

    agent_metrics 段是 trace_agent_call 装饰器一路积累的进程内
    聚合（重启清零，单 worker 视角，见 AgentMetrics 的说明）；
    tool_call_log 段取最近 20 条工具调用
    """
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="系统初始化中")
    return {
        "agent_metrics": get_metrics().get_summary(),
        "tool_call_log": mcp_server.get_call_log(last_n=20),
    }


@app.get("/health")
async def health_check():
    """存活检查（部署探针用）"""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,  # 开发用；生产以 uvicorn 命令行启动并去掉 reload
    )
