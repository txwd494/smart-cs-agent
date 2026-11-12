"""
tracing.otel_config — 全链路追踪与指标收集

本模块是整个系统的「可观测性」基础设施，所有 Agent 的方法都会通过
trace_agent_call 装饰器埋点，从而拿到每次 Agent 调用的 Span（一次调用的时间线记录）。

模块分四部分：
1. Tracer 初始化（init_tracer / get_tracer）：OpenTelemetry 的全局入口
2. trace_agent_call 装饰器：给任意 async 方法创建 Span 并记录耗时、成功与否
3. business_span：动态名 Span 的上下文管理器（2.5）。装饰器适合静态
   方法名，两种场景用它：名称含运行期参数（MCP 的 mcp.tool.{工具名}）、
   请求边界（api.main 的 api.chat，携带 session / intent 等业务属性）。
   显式自建而非挂 fastapi 内置 telemetry 的服务端 Span：不依赖框架
   instrumentation 的内部行为，失效模式可见
4. AgentMetrics：进程内的指标聚合（调用次数 / 平均耗时 / 错误率）

接线设计（两处）：
- 指标收集必须在装饰器里调 record_call：不接线的话 /api/metrics
  永远返回空字典，指标类沦为摆设
- 计时用 time.perf_counter()（单调时钟）：time.time() 是墙钟，
  系统校时会跳变，不适合测耗时
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# ── 可选依赖模式 ──────────────────────────────────────────────────────────
# OpenTelemetry 用 try/except 导入：装了就有追踪，没装系统照样跑。
# 这样 tracing 是「增强功能」而非「硬依赖」，部署门槛更低。
# _HAS_OTEL 是模块级标志，后面所有函数都先查它再决定走正常路径还是降级路径。
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


# 全局单例。下划线前缀表示「模块内部状态，外部请用函数访问」。
_tracer = None   # OpenTelemetry Tracer，负责创建 Span
_metrics = None # AgentMetrics 指标收集器（本版新增，见模块 docstring）


def init_tracer(
    service_name: str = "smart-cs-agent",
    otlp_endpoint: str | None = None,
) -> None:
    """
    初始化 OpenTelemetry 追踪器。应用启动时调用一次（在 api/main.py 的 lifespan 里）。

    Args:
        service_name: 服务名，会作为 Resource 属性附加到每个 Span 上，
                      在 Jaeger UI 里用于区分不同服务。
        otlp_endpoint: OTLP 收集器地址（如 Jaeger 的 http://localhost:4317）。
                       传 None 则退化为 ConsoleSpanExporter（把 Span 打到控制台），
                       适合本地开发、没部署 Jaeger 的场景。
    """
    global _tracer

    # OTel 未安装：直接返回，系统其余部分不受影响
    if not _HAS_OTEL:
        return

    # Resource 是 Span 的元数据载体，service.name 是最关键字段
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # 选导出器（exporter 决定 Span 数据发到哪里）：
    # - 有 endpoint：通过 gRPC 发给 OTLP 收集器（Jaeger/Tempo 等）
    # - 没有：打到控制台
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        except ImportError:
            # opentelemetry-exporter-otlp 包缺失时降级到控制台输出
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    # BatchSpanProcessor：把 Span 攒成批量再导出（异步、低开销）。
    # 代价是延迟：默认约 5 秒一批，开发时控制台「慢半拍」是正常现象。
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # 注册为进程级全局 provider，之后 trace.get_tracer() 拿到的都是这套配置
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer():
    """
    获取全局 Tracer（懒加载）。

    为什么需要懒加载：Agent 模块在 import 时就会执行 @trace_agent_call 装饰，
    那时 init_tracer 可能还没被调用。懒加载保证「先 import 后初始化」的顺序
    不会报错：第一次真正调用被装饰的方法时才要求 tracer 存在。
    """
    global _tracer
    if _tracer is None:
        if _HAS_OTEL:
            # 用 OTel 的全局 provider（没人调 init_tracer 时是默认空实现，
            # Span 依然能创建，只是不被导出，零成本）
            _tracer = trace.get_tracer("smart-cs-agent")
        else:
            # OTel 未安装，显式返回 None，调用方据此走降级路径
            return None
    return _tracer


def get_metrics() -> "AgentMetrics":
    """
    获取全局 AgentMetrics 实例。

    api/main.py 通过它拿同一个实例，/api/metrics 才能读到
    装饰器一路积累下来的指标（装饰器与端点共享单例，这就是接线点）。
    """
    global _metrics
    if _metrics is None:
        _metrics = AgentMetrics()
    return _metrics


@contextmanager
def business_span(name: str, **attributes: Any) -> Iterator[Any]:
    """
    动态名业务 Span（2.5）：用法与装饰器互补的场景见模块 docstring。

    属性键含点号时用 **{"chat.session_id": ...} 传。OTel 未安装时
    yield None，调用方判 None 决定是否补后续属性（与装饰器的
    tracer None 透传同一降级模式）。
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def trace_agent_call(agent_name: str) -> Callable:
    """
    Agent 调用追踪装饰器（装饰器工厂）。

    用法：
        @trace_agent_call("knowledge_rag")
        async def process(self, state):
            ...

    注意三层嵌套关系：
        trace_agent_call("xxx")  → 返回 decorator
        decorator(func)          → 返回 wrapper
        wrapper(*args, **kwargs) → 实际执行时的入口（async）

    为什么是「工厂」：装饰器本身需要接收 agent_name 参数，
    所以必须再包一层把参数固定住。Python 里带参数的装饰器都是这个形状。

    每个 agent_name 可以装饰多个方法（如 supervisor 的 route_decision
    和 synthesize_response），Span 用「agent 名 + 方法名」区分。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)  # 保留原函数的名字/docstring，没有它调试信息全是 wrapper
        async def wrapper(*args, **kwargs) -> Any:
            # 每次调用时才取 tracer（而非装饰时），配合懒加载
            tracer = get_tracer()

            # 降级路径：OTel 不可用时透传，功能不缺失，只是没追踪
            if tracer is None:
                return await func(*args, **kwargs)

            # Span 命名约定：agent.{agent名}.{方法名}
            # 例如 agent.supervisor.route_decision
            # 在 Jaeger UI 里按这个前缀就能过滤出所有 Agent 调用
            span_name = f"agent.{agent_name}.{func.__name__}"

            # start_as_current_span 是上下文管理器：
            # 进入时开启 Span 并设为「当前」，退出时自动结束。
            # 嵌套调用的方法各自开 Span 时，OTel 自动建立父子关系，
            # 这就是「全链路」追踪树的来源。
            with tracer.start_as_current_span(span_name) as span:
                # Span attribute 只接受基本类型（str/int/float/bool）
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.method", func.__name__)

                # perf_counter：单调递增时钟，只用于测时间间隔
                start = time.perf_counter()

                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", True)
                    # 记录返回值的字段名（不记内容，控制体量），
                    # 排查时能看出这个节点产出了什么
                    if isinstance(result, dict):
                        span.set_attribute("agent.result_keys", str(list(result.keys())))

                    # 接上指标收集（见模块 docstring 接线设计）
                    get_metrics().record_call(agent_name, duration_ms, success=True)

                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000

                    span.set_attribute("agent.duration_ms", duration_ms)
                    span.set_attribute("agent.success", False)
                    span.set_attribute("agent.error", str(e))
                    # record_exception 把异常堆栈完整记进 Span，
                    # Jaeger UI 里可直接展开看调用栈
                    span.record_exception(e)

                    # 失败也计入指标（错误率统计需要）
                    get_metrics().record_call(agent_name, duration_ms, success=False)

                    # 关键：重新抛出。装饰器只做观测，不吞异常，
                    # 否则上层（LangGraph）感知不到失败，错误处理就失效了
                    raise

        return wrapper

    return decorator


class AgentMetrics:
    """
    Agent 调用指标收集器（进程内）。

    与 OTel 的分工：
    - OTel Span：单次调用的完整时间线，用于排查问题（ Jaeger 里的瀑布图）
    - AgentMetrics：跨调用的聚合统计，用于监控大盘（/api/metrics）

    三个 dict 以 agent_name 为键平行维护：
    _call_counts    总调用次数
    _total_duration 累计耗时（ms）
    _error_counts   失败次数

    进程重启即清零。真实生产会用 Prometheus 这类时序库，
    这里从简，够 /api/metrics 展示。
    """

    def __init__(self):
        self._call_counts: dict[str, int] = {}
        self._total_duration: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}

    def record_call(self, agent_name: str, duration_ms: float, success: bool) -> None:
        """记录一次 Agent 调用。由 trace_agent_call 在成功/失败两个分支各调用一次。"""
        self._call_counts[agent_name] = self._call_counts.get(agent_name, 0) + 1
        self._total_duration[agent_name] = self._total_duration.get(agent_name, 0.0) + duration_ms
        if not success:
            self._error_counts[agent_name] = self._error_counts.get(agent_name, 0) + 1

    def get_summary(self) -> dict[str, Any]:
        """汇总为 {agent_name: {total_calls, avg_duration_ms, error_rate}} 结构。"""
        summary = {}
        for agent_name in self._call_counts:
            calls = self._call_counts[agent_name]
            total_ms = self._total_duration[agent_name]
            errors = self._error_counts.get(agent_name, 0)
            summary[agent_name] = {
                "total_calls": calls,
                "avg_duration_ms": total_ms / calls if calls > 0 else 0,
                "error_rate": errors / calls if calls > 0 else 0,
            }
        return summary
