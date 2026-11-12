"""
agents.state — Supervisor 编排的全局状态契约

AgentState 是 StateGraph 的状态 schema：图中每个节点收到的 state、
每次返回的增量更新，都以这些字段为边界。所有 Agent 模块共同依赖，
所以独立成文件。

字段与合并语义，为什么只有 messages 带 reducer：
- messages 配 add_messages：按消息 ID 合并、累积。对话历史因此跨轮存活
  （配合 checkpointer 按 thread 存档），这是跨轮信息通道之一
- 其余字段无注解，是 LastValue 通道：后写覆盖。端点每轮传入的
  initial_state 会直接覆盖 checkpoint 里的旧值，所以 intent /
  sub_results 等字段只反映当轮。跨轮存活只有两条路：messages（经
  checkpointer）与 working_memory（图外自管，见 memory/working_memory.py
  模块 docstring 里与 checkpointer 的分工说明）

设计说明：
- 独立成 agents/state.py：所有 Agent 模块共同依赖的状态契约，
  单独成文件避免循环依赖
- 不设 retry_count 之类的无消费字段：字段进了状态契约就要有
  读写方，否则是纯噪声
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Supervisor 编排的全局状态（合并语义见模块 docstring）"""

    # 唯一带 reducer 的字段：按 ID 累积。端点写入用户消息，
    # synthesize 写入最终回复，是这份数据仅有的两个写入点
    messages: Annotated[list[BaseMessage], add_messages]

    # 以下均为 LastValue 通道（后写覆盖）
    user_id: str            # 端点写入；工单 Agent 建单时使用
    session_id: str         # 端点写入；一键三用：working memory 的键、
                            # checkpointer 的 thread_id、API 返回值
    intent: str             # route_decision 写入；当轮路由结果，
                            # 条件边 route_to_agent 的路由依据
    sub_results: dict[str, Any]   # 各子 Agent 写入（节点内手工 merge，
                                   # 本图无并行分支所以安全）
    compliance_passed: bool       # compliance 节点写入；synthesize 的分支依据
    final_response: str     # synthesize 写入；最终回复本体
    current_agent: str      # route_decision 写入 "supervisor"；
                            # 子 Agent 不更新它，值停留在 supervisor
