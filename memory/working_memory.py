"""
memory.working_memory — 工作记忆（分层记忆第一层）

系统按生命周期把记忆分三层，本模块是最里面的一层：
- working_memory（本模块）：Agent 推理的中间状态，进程内存，按 session 隔离
- short_term_memory（2.1）：跨轮对话历史，Redis（开发期走进程内降级）
- long_term_memory（2.2）：知识文档，FAISS

核心职责：按 session 隔离的跨轮 KV 存储 + 带时间戳的事件日志。
Supervisor 每轮路由前读 get_context 注入 prompt、路由后 update 写回
last_intent，下一轮由此拿到上一轮的路由结果。这条信息没有别的来源：
路由调用是独立的 LLM 调用，其输出不进 AgentState.messages；checkpointer
也不携带（无 reducer 的普通字段每次请求被端点的输入合并覆盖）。

与 checkpointer 的分工（图外为什么还要一个自管存储）：
- checkpointer 按 thread 存 superstep 级全量快照，字段要跨轮存活需逐个配
  reducer，且 state 会广播给全部节点
- 本模块任意键零合约写入即跨轮存活；引用只发给选定组件；节点执行中途的
  多次写入也各自留痕，get_history 的事件级日志是图机制覆盖不到的形态
- 生命周期独立：enable_checkpointing=False 或换持久后端不影响本模块；
  反向同样成立，本模块随进程消失，不提供持久性

拷贝与生命周期设计（收紧三处，保留两处）：
收紧：
1. update 入日志前对 data 做一层拷贝。直接持有调用方 dict 的引用，
   调用方事后修改会篡改已存历史
2. get_history 做 entry 层 + data 层拷贝。返回内部 dict 的引用，
   调用方修改返回值会污染内部状态
3. docstring 按实际行为撰写：生命周期是跨请求存活（否则
   last_intent 第二轮就读不到）；「子 Agent 中间结果」职责当前
   无调用方，如实标注未接线
保留：
- clear / export_for_persistence 当前零调用（前者导致 session 键
  空间只增不减），保留接口，接线列入后续优化
- 写路径加锁、读路径不加（依据见类 docstring 的并发分析）
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Any


class WorkingMemory:
    """
    工作记忆：按 session 隔离的跨轮 KV 存储 + 事件日志。

    双结构（一次 update 同时写两处，读方按需选择读法）：
    - _store   append-only 事件日志，元素 {timestamp, data}，容量截断
               保最新。回答「这个 session 依次发生过什么」
    - _context 合并快照，同键后写覆盖先写。回答「当前状态是什么」

    并发设计（写加锁、读不加，三点依据）：
    1. CPython GIL 下 dict.get 与 list 切片是原子操作，无锁读拿到的是自洽
       视图，至多落后一次更新，本用途可接受
    2. 容量截断用切片重绑定而非原地删除，并发读手里的旧 list 引用不会
       被破坏
    3. 当前接线唯一写入口是 async 的 Supervisor.route_decision，运行在单
       事件循环上，锁实际不会竞争，属于对其他调用场景（同步代码、多线程）
       的保守防护
    对照：tracing.AgentMetrics 在同样的「单事件循环唯一写入口」条件下选择
    不加锁。两种取舍在当前接线里行为无差别。
    """

    def __init__(self, max_entries_per_session: int = 50):
        # defaultdict：写侧免判空（下标访问自动建键）；读侧一律用 .get，
        # 不存在的 session 不建空条目，读不污染键空间
        self._store: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._context: dict[str, dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()
        # 上限只约束 _store 的事件条数；_context 的键集合不随之收缩，
        # 键一旦写入存活到 clear（合并语义里没有「删键」这个事件）
        self._max_entries = max_entries_per_session

    def update(self, session_id: str, data: dict[str, Any]) -> None:
        """
        写入一次状态更新：追加进事件日志 + 合并进当前快照。

        data 的多个键共享同一条 entry、同一时间戳；要成为独立事件就分开调用。
        """
        with self._lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                # 一层拷贝：调用方事后修改传入的 dict
                # 不会篡改已存历史。值若为可变对象仍是共享引用，
                # 拷贝深度与 get_context 保持同一约定
                "data": dict(data),
            }
            self._store[session_id].append(entry)

            # >0 守卫：max_entries=0 时切片 [-0:] 等于切全部，
            # 截断反而失效（-0 == 0），这里堵上该边界
            if self._max_entries > 0 and len(self._store[session_id]) > self._max_entries:
                # 切片重绑定而非原地删（并发依据见类 docstring 第 2 点）。
                # 每次 O(max)，50 的量级无所谓；deque(maxlen) 追加是 O(1)
                # 但不支持切片，get_history 取最近 N 条需要切片，所以用 list
                self._store[session_id] = self._store[session_id][-self._max_entries:]

            self._context[session_id].update(data)

    def get_context(self, session_id: str) -> dict[str, Any]:
        """读取该 session 的合并快照（同键以最近一次写入为准）。"""
        # dict(...) 浅拷贝：调用方改返回值不影响内部状态；值若为可变对象
        # 仍是共享引用（本系统实际存标量，如 intent 字符串，不受影响）
        return dict(self._context.get(session_id, {}))

    def get_history(self, session_id: str, last_n: int = 10) -> list[dict[str, Any]]:
        """
        读取最近 N 条事件（带时间戳的更新日志，粒度 = 每次 update 调用）。

        当前接线里每轮图执行恰好产生一条（Supervisor 的路由决策）；
        子 Agent 接入中间结果写入后，这里才成为完整的执行时间线。
        """
        entries = self._store.get(session_id, [])
        # entry 层 + data 层两份拷贝：调用方改返回值不污染内部 _store，
        # 包括 data 层；data 里的值若为可变对象仍是共享引用
        #（本模块统一约定：模块自管的 dict 结构拷贝，值共享）
        return [
            {"timestamp": entry["timestamp"], "data": dict(entry["data"])}
            for entry in entries[-last_n:]
        ]

    def clear(self, session_id: str) -> None:
        """
        释放该 session 的全部工作记忆（日志与快照一起删）。

        本模块唯一释放内存的操作：容量上限只管单 session 的日志条数，
        session 的个数只有这里管，不调用就只增不减。
        注意与 checkpointer 互不知晓：清掉本模块后，图那边同一 thread 的
        状态仍在，完整结束一个会话需要两边各自处理。
        """
        with self._lock:
            self._store.pop(session_id, None)
            self._context.pop(session_id, None)

    def export_for_persistence(self, session_id: str) -> dict[str, Any]:
        """
        打包导出，作为交给短期记忆的对接接口（当前无调用方，
        保留接口待接）。
        """
        # 两次读之间不持锁，与并发 update 交错时 context 与 history 可能
        # 相差一个事件，导出场景可接受
        return {
            "session_id": session_id,
            "context": self.get_context(session_id),
            "history": self.get_history(session_id),
            "exported_at": datetime.now().isoformat(),
        }
