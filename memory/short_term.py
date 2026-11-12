"""
memory.short_term — 短期记忆（对话的服务端存档，三层记忆第二层）

三层记忆至此补齐：working_memory 是 Agent 推理状态（图外自管）、
long_term_memory 是知识文档（检索后端）、本模块是对话本身的
服务端存档，回答「这个 session 都聊了什么」，消费方是
/api/history 与前端。

与 checkpointer 的分工（同一份对话，两个消费者）：
- checkpointer：图执行的内部机制，进程内、无查询接口，
  服务模型的上下文（messages 通道）
- 本模块：服务层资产，有查询接口、Redis 支撑跨进程跨重启、
  TTL + 滑动窗口。双写是这两个消费者分离的代价（已知边界：
  两份存档无同步保证；失败窗口行为是「有问无答」，符合事实）

双形态：Redis 是生产形态，进程内降级是开发期主路径
（本机无 Docker），接口不变只换实现。

Redis 数据结构：每个 session 一个 list，键名 smartcs:short_term:{id}
做命名空间。写入三操作：rpush 追加 + ltrim 保窗口 + expire 刷 TTL。

降级与边界语义设计（四处）：
1. 运行中断线降级。try/except 只包连接尝试是不够的：rpush/lrange
   等操作无保护时，Redis 运行中挂掉会让 add_message 抛
   ConnectionError 直接打到 /api/chat。每个 Redis 操作单独
   try/except，失败时该条走降级并置 None 触发下次重连
2. 降级路径带过期检查：最新消息早于 TTL 即整个 session 过期。
   降级侧永不过期会造成同一类存储两种行为；对齐 Redis
   「整键 TTL，写入刷新」的可见行为
3. get_history 逐条拷贝消息 dict。降级路径返回内部 dict 的引用，
   调用方改返回值会污染存储（与 working_memory 同病同防）
4. last_n=0 语义：`last_n or self.max_turns` 的 falsy 判断把 0
   吞成窗口值，显式判 None 才对

不设 get_context_window 之类的 token 估算接口：len//2 对中文
严重低估，且无消费方。

已知边界：降级期历史存进程内，Redis 中途恢复后两边不合并
（断线重连数据分裂）；降级状态下每次操作重试连接无退避（本机
连接拒绝毫秒级，远端不可达场景会拖慢请求，生产路线按 Redis
硬依赖建设，不再依赖降级）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None


class ShortTermMemory:
    """短期记忆：Redis 或进程内降级双形态（架构与差异见模块 docstring）"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_turns: int = 20,
        ttl_seconds: int = 1800,
    ):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._redis_url = redis_url
        self._redis: Any = None
        self._fallback_store: dict[str, list[dict[str, Any]]] = {}

    async def _get_redis(self):
        """
        懒加载 Redis 连接，不可用返回 None（调用方走降级）。

        只在首次与重连时尝试：from_url + ping 验证连通，
        任何异常置 None。降级状态下每次操作都会走到这里重试
        （无退避，见模块 docstring 的已知边界）
        """
        if self._redis is None:
            if aioredis is None:
                return None
            try:
                client = aioredis.from_url(self._redis_url, decode_responses=True)
                await client.ping()
                self._redis = client
            except Exception:
                self._redis = None
        return self._redis

    def _session_key(self, session_id: str) -> str:
        return f"smartcs:short_term:{session_id}"

    # ── 降级路径的三件小事 ──────────────────────────────────────────────

    def _fallback_trim(self, session_id: str) -> None:
        """降级路径滑动窗口（与 Redis ltrim(-max_turns, -1) 等价）"""
        if session_id in self._fallback_store:
            self._fallback_store[session_id] = (
                self._fallback_store[session_id][-self.max_turns:]
            )

    def _fallback_is_expired(self, session_id: str) -> bool:
        """
        降级路径过期检查（差异 2）。

        对齐 Redis「整键 TTL、每次写入刷新」的可见行为：以最新
        消息的时间戳为准，早于 TTL 即整个 session 过期。
        """
        messages = self._fallback_store.get(session_id)
        if not messages:
            return False
        last_ts = datetime.fromisoformat(messages[-1]["timestamp"])
        return datetime.now() - last_ts > timedelta(seconds=self.ttl_seconds)

    def _fallback_read(self, session_id: str, last_n: int | None) -> list[dict]:
        """降级路径读取：过期视为空，逐条拷贝（差异 3）"""
        if self._fallback_is_expired(session_id):
            return []
        history = self._fallback_store.get(session_id, [])
        if last_n is not None:
            history = history[-last_n:] if last_n > 0 else []
        # 拷贝隔离：dict 结构层拷贝，值（str）不可变，此层已足够
        return [dict(m) for m in history]

    # ── 对外接口（Redis / 降级双路径，签名与语义两边一致）───────────────

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话消息（role: user / assistant）"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        r = await self._get_redis()
        if r is None:
            self._fallback_store.setdefault(session_id, []).append(message)
            self._fallback_trim(session_id)
            return

        key = self._session_key(session_id)
        try:
            # 三原子操作：追加、保窗口、刷 TTL
            await r.rpush(key, json.dumps(message, ensure_ascii=False))
            await r.ltrim(key, -self.max_turns, -1)
            await r.expire(key, self.ttl_seconds)
        except Exception:
            # 差异 1：运行中断线不抛到调用方。本条降级落盘，
            # 置 None 让下次操作重试连接
            self._redis = None
            self._fallback_store.setdefault(session_id, []).append(message)
            self._fallback_trim(session_id)

    async def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        """获取对话历史（默认窗口内全部；last_n=0 返回空，见差异 4）"""
        r = await self._get_redis()
        if r is None:
            return self._fallback_read(session_id, last_n)

        key = self._session_key(session_id)
        try:
            n = self.max_turns if last_n is None else last_n
            if n <= 0:
                return []
            raw = await r.lrange(key, -n, -1)
            return [json.loads(item) for item in raw]
        except Exception:
            self._redis = None
            return self._fallback_read(session_id, last_n)

    async def clear(self, session_id: str) -> None:
        """清除指定 session 的短期记忆"""
        r = await self._get_redis()
        if r is None:
            self._fallback_store.pop(session_id, None)
            return

        try:
            await r.delete(self._session_key(session_id))
        except Exception:
            self._redis = None
            self._fallback_store.pop(session_id, None)
