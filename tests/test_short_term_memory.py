"""test_short_term_memory — 降级路径的固化（本机无 Redis，
构造时连不上自动降级；Redis 正常路径未覆盖）

Redis 侧用 monkeypatch 强制 _get_redis 返回 None 走降级分支，
避免测试依赖本机 Redis 恰好在线造成的路径漂移。
"""

import pytest

from memory.short_term import ShortTermMemory


@pytest.fixture
def stm(monkeypatch):
    mem = ShortTermMemory()
    # 无论本机 Redis 状态如何，强制走进程内降级
    async def none_redis():
        return None
    monkeypatch.setattr(mem, "_get_redis", none_redis)
    return mem


async def test_add_and_read(stm):
    await stm.add_message("s1", "user", "你好")
    await stm.add_message("s1", "assistant", "您好")
    msgs = await stm.get_history("s1")
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "你好"), ("assistant", "您好"),
    ]
    assert all("timestamp" in m for m in msgs)


async def test_session_isolation(stm):
    await stm.add_message("a", "user", "1")
    await stm.add_message("b", "user", "2")
    assert len(await stm.get_history("a")) == 1
    assert (await stm.get_history("b"))[0]["content"] == "2"


async def test_sliding_window(stm):
    # 窗口单位是消息条数（Redis ltrim(-max_turns,-1) 同款语义）
    stm.max_turns = 4
    for i in range(5):
        await stm.add_message("s1", "user", f"u{i}")
        await stm.add_message("s1", "assistant", f"a{i}")
    msgs = await stm.get_history("s1")
    assert len(msgs) == 4  # 保最新
    assert msgs[0]["content"] == "u3"


async def test_last_n_zero_returns_empty(stm):
    # 边界锚点：falsy 判断会把 0 吞成窗口值
    await stm.add_message("s1", "user", "x")
    assert await stm.get_history("s1", last_n=0) == []


async def test_last_n_partial(stm):
    for i in range(4):
        await stm.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
    msgs = await stm.get_history("s1", last_n=2)
    assert [m["content"] for m in msgs] == ["m2", "m3"]


async def test_expired_session_returns_empty(stm):
    # 降级侧过期检查对齐 Redis「整键 TTL、以最新消息时间戳为准」
    #（2.1 差异 2）：过期是 session 级判定；降级库直接以 session_id 为键
    from datetime import datetime

    await stm.add_message("s1", "user", "消息")
    old = datetime.fromisoformat(
        stm._fallback_store["s1"][-1]["timestamp"]
    ).replace(year=2000).isoformat()
    stm._fallback_store["s1"][-1]["timestamp"] = old
    assert await stm.get_history("s1") == []


async def test_returned_copy_isolation(stm):
    await stm.add_message("s1", "user", "原")
    msgs = await stm.get_history("s1")
    msgs[0]["content"] = "改"
    assert (await stm.get_history("s1"))[0]["content"] == "原"


async def test_clear(stm):
    await stm.add_message("s1", "user", "x")
    await stm.clear("s1")
    assert await stm.get_history("s1") == []
