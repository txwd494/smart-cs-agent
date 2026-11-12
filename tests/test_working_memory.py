"""test_working_memory — 1.3 冒烟断言的固化（KV 合并 / 截断 / 拷贝）"""

from memory.working_memory import WorkingMemory


def test_update_and_merge():
    # update 收 data dict：同 session 多次 update 合并进快照
    wm = WorkingMemory()
    wm.update("s1", {"last_intent": "knowledge_rag"})
    wm.update("s1", {"another_key": 1})
    assert wm.get_context("s1") == {"last_intent": "knowledge_rag", "another_key": 1}


def test_each_update_appends_log_entry():
    # 事件日志：每次 update 追加一条 entry（1.3 双结构）
    wm = WorkingMemory()
    wm.update("s1", {"a": 1})
    wm.update("s1", {"b": 2})
    assert len(wm.get_history("s1")) == 2
    assert wm.get_context("s1") == {"a": 1, "b": 2}


def test_missing_session_returns_empty():
    wm = WorkingMemory()
    assert wm.get_context("nope") == {}
    assert wm.get_history("nope") == []


def test_read_does_not_create_key():
    wm = WorkingMemory()
    wm.get_context("never")
    assert wm.get_history("never") == []


def test_truncation_keeps_latest():
    wm = WorkingMemory(max_entries_per_session=3)
    for i in range(5):
        wm.update("s1", {"seq": i})
    assert len(wm.get_history("s1")) == 3
    assert wm.get_context("s1")["seq"] == 4  # 快照保最新值


def test_max_entries_zero_keeps_all():
    # 边界锚点：[-0:] 等于全保留，截断失效
    wm = WorkingMemory(max_entries_per_session=0)
    for i in range(3):
        wm.update("s1", {"seq": i})
    assert len(wm.get_history("s1")) == 3


def test_get_history_last_n():
    wm = WorkingMemory()
    for i in range(5):
        wm.update("s1", {"seq": i})
    entries = wm.get_history("s1", last_n=2)
    assert len(entries) == 2
    assert entries[-1]["data"]["seq"] == 4


def test_get_history_double_copy():
    # entry 层 + data 层双层拷贝（1.3 冒烟抓出的缺陷）。
    # 边界：data 值层（如嵌套 list）仍是共享引用，源码注释已记录，
    # 本断言停在 dict 层
    wm = WorkingMemory()
    wm.update("s1", {"data": {"nested": [1]}})
    entry = wm.get_history("s1")[0]
    entry["data"]["data"] = {"injected": True}  # data 层键改写
    entry["injected"] = True                     # entry 层
    assert wm.get_context("s1") == {"data": {"nested": [1]}}
    assert "injected" not in wm.get_history("s1")[0]


def test_clear():
    wm = WorkingMemory()
    wm.update("s1", {"a": 1})
    wm.update("s2", {"b": 2})
    wm.clear("s1")
    assert wm.get_context("s1") == {}
    assert wm.get_context("s2") == {"b": 2}  # 其他 session 不受影响


def test_export():
    wm = WorkingMemory()
    wm.update("s1", {"a": 1})
    export = wm.export_for_persistence("s1")
    assert isinstance(export, dict) and export  # 非空导出
