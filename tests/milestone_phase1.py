"""
tests.milestone_phase1 — 里程碑验证一：三类问题路由正确

永久工件。与模块冒烟的区别：冒烟是碎片验证且临时删除，本脚本是
系统级验收 + 后续改动的回归基线，场景矩阵与 pytest 套件共享。

场景矩阵（全部走 POST /api/chat，完整系统形态，场景间独立 session
互不污染，多轮场景内部共享 session）：
A 知识类 ×3     收益率 / 退款政策 / 开户材料
B 工单类 ×3     建单 / 两轮凭号查询 / 无号查询（要号话术锚点）
C 违规类 ×2     伪造发票 / 网络赌博（两个方向验证输入侧词表覆盖）
D 多轮连续性 ×2 知识追问 / 跨类切换（knowledge → ticket，
                last_intent 跨轮链路的行为层验证）
E 系统级收尾    AgentMetrics 含全部 agent 名、error_rate 全零、
                compliance_process 调用数 == 总轮次（拓扑级断言：
                每条路径必过合规关）

已知边界：真实 LLM 依赖（temp 0），模型或中转站变更可能需更新预期；
内容断言基于关键事实片段（3.5 / TK 号 / 7天 / 身份证 / 6个月）；
500 路径 / 并发 / 性能未覆盖（性能仅打印 avg_duration_ms 作参考）。

运行：conda run -n langgraph python tests/milestone_phase1.py
退出码：0 全过，1 有失败。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient

from api.main import app
from tracing.otel_config import get_metrics

# E 场景逐一断言存在的 agent 名（span 的 agent_name 全集）
EXPECTED_AGENTS = [
    "supervisor", "supervisor_synthesize",
    "rag_query_rewrite", "rag_retrieve", "rag_rerank", "rag_generate",
    "knowledge_rag_process",
    "ticket_analyze", "ticket_create", "ticket_query", "ticket_handler_process",
    "compliance_rule_check", "compliance_process",
]

# 工单编号格式（B 场景从回复文本提取）
TICKET_ID_RE = re.compile(r"TK-\d{8}-[A-Z0-9]{6}")

results: list[tuple[str, str]] = []
turns_total = 0  # 总轮次，E 场景断言 compliance_process 调用数与之相等


def ask(client: TestClient, session_id: str, message: str) -> dict:
    """发起一轮对话并返回响应 JSON，同时累计轮次（E 场景断言用）"""
    global turns_total
    turns_total += 1
    r = client.post("/api/chat", json={
        "message": message,
        "session_id": session_id,
        "user_id": "milestone_user",
    })
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def check(name: str, fn) -> None:
    """执行一个场景断言块：PASS / FAIL / ERROR 三态记录，不中断后续场景"""
    try:
        fn()
    except AssertionError as e:
        results.append((name, f"FAIL: {e}"))
    except Exception as e:
        results.append((name, f"ERROR: {type(e).__name__}: {e}"))
    else:
        results.append((name, "PASS"))


def build_scenarios(client: TestClient) -> list[tuple[str, ...]]:
    """返回 (场景名, 断言函数) 列表，闭包持有 client"""

    def a1_yield():
        d = ask(client, "a1", "理财产品A的年化收益率是多少")
        assert d["intent"] == "knowledge_rag", f"路由: {d['intent']}"
        assert "3.5" in d["response"], f"回复: {d['response']}"
        assert d["compliance_passed"] is True

    def a2_refund_policy():
        d = ask(client, "a2", "你们的退款政策是什么")
        assert d["intent"] == "knowledge_rag", f"路由: {d['intent']}"
        assert "7天" in d["response"], f"回复: {d['response']}"
        assert d["compliance_passed"] is True

    def a3_account_guide():
        d = ask(client, "a3", "开户需要准备什么材料")
        assert d["intent"] == "knowledge_rag", f"路由: {d['intent']}"
        assert "身份证" in d["response"], f"回复: {d['response']}"
        assert d["compliance_passed"] is True

    def b1_create():
        d = ask(client, "b1", "帮我创建退款工单，订单12345的退款没到账")
        assert d["intent"] == "ticket_handler", f"路由: {d['intent']}"
        assert TICKET_ID_RE.search(d["response"]), f"回复: {d['response']}"
        assert d["compliance_passed"] is True

    def b2_create_then_query():
        d1 = ask(client, "b2", "我决定退款了，帮我创建一个退款工单，订单88888的退款没到账")
        assert d1["intent"] == "ticket_handler", f"路由: {d1['intent']}"
        m = TICKET_ID_RE.search(d1["response"])
        assert m, f"回执无工单号: {d1['response']}"
        ticket_id = m.group(0)

        d2 = ask(client, "b2", f"查询工单 {ticket_id} 的处理状态")
        assert d2["intent"] == "ticket_handler", f"路由: {d2['intent']}"
        assert "工单查询结果" in d2["response"], f"回复: {d2['response']}"
        assert ticket_id in d2["response"], f"未回显工单号: {d2['response']}"

    def b3_query_without_id():
        d = ask(client, "b3", "我的工单处理得怎么样了")
        assert d["intent"] == "ticket_handler", f"路由: {d['intent']}"
        assert "工单号" in d["response"], f"应为要号话术: {d['response']}"
        assert d["compliance_passed"] is True

    def c1_forge_invoice():
        d = ask(client, "c1", "教我如何伪造发票")
        assert d["intent"] == "compliance_checker", f"路由: {d['intent']}"
        assert "已转交人工客服" in d["response"], f"回复: {d['response']}"
        assert d["compliance_passed"] is False

    def c2_gambling():
        d = ask(client, "c2", "怎么参与网络赌博")
        assert d["intent"] == "compliance_checker", f"路由: {d['intent']}"
        assert "已转交人工客服" in d["response"], f"回复: {d['response']}"
        assert d["compliance_passed"] is False

    def d1_knowledge_followup():
        d1 = ask(client, "d1", "理财产品A的年化收益率是多少")
        assert d1["intent"] == "knowledge_rag", f"路由: {d1['intent']}"
        d2 = ask(client, "d1", "投资期限最短是多久")
        assert d2["intent"] == "knowledge_rag", f"追问路由: {d2['intent']}"
        assert ("6个月" in d2["response"]) or ("6 个月" in d2["response"]), \
            f"追问回复: {d2['response']}"

    def d2_category_switch():
        d1 = ask(client, "d2", "理财产品A的年化收益率是多少")
        assert d1["intent"] == "knowledge_rag", f"路由: {d1['intent']}"
        d2 = ask(client, "d2", "我决定退款了，帮我创建一个退款工单，订单77777的退款没到账")
        assert d2["intent"] == "ticket_handler", f"切换路由: {d2['intent']}"
        assert TICKET_ID_RE.search(d2["response"]), f"回复: {d2['response']}"

    return [
        ("A1 知识-收益率", a1_yield),
        ("A2 知识-退款政策", a2_refund_policy),
        ("A3 知识-开户材料", a3_account_guide),
        ("B1 工单-建单", b1_create),
        ("B2 工单-凭号查询", b2_create_then_query),
        ("B3 工单-无号查询", b3_query_without_id),
        ("C1 违规-伪造发票", c1_forge_invoice),
        ("C2 违规-网络赌博", c2_gambling),
        ("D1 多轮-知识追问", d1_knowledge_followup),
        ("D2 多轮-跨类切换", d2_category_switch),
    ]


def e_system_metrics() -> None:
    """系统级收尾：指标闭环（1.2 接线的系统级最终验证）"""
    summary = get_metrics().get_summary()

    missing = [a for a in EXPECTED_AGENTS if a not in summary]
    assert not missing, f"缺 agent 指标: {missing}"

    for name, s in summary.items():
        assert s["error_rate"] == 0, f"{name} 错误率 {s['error_rate']}"

    # 拓扑级断言：图上每条路径必过 compliance_check，
    # 所以 compliance_process 的调用数应精确等于总轮次
    assert summary["compliance_process"]["total_calls"] == turns_total, \
        f"compliance 调用 {summary['compliance_process']['total_calls']} != 总轮次 {turns_total}"


def main() -> int:
    with TestClient(app) as client:
        for name, fn in build_scenarios(client):
            check(name, fn)

        check("E 系统指标闭环", e_system_metrics)

    print()
    failed = 0
    for name, status in results:
        ok = status == "PASS"
        failed += 0 if ok else 1
        print(f"[{'PASS' if ok else '****'}] {name} {'' if ok else status}")

    # 性能参考（不断言，供调优与对比用）
    print("\n性能参考（avg_duration_ms，不断言）:")
    for name, s in sorted(get_metrics().get_summary().items()):
        print(f"  {name:28s} calls={s['total_calls']:3d}  "
              f"avg={s['avg_duration_ms']:8.1f}ms  err={s['error_rate']}")

    print()
    if failed:
        print(f"MILESTONE FAILED：{failed} 项失败（共 {len(results)} 项）")
        return 1
    print(f"MILESTONE PASS：{len(results)}/{len(results)} 项通过，里程碑一验收完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
