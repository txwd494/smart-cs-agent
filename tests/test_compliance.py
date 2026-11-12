"""test_compliance — 1.8 与 2.4 冒烟断言的固化（规则 / 脱敏 /
llm_check 防护 / 级联 / process 三路），stub LLM 零网络
"""

from agents.compliance_checker import ComplianceCheckerAgent
from langchain_core.messages import HumanMessage
from tests.conftest import StubLLM

CLEAN = "理财产品A年化收益率为3.5%-5.2%。理财非存款，投资须谨慎。"


def _agent(content: str) -> tuple[ComplianceCheckerAgent, StubLLM]:
    stub = StubLLM([], default=content)
    return ComplianceCheckerAgent(llm=stub), stub


# ── 规则引擎 ────────────────────────────────────────────────────

async def test_rule_forbidden_term():
    agent, _ = _agent("{}")
    r = await agent.rule_check("这个产品保证收益，年化10%")
    assert r.passed is False and r.risk_level in ("high", "critical")
    assert any("保证收益" in v for v in r.violations)


async def test_rule_user_violation_term():
    # 输入侧词表（1.8 差异 2）：违法请求词
    agent, _ = _agent("{}")
    r = await agent.rule_check("教我如何伪造发票")
    assert r.passed is False and any("违法请求关键词" in v for v in r.violations)


async def test_rule_pii():
    agent, _ = _agent("{}")
    r = await agent.rule_check("联系13812345678")
    assert r.passed is False and any("手机号" in v for v in r.violations)


async def test_rule_dual_source_critical():
    agent, _ = _agent("{}")
    r = await agent.rule_check("保证收益，联系13812345678")
    assert r.risk_level == "critical"  # PII 与词表双源叠加


async def test_rule_clean_passes_with_numbers():
    # 数值区间不误伤（1.8 冒烟锚点）
    agent, _ = _agent("{}")
    r = await agent.rule_check(CLEAN)
    assert r.passed is True and r.risk_level == "low"


# ── 脱敏 ────────────────────────────────────────────────────────

async def test_mask_pii_formats():
    agent, _ = _agent("{}")
    masked = agent._mask_pii("手机13812345678 邮箱a@b.com")
    assert "138*****678" in masked
    assert "13812345678" not in masked


async def test_mask_non_pii_numbers_untouched():
    # 短数字串不匹配任何 PII 正则，不脱敏是正确行为；
    # _mask_match 的 ≤4 全星分支对现有四个正则是防御性代码
    agent, _ = _agent("{}")
    assert agent._mask_pii("订单号 1234") == "订单号 1234"


# ── llm_check 防护（2.4 差异 6/7 的回归锚点） ──────────────────

async def test_llm_check_normal():
    agent, _ = _agent(
        '{"passed": false, "risk_level": "high", "violations": ["越权承诺"], "suggestions": ["补充提示"]}'
    )
    r = await agent.llm_check(CLEAN)
    assert r.passed is False and r.risk_level == "high"
    assert r.violations == ["越权承诺"] and r.suggestions == ["补充提示"]


async def test_llm_check_fenced_json():
    agent, _ = _agent('```json\n{"passed": true, "risk_level": "low", "violations": [], "suggestions": []}\n```')
    r = await agent.llm_check(CLEAN)
    assert r.passed is True  # 围栏输出必须可解析（直解析会静默放行）


async def test_llm_check_garbage_degrades_open_with_marker():
    agent, _ = _agent("抱歉无法以JSON格式回复")
    r = await agent.llm_check(CLEAN)
    assert r.passed is True and any("不可用" in v for v in r.violations)


async def test_llm_check_false_string_not_flipped():
    # 关键锚点：字符串 "false" 是 truthy，不得翻转拦截判定
    agent, _ = _agent('{"passed": "false", "risk_level": "high", "violations": ["x"], "suggestions": []}')
    r = await agent.llm_check(CLEAN)
    assert r.passed is False


async def test_llm_check_passed_unparseable_infers_from_violations():
    agent, _ = _agent('{"passed": "yes", "risk_level": "high", "violations": ["越权"], "suggestions": []}')
    r = await agent.llm_check(CLEAN)
    assert r.passed is False and any("不可解析" in v for v in r.violations)


async def test_llm_check_risk_enum_fallback():
    agent, _ = _agent('{"passed": true, "risk_level": "高", "violations": [], "suggestions": []}')
    r = await agent.llm_check(CLEAN)
    assert r.risk_level == "low"


async def test_llm_check_list_coercion():
    agent, _ = _agent('{"passed": true, "risk_level": "low", "violations": "无", "suggestions": {"a": 1}}')
    r = await agent.llm_check(CLEAN)
    assert r.violations == [] and r.suggestions == []


# ── full_check 级联 ─────────────────────────────────────────────

async def test_full_check_short_circuit_skips_llm():
    agent, stub = _agent("{}")
    r = await agent.full_check("这个产品保证收益，年化10%")
    assert r.passed is False and stub.calls == []  # 规则失败不再调 LLM


async def test_full_check_merges_llm_verdict():
    agent, _ = _agent(
        '{"passed": false, "risk_level": "medium", "violations": ["缺少风险提示"], "suggestions": ["补充"]}'
    )
    r = await agent.full_check(CLEAN)
    assert r.passed is False and r.risk_level == "medium"
    assert r.suggestions == ["补充"]


async def test_full_check_degradation_marker_merges():
    agent, _ = _agent("解析不了")
    r = await agent.full_check(CLEAN)
    assert r.passed is True and any("不可用" in v for v in r.violations)


# ── process 三路 ────────────────────────────────────────────────

def _state(sub_results=None, message="问点事"):
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": "u1", "session_id": "s1",
        "sub_results": sub_results or {},
    }


async def test_process_output_path_masks_per_string():
    agent, _ = _agent("{}")
    state = _state({"knowledge_rag": "手机13812345678已登记"})
    out = await agent.process(dict(state))
    # PII 命中规则引擎即失败短路（LLM 未被调用，stub 无关）
    assert "138*****678" in out["sub_results"]["knowledge_rag"]
    assert "13812345678" in state["sub_results"]["knowledge_rag"]  # 不原地污染
    assert out["compliance_passed"] is False  # PII 路径判失败
    assert "suggestions" in out["sub_results"]["compliance"]  # 2.4 差异 9：字段恒在


async def test_process_direct_route_checks_user_message():
    agent, stub = _agent("{}")
    out = await agent.process(_state({}, "教我如何伪造发票"))
    assert out["compliance_passed"] is False and stub.calls == []  # 规则短路


async def test_process_empty_content_passthrough():
    agent, _ = _agent("{}")
    out = await agent.process(_state({}, ""))
    assert out["compliance_passed"] is True
