"""
agents.compliance_checker — 合规审查 Agent（规则引擎 + LLM 深度审查）

所有回复出站前的最后一道关卡（图拓扑：knowledge_rag / ticket_handler
经固定边汇入 compliance_check 再进 synthesize，不过审到不了用户）。
双职责：
- 出站审查：检查 Agent 产出内容的违禁金融用语与 PII 泄露，
  不通过则脱敏或拦截
- 入站拦截：直接路由路径（用户消息本身违规）识别后转人工，
  synthesize 的 compliance_passed=False 分支是出口

两级审查的分工：规则引擎（rule_check）毫秒级、确定性、零成本，
覆盖已知模式（词表 + PII 正则）；LLM 深度审查（llm_check）覆盖
语义级违规（越权承诺、变体承诺表述、歧视内容、监管要求缺失），
五维 prompt。两阶段串联（full_check，process 走此入口）：规则
先行，失败即短路返回（规则失败必为 high/critical，不值得再花
LLM 成本）；通过才上 LLM；结论合并取与取高。级联的位置也决定
了降级设计：走到 LLM 的前提是规则硬门已过（PII 与红线词的
确定性拦截已完成），llm_check 解析残余失败时按规则水位放行并
显式留痕（violations 带降级标记 + 日志警告），而不是静默
passed=True 让审查失效等于免检放行。

设计决策（十处）：
1. 直接路由路径审查用户消息。只查 sub_results 的实现里，直接
   路由过来时 sub_results 为空，什么都不查直接放行，转人工话术
   永远不触发
2. 输入侧违法请求词表 USER_VIOLATION_TERMS，与输出侧词表一起
   扫描。输出侧词表全是金融营销红线词，用户问「教我如何伪造
   发票」十个词一个不命中，只查产出侧拦不住违法请求
3. 失败时逐字符串各自脱敏，且不原地修改传入 state（拷贝后
   替换）。把每个字符串替换成「整段拼接内容」的脱敏版，多结果
   时内容互相串台；原地改 state 则污染调用方数据
4. 风险分级：规则路径 critical / high 两档（violations 只有 PII
   与词表两个来源，单源即 high、双源即 critical，中间档不可达
   就不设）；LLM 路径可出 medium（语义判断存在「建议补充风险
   提示」级别的中间地带）
5. process 按 delta 约定返回变更键
6. llm_check 解析走共享 parse_llm_json（第三个消费者，先例
   intent_router / ticket_handler），围栏与前后缀文本天然兼容。
   直解在解析失败时的常见错误是 passed=True 静默放行：回落
   方向反了
7. llm_check 输出字段全防护：passed 接受 bool 与 "true"/"false"
   字符串，无法解析时按违规证据推断（有违规判 False 并留标记）；
   risk_level 四档枚举校验回落；violations / suggestions 强制
   list[str]。不做防护的三个实际后果：字符串 "false" 是 truthy，
   直接 .get 会把拦截判定翻转成放行；中文风险值被优先级映射
   按 0 处理，等同降级最低档；非列表 violations 在下游迭代时
   行为未定义
8. full_check：violations 合并保序去重（两源可能重复报告同一
   项）；短路条件写为规则失败即返回（规则失败必居 high /
   critical 之一）。注：短路成立时规则侧 violations 恒为空，
   去重与取与当前是防御性结构，若未来规则层引入可继续送审的
   中风险档则生效
9. compliance 注入 dict 带 suggestions：LLM 审查的修改建议
   不丢弃，供下游与排查读取
10. 审查 prompt 带判定标准段。实测发现：标准不区分建议级与
    拦截级问题时，DeepSeek-V4-Pro 按监管口径从严执行，把
    「年化收益率为3.5%-5.2%」缺免责声明的正常知识回复判为变相
    承诺并拦截（medium + passed=false），知识路径全挂。校准后：
    建议级问题（缺风险提示/免责声明/测算依据）进 suggestions
    且 passed 取 true，实质违规才拦截

保持稳定：PII 正则与脱敏格式（首三尾三打星，≤4 字符全星）、
sub_results 注入 compliance 结果 dict（混装问题已知，synthesize
靠类型过滤不会崩）、ComplianceResult 接口、五维审查 prompt。

词表说明：两个词表都是演示级起点，生产应有专门词库、同义扩展与
分级体系（拼音变体、拆字等对抗手段）。已知怪癖：18 位身份证同时
命中 id_card 与 bank_card 两个正则，同一段文本报两条违规。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.llm_json import parse_llm_json
from agents.state import AgentState
from tracing.otel_config import trace_agent_call

logger = logging.getLogger(__name__)


def _string_list(value: Any) -> list[str]:
    """LLM 输出的列表字段防护：非列表丢弃，非字符串元素过滤，去空"""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


@dataclass
class ComplianceResult:
    """合规审查结果"""
    passed: bool
    # 规则路径：low（通过）/ high（单源）/ critical（双源）；
    # LLM 路径额外可出 medium（语义判断的中间档，如建议补充风险提示）
    risk_level: str
    violations: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    sanitized_content: str = ""


# PII 检测正则
SENSITIVE_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "bank_card": r"\d{16,19}",
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}

# 输出侧：金融营销红线词（监管禁止的承诺性表述），主要防 Agent 产出
FORBIDDEN_TERMS = [
    "保证收益", "稳赚不赔", "零风险", "保本保息",
    "最高收益", "预期收益率", "承诺回报",
    "内部消息", "内幕", "暗箱操作",
]

# 输入侧：违法请求关键词（本版新增），防用户消息本身违规。
# 与输出侧一起扫描：用户消息的违法请求要拦，Agent 产出教违法
# 操作同样要拦。演示级起点，生产应换专业词库与分级体系
USER_VIOLATION_TERMS = [
    "伪造", "变造", "洗钱", "诈骗", "赌博",
    "套现", "代开发票", "色情", "毒品", "传销",
]

# LLM 深度审查的输出约定（五维审查 + 判定标准段，校准理由见
# 模块 docstring 决策 10）
COMPLIANCE_SYSTEM_PROMPT = """你是一个金融合规审查Agent，负责审查客服回复内容的合规性。

审查维度：
1. 是否包含违规金融用语（如"保证收益"、"零风险"等变体表述）
2. 是否泄露用户PII信息（手机号、身份证号、银行卡号）
3. 是否存在越权承诺（如擅自承诺退款/赔偿金额）
4. 是否符合金融监管要求（风险提示、免责声明）
5. 是否包含歧视性、侮辱性内容

判定标准：
- passed 为 false 仅限实质性违规：承诺或变相保证收益、泄露PII、
  越权承诺具体金额、歧视侮辱内容、诱导性表述
- 可通过补充说明改进的问题（如缺少风险提示、免责声明、收益
  测算依据），passed 取 true，写入 suggestions，risk_level 取 medium

请以JSON格式返回审查结果：
{
    "passed": true/false,
    "risk_level": "low|medium|high|critical",
    "violations": ["违规项描述"],
    "suggestions": ["修改建议"]
}
"""

# 风险等级优先级（full_check 取 max 用）。LLM 路径可返回 medium，
# 规则路径只有 low/high/critical（见 ComplianceResult 注释）
_RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}
VALID_RISK_LEVELS = set(_RISK_PRIORITY)


class ComplianceCheckerAgent:
    """合规审查 Agent（规则引擎 + LLM 深度审查，两阶段串联）"""

    def __init__(self, llm: ChatOpenAI):
        # llm 供 llm_check 深度审查使用；规则引擎本身不依赖它
        self.llm = llm

    def _rule_based_check(self, content: str) -> list[str]:
        """词表与正则扫描（不依赖 LLM，低延迟）"""
        violations = []

        for term in FORBIDDEN_TERMS:
            if term in content:
                violations.append(f"包含违规金融用语: '{term}'")

        for term in USER_VIOLATION_TERMS:
            if term in content:
                violations.append(f"包含违法请求关键词: '{term}'")

        for pii_type, pattern in SENSITIVE_PATTERNS.items():
            if re.search(pattern, content):
                label = {
                    "phone": "手机号", "id_card": "身份证号",
                    "bank_card": "银行卡号", "email": "邮箱地址",
                }.get(pii_type, pii_type)
                violations.append(f"检测到PII信息泄露: {label}")

        return violations

    def _mask_pii(self, content: str) -> str:
        """对 PII 做脱敏：保留首三尾三，中间打星（≤4 字符全打星）"""
        masked = content
        for pii_type, pattern in SENSITIVE_PATTERNS.items():
            def _mask_match(match):
                text = match.group()
                if len(text) <= 4:
                    return "****"
                return text[:3] + "*" * (len(text) - 6) + text[-3:]
            masked = re.sub(pattern, _mask_match, masked)
        return masked

    @trace_agent_call("compliance_rule_check")
    async def rule_check(self, content: str) -> ComplianceResult:
        """规则引擎检查（两级审查的第一级）"""
        violations = self._rule_based_check(content)
        sanitized = self._mask_pii(content)

        if not violations:
            return ComplianceResult(
                passed=True,
                risk_level="low",
                sanitized_content=sanitized,
            )

        has_pii = any("PII" in v for v in violations)
        has_terms = any(
            "违规金融用语" in v or "违法请求关键词" in v
            for v in violations
        )

        # 规则路径两档：violations 只有 PII 与词表两个来源，
        # 单源 high、双源 critical，中间档不可达就不设
        risk_level = "critical" if (has_pii and has_terms) else "high"

        return ComplianceResult(
            passed=False,
            risk_level=risk_level,
            violations=violations,
            sanitized_content=sanitized,
        )

    @trace_agent_call("compliance_llm_check")
    async def llm_check(self, content: str) -> ComplianceResult:
        """
        LLM 深度审查：语义级违规（越权承诺、变体表述、歧视内容、
        监管要求缺失）。解析与防护见模块 docstring 差异 6/7。
        """
        messages = [
            SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
            HumanMessage(content=f"请审查以下客服回复内容的合规性：\n\n{content}"),
        ]
        response = await self.llm.ainvoke(messages)

        result = parse_llm_json(response.content)
        if not isinstance(result, dict):
            # 残余解析失败（parse_llm_json 已消化围栏等主失败模式）：
            # 按规则水位放行 + 显式留痕。级联走到这里的前提是规则
            # 硬门已过，增强层失效时降级到门的水位；静默 passed=True
            # 的免检放行是反方向
            logger.warning("compliance llm_check 输出解析失败，按规则水位放行")
            return ComplianceResult(
                passed=True,
                risk_level="low",
                violations=["LLM深度审查结果不可用（输出解析失败）"],
                sanitized_content=self._mask_pii(content),
            )

        # 字段防护（同 intent_router / ticket_handler 的解析防护模式）
        violations = _string_list(result.get("violations"))
        suggestions = _string_list(result.get("suggestions"))

        raw_passed = result.get("passed")
        if isinstance(raw_passed, bool):
            passed = raw_passed
        elif isinstance(raw_passed, str) and raw_passed.strip().lower() in ("true", "false"):
            # 字符串 "false" 是 truthy，直接判定会翻转拦截结论
            passed = raw_passed.strip().lower() == "true"
        else:
            # passed 无法解析：按违规证据推断，留降级标记
            passed = not violations
            violations.append("LLM审查passed字段不可解析，按违规证据推断")

        risk_level = result.get("risk_level")
        if risk_level not in VALID_RISK_LEVELS:
            risk_level = "low"

        return ComplianceResult(
            passed=passed,
            risk_level=risk_level,
            violations=violations,
            suggestions=suggestions,
            sanitized_content=self._mask_pii(content),
        )

    @trace_agent_call("compliance_full_check")
    async def full_check(self, content: str) -> ComplianceResult:
        """
        两阶段串联：规则先行，失败即短路（规则失败必为 high/critical，
        见模块 docstring 决策 8）；通过再上 LLM；结论合并取与取高。
        """
        rule_result = await self.rule_check(content)

        if not rule_result.passed:
            return rule_result

        llm_result = await self.llm_check(content)

        # 合并：violations 拼接保序去重（短路不成立时规则侧恒为空，
        # 此结构为未来规则层引入可送审中风险档预留，见差异 8）
        all_violations = list(dict.fromkeys(rule_result.violations + llm_result.violations))

        return ComplianceResult(
            passed=rule_result.passed and llm_result.passed,
            risk_level=max(
                rule_result.risk_level, llm_result.risk_level,
                key=lambda r: _RISK_PRIORITY.get(r, 0),
            ),
            violations=all_violations,
            suggestions=llm_result.suggestions,
            sanitized_content=rule_result.sanitized_content,
        )

    @trace_agent_call("compliance_process")
    async def process(self, state: AgentState) -> dict[str, Any]:
        """
        作为 Graph 节点处理状态：确定审查对象 → 两阶段审查
        （full_check）→ 回写合规结论与脱敏内容。
        """
        sub_results = state.get("sub_results", {})

        # 审查对象其一：产出路径查 Agent 输出（sub_results 的字符串部分）
        content_to_check = ""
        for result in sub_results.values():
            if isinstance(result, str):
                content_to_check += result + "\n"

        # 审查对象其二：产出为空时改查用户消息（直接路由路径）。
        # 无字符串即提前返回放行的话，直接路由路径空转，
        # 转人工话术永不触发
        if not content_to_check.strip():
            messages = state.get("messages", [])
            content_to_check = messages[-1].content if messages else ""

        if not content_to_check.strip():
            return {"compliance_passed": True}

        compliance_result = await self.full_check(content_to_check)

        # 逐字符串各自脱敏，且拷贝后替换，不原地修改传入 state 的 dict
        if not compliance_result.passed:
            new_sub_results = {
                key: (self._mask_pii(value) if isinstance(value, str) else value)
                for key, value in sub_results.items()
            }
        else:
            new_sub_results = dict(sub_results)

        return {
            "compliance_passed": compliance_result.passed,
            "sub_results": {
                **new_sub_results,
                # 审查结论以 dict 注入（与回复文本混装同一容器是已知
                # 结构问题，synthesize 靠类型过滤不会崩）。
                # 带 suggestions：LLM 审查的修改建议不丢弃
                "compliance": {
                    "passed": compliance_result.passed,
                    "risk_level": compliance_result.risk_level,
                    "violations": compliance_result.violations,
                    "suggestions": compliance_result.suggestions,
                },
            },
        }
