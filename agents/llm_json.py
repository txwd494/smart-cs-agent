"""
agents.llm_json — LLM JSON 输出的共享解析助手

从 LLM 的自由文本响应中提取 JSON 对象。当前消费者：
- agents/intent_router.py（意图分类）
- agents/ticket_handler.py（工单信息提取）
- agents/compliance_checker.py（合规审查结果）

做法：截取首个 { 到最后一个 } 之间的片段再 json.loads，
markdown 围栏（```json ... ```）与 JSON 前后的解释文字天然兼容。

为什么不做直 json.loads：LLM 带围栏或解释文字输出时直解析必然
失败，结果被静默丢弃走兜底，分类 / 工单 / 审查结果凭空丢失
且无痕迹。围栏是中文模型的常见输出形态，这层兼容是必需品。
"""

from __future__ import annotations

import json
from typing import Any


def parse_llm_json(content: str) -> dict[str, Any] | None:
    """
    从 LLM 输出中提取 JSON 对象，失败返回 None（由调用方走各自兜底）。

    局限：若解释文字本身含大括号，截取起点可能错位而解析失败，
    返回 None 走兜底，属可接受降级。
    """
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
