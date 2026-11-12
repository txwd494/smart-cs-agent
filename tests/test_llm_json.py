"""test_llm_json — 1.5/1.7/2.4 三个消费者的共享解析助手"""

import pytest

from agents.llm_json import parse_llm_json


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),                                    # 裸 JSON
    ('```json\n{"a": 1}\n```', {"a": 1}),                      # 围栏
    ('好的，结果如下：\n{"a": 1}\n以上。', {"a": 1}),          # 前后缀文本
    ('{"a": {"b": [1, 2]}}', {"a": {"b": [1, 2]}}),            # 嵌套
])
def test_parse_valid(text, expected):
    assert parse_llm_json(text) == expected


@pytest.mark.parametrize("text", [
    "完全不是 JSON",
    "[1, 2, 3]",       # 非 dict 顶层
    "",
])
def test_parse_invalid_returns_none(text):
    assert parse_llm_json(text) is None


def test_extract_first_to_last_brace():
    # 截取首个 { 到最后一个 }：单对象带前后缀可解
    assert parse_llm_json('前缀 {"a": 1} 后缀') == {"a": 1}
    # 已知局限（docstring 记录）：解释文字含大括号时起止错位，返回 None 走兜底
    assert parse_llm_json('{"a": 1} 中间文字 {"b": 2}') is None
