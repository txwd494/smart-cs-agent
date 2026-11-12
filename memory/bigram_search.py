"""
memory.bigram_search — 字符二元组检索（长期记忆的降级路径）

字符二元组评分逻辑，独立成降级模块：真实 embedding 不可用时的
检索兜底（调用方 memory/long_term.py 的 search 降级分支）。

评分：查询与文档各自取字符二元组集合（相邻双字），重合数除以
查询二元组数，得 0 到 1 分。二元组是中文的最小可用检索单元：
不依赖分词器，无字面重合的语义近似（口语化查询）无法命中，
这正是需要真实 embedding 的原因。
"""

from __future__ import annotations


def _clean(text: str) -> str:
    """小写并去空白，统一检索前的文本形态"""
    return "".join(ch for ch in text.lower() if not ch.isspace())


def _bigrams(text: str) -> set[str]:
    """字符二元组集合（相邻双字）"""
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _overlap_score(query: str, content: str) -> float:
    """
    查询与文档的相关性评分（0-1）。

    查询二元组的命中比例。单字符查询退化为包含判断
    （二元组需要至少两个字符）。
    """
    q, c = _clean(query), _clean(content)
    if not q or not c:
        return 0.0
    if len(q) == 1:
        return 1.0 if q in c else 0.0
    q_grams = _bigrams(q)
    return len(q_grams & _bigrams(c)) / len(q_grams)
