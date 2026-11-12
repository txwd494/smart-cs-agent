"""conftest — pytest 套件公共夹具（3.1）

两级测试的公共依赖都在这里：
- stub_llm / StubLLM：提示匹配式假 LLM（按收到的系统提示关键词
  返回脚本响应，不依赖图内部调用次序，结构改动不连带测试全崩）
- FakeEmbeddings：确定性向量假实现（文本哈希生成向量，接口对齐
  aembed_documents / aembed_query，真实建 FAISS 索引但零网络）
- stub_event_loop 内跑的异步测试由 pytest-asyncio auto 模式承接

写透约定：单测文件按源码模块一对一（test_working_memory.py 对
memory/working_memory.py），断言口径与已删除的各模块冒烟一致
（那批冒烟是这些断言的第一次验证）。
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest


class StubLLM:
    """提示匹配式假 LLM。

    rules: (匹配子串, 回复文本) 列表，按序取首个命中；都不命中用
    default。匹配对象是消息列表拼串（系统提示 + 用户内容），覆盖
    各 Agent 的系统提示关键词即可稳定路由响应。
    """

    def __init__(self, rules: list[tuple[str, str]], default: str = "{}"):
        self.rules = rules
        self.default = default
        self.calls: list[str] = []  # 收到的消息拼串，供调用断言

    async def ainvoke(self, messages):
        text = "\n".join(str(m.content) for m in messages)
        self.calls.append(text)
        for pattern, response in self.rules:
            if pattern in text:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content=self.default)


class FakeEmbeddings:
    """确定性假嵌入：文本 sha256 哈希铺成向量再归一化。

    语义上无意义（同文本同向量、异文本近似正交），测试只依赖
    确定性：同文本两次嵌入结果一致，建索引与检索可复现。
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.doc_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [b / 255.0 for b in (digest * ((self.dim // len(digest)) + 1))[: self.dim]]
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    async def aembed_documents(self, texts):
        self.doc_calls += len(texts)
        return [self._vector(t) for t in texts]

    async def aembed_query(self, text):
        self.query_calls += 1
        return self._vector(text)


class BrokenEmbeddings:
    """任何调用都抛异常：测 add 失败路径与 search 降级路径"""

    async def aembed_documents(self, texts):
        raise RuntimeError("embedding service down")

    async def aembed_query(self, text):
        raise RuntimeError("embedding service down")


@pytest.fixture
def stub_llm_factory():
    """按需构造 StubLLM 的工厂（多数测试文件各自内联构造即可）"""
    return lambda rules, default="{}": StubLLM(rules, default)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()
