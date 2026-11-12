"""
memory.long_term — 长期记忆（真实向量检索版）

分层记忆的第三层：知识文档的存储与语义检索，为知识检索 Agent
（agents/knowledge_rag.py）的 retrieve_documents 提供后端。

技术栈（五件事是一个整体）：
- 真实 embedding：GLM-Embedding-3（实测 2048 维，OpenAI 兼容
  协议）。语义检索的前提
- FAISS 索引：IndexFlatIP + 归一化向量 = cosine 相似度
  （精确检索，当前规模够用）
- 文档分块：长文档切块入库，块元数据带文件名与块序号（可溯源）
- 磁盘持久化：save 落盘索引 + 文档元数据，init 时加载。真实
  embedding 下这是必需品：索引不落盘等于每次重启全库重新付费
  嵌入
- 知识库目录加载：文档来自 data/knowledge_base/ 目录而非内联
  字符串

降级路径：字符二元组重合度检索（memory/bigram_search.py）作为
embedding 服务不可用时的兜底（与短期记忆的 Redis 降级同构）。
读写失败策略不对称，理由：add_document 嵌入失败抛异常（无向量的
文档进不了索引，静默收下等于不可检索的假入库）；search 嵌入失败
降级二元组加日志（服务仍可用，语义质量回落）。

实现要点（六处）：
1. 真实 embedding 驱动语义检索
2. 维度不硬编码：首次嵌入结果定维再建索引。硬编码 1536 与特定
   模型耦合，换模型即 FAISS 断言崩溃
3. 批量入库一次 aembed_documents（N 篇一次往返，非逐条循环）
4. 分块规则：切分只按中文句读（。！？）与换行，块边界不会落进
   小数（按英文句点切会把 3.5% 切成两块，文本虽可还原但边界
   劣化检索质量）；句子分支与段落分支共用 overlap；超长句硬切
   兜底
5. save 与 load-on-init 实际接线（见上），chunk 元数据带序号
6. 方法为 async（嵌入是网络 IO）

已知边界：索引与知识库目录的陈旧性（save 后 KB 文件改动不反映，
需删索引文件重建，rebuild 开关未展开）；IndexFlatIP 复杂度随库
线性，大规模需 IVF/HNSW；embedding 服务为外部依赖（延迟、计费、
降级时语义质量回落）；GLM-Embedding-3 对无词汇重合的短口语查询
top-1 判别弱（实测「钱放你们那能赚多少」被开户指南以 0.45 vs 0.35
反超；BGE 式指令前缀可救回口语查询但压塌精确查询的区分度，无净
收益未采用），管线靠 RAG 的查询改写与 LLM 重排兜底，本层只承担
召回职责；无锁（单事件循环唯一写入口，同 working memory 的分析）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from memory.bigram_search import _clean, _overlap_score

try:
    import faiss
except ImportError:
    faiss = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None


def _default_embeddings():
    """兜底 embedding 客户端：从环境变量构造（.env 由入口加载）。"""
    if OpenAIEmbeddings is None:
        return None
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL_NAME", "GLM-Embedding-3"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# 句读切分：中文句号/问号/叹号与换行。不用英文句点，
# 避免块边界落进小数（3.5%）与缩写（Mr.）
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> list[str]:
    """
    文本分块：段落优先、句读兜底、超长硬切，统一 overlap 窗口。

    切分只按中文句读，不按英文句点：句点会把块边界落进小数；
    句子分支与段落分支共用 overlap；超长句硬切兜底。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            # overlap：下一块以本块尾部开头（段落与句读分支统一）
            current = current[-overlap:] if len(current) > overlap else ""

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += para + "\n\n"
            continue
        # 段落装不下：按句读切
        for sentence in [s for s in _SENTENCE_SPLIT.split(para) if s.strip()]:
            if len(current) + len(sentence) <= chunk_size:
                current += sentence
                continue
            flush()
            # 超长单句硬切（差异 4 第三处）
            while len(sentence) > chunk_size:
                chunks.append(sentence[:chunk_size])
                sentence = sentence[chunk_size - overlap:]
            current = sentence
    flush()

    # 收尾：current 里剩的是 overlap 尾巴或短内容，太短并入末块
    if current and len(current) > overlap:
        chunks.append(current)
    return [c.strip() for c in chunks if len(c.strip()) > overlap] or [text[:chunk_size]]


class LongTermMemory:
    """长期记忆：真实 embedding + FAISS + 持久化（架构见模块 docstring）"""

    def __init__(self, index_path: str | None = None, embeddings=None):
        """
        Args:
            index_path: 持久化路径（None 则纯内存，不落盘）；索引与
                        文档元数据存 <index_path> 与 <index_path>.meta.json
            embeddings: 嵌入客户端（OpenAI 兼容）。None 则从 .env 构造；
                        注入抛异常的 stub 可测降级路径
        """
        self.index_path = Path(index_path) if index_path else None
        self._embeddings = embeddings if embeddings is not None else _default_embeddings()
        self._documents: list[dict[str, Any]] = []
        self._index: Any = None  # 首次嵌入定维后创建（差异 2：不硬编码）
        self._load_from_disk()

    # ── 持久化 ─────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """启动时加载落盘索引与文档元数据（有索引文件即加载，重启零嵌入成本）"""
        if self.index_path is None or faiss is None:
            return
        if not self.index_path.exists():
            return
        meta_path = self.index_path.with_suffix(".meta.json")
        try:
            self._index = faiss.read_index(str(self.index_path))
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._documents = json.load(f)
        except Exception:
            # 索引文件损坏：弃档重建，等同无持久化
            self._index = None
            self._documents = []

    def save(self) -> None:
        """落盘索引与文档元数据（真实 embedding 下的必需品）"""
        if self.index_path is None or faiss is None or self._index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.index_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    # ── 嵌入与索引 ──────────────────────────────────────────────────

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入。维度探测在此完成（首次调用建索引）"""
        vectors = await self._embeddings.aembed_documents(texts)
        if self._index is None and faiss is not None:
            dim = len(vectors[0])
            self._index = faiss.IndexFlatIP(dim)
        return vectors

    def _add_to_index(self, vectors: list[list[float]]) -> None:
        """归一化后进 FAISS（归一化 + 内积 = cosine）"""
        if self._index is None or faiss is None:
            return
        arr = np.array(vectors, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        self._index.add(arr)

    # ── 对外接口 ───────────────────────────────────────────────────

    async def add_document(self, content: str, source: str = "", metadata: dict | None = None) -> str:
        """
        添加一篇文档（整体入库，不分块；分块入口是 add_chunks /
        load_knowledge_base）。嵌入失败抛异常且不残留文档：先嵌入
        成功再入库。顺序反了的话，异常路径上半篇文档已进 _documents
        却没有向量，正是要避免的「不可检索的假入库」。
        """
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        vectors = await self._embed([content]) if self._embeddings is not None else None
        self._documents.append({
            "id": doc_id,
            "content": content,
            "source": source,
            "metadata": metadata or {},
        })
        if vectors is not None:
            self._add_to_index(vectors)
        return doc_id

    async def add_chunks(self, chunks: list[str], source: str, metadata: dict | None = None) -> list[str]:
        """批量入库分块（一次 aembed_documents，差异 3）。元数据带块序号。"""
        if not chunks:
            return []
        vectors = await self._embed(chunks) if self._embeddings is not None else None
        doc_ids = []
        for i, chunk in enumerate(chunks):
            meta = {**(metadata or {}), "chunk_index": i, "chunk_total": len(chunks)}
            doc_id = hashlib.md5(f"{source}:{i}:{chunk}".encode()).hexdigest()[:12]
            self._documents.append({
                "id": doc_id,
                "content": chunk,
                "source": source,
                "metadata": meta,
            })
            doc_ids.append(doc_id)
        if vectors is not None:
            self._add_to_index(vectors)
        return doc_ids

    async def load_knowledge_base(self, kb_dir: str) -> int:
        """从目录批量加载知识库（txt），分块入库，返回块数"""
        kb_path = Path(kb_dir)
        if not kb_path.exists():
            return 0
        count = 0
        for file_path in sorted(kb_path.glob("**/*.txt")):
            content = file_path.read_text(encoding="utf-8")
            chunks = _chunk_text(content)
            await self.add_chunks(chunks, source=file_path.name, metadata={"file": str(file_path)})
            count += len(chunks)
        return count

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        语义检索 top_k 篇文档（cosine 相似度降序）。

        embedding 服务不可用时降级为字符二元组重合度检索（1.6 的
        实现，保留于此作兜底），语义质量回落但不拒服务。
        """
        if self._embeddings is not None and self._index is not None and self._documents:
            try:
                qvec = (await self._embeddings.aembed_query(query))
                if faiss is not None:
                    arr = np.array([qvec], dtype=np.float32)
                    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
                    k = min(top_k, len(self._documents))
                    scores, indices = self._index.search(arr, k)
                    results = []
                    for score, idx in zip(scores[0], indices[0]):
                        if 0 <= idx < len(self._documents):
                            doc = dict(self._documents[idx])
                            doc["score"] = float(score)
                            results.append(doc)
                    return results
            except Exception:
                # 降级：嵌入服务不可用，回落二元组
                pass
        return self._bigram_search(query, top_k)

    def _bigram_search(self, query: str, top_k: int) -> list[dict]:
        """二元组降级检索（1.6 的 _overlap_score，语义见其注释）"""
        scored = []
        q = _clean(query)
        if not q:
            return []
        for doc in self._documents:
            score = _overlap_score(query, doc["content"])
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**doc, "score": s} for s, doc in scored[:top_k]]

    async def clear(self) -> None:
        """清空内存与落盘索引（测试与重建用）"""
        self._documents = []
        self._index = None
        if self.index_path is not None:
            for p in [self.index_path, self.index_path.with_suffix(".meta.json")]:
                if p.exists():
                    p.unlink()
