"""
agents.knowledge_rag — 知识检索 Agent（RAG 四步流水线）

知识查询路径的执行者：拿用户问题到长期记忆检索相关文档，基于文档
生成有依据的回答。防幻觉的核心约束在 RAG_SYSTEM_PROMPT：严格基于
文档、不编造、找不到转人工、金融信息加风险提示、标注来源。

流水线四步，每步一个 trace span，process 为父 span，trace 树即流水线：
1. rewrite_query       口语问题 → 检索友好查询（补专业术语）
2. retrieve_documents  LongTermMemory.search 取 top 5（召回，求全）
3. rerank_documents    LLM 看查询与文档摘要选 top 3（精排，求准）
4. generate_answer     原始问题 + 3 篇文档 → 回答

注意检索用改写后的问题、生成用原始问题：生成要贴用户原话。
空检索有兜底：无文档时返回固定转人工话术，不调 LLM。

成本结构：知识路径单轮 4 次 LLM 调用（路由、改写、重排、生成各一），
延迟与费用主要来自这里；改写在降级检索下价值有限，接真实
embedding 后四步才各得其所。

rerank 的输出防护（三处，均为 LLM 索引输出的真实风险面）：
1. 边界条件 0 <= i < len(documents)：只查上界挡不住 Python
   负索引穿透，LLM 返回 -1 会从列表尾部取文档
2. 解析去重与截断：重复索引不重复取（dict.fromkeys 保序），
   结果截断到 top_k
3. process 按 delta 约定返回变更键
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from memory.long_term import LongTermMemory
from tracing.otel_config import trace_agent_call


RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答Agent，负责根据检索到的文档回答用户问题。

回答规则：
1. 严格基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确告知用户并建议转人工
3. 回答要简洁专业，适合客服场景
4. 对于金融产品信息，必须标注"以上信息仅供参考，具体以合同条款为准"
5. 在回答末尾标注引用的文档来源

回答格式：
- 先直接回答用户问题
- 如有必要补充相关信息
- 金融场景需添加风险提示
"""

QUERY_REWRITE_PROMPT = """请将用户的口语化问题改写为更适合向量检索的查询语句。
保留核心语义，去除口语化表达，补充专业术语。
只返回改写后的查询，不要其他内容。

用户原始问题: {query}
"""

# rerank 喂给 LLM 的摘要长度：截断展示用，排序判断足够
_RERANK_SUMMARY_CHARS = 200


class KnowledgeRAGAgent:
    """知识检索 Agent：RAG 四步流水线（结构见模块 docstring）"""

    def __init__(self, llm: ChatOpenAI, long_term_memory: LongTermMemory | None = None):
        self.llm = llm
        self.long_term_memory = long_term_memory or LongTermMemory()

    @trace_agent_call("rag_query_rewrite")
    async def rewrite_query(self, original_query: str) -> str:
        """Query 改写：口语化问题转为检索友好的查询"""
        messages = [
            HumanMessage(content=QUERY_REWRITE_PROMPT.format(query=original_query)),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content.strip()

    @trace_agent_call("rag_retrieve")
    async def retrieve_documents(self, query: str, top_k: int = 5) -> list[dict]:
        """从长期记忆检索 top_k 篇文档（召回阶段）。

        2.2 起检索走真实 embedding（网络 IO），补 await；
        1.6 降级版时代此方法无 IO，async 仅为接口统一。
        """
        return await self.long_term_memory.search(query, top_k=top_k)

    @trace_agent_call("rag_rerank")
    async def rerank_documents(
        self, query: str, documents: list[dict], top_k: int = 3
    ) -> list[dict]:
        """LLM 对候选文档做相关性精排，取 top_k。"""
        if not documents:
            return []

        doc_summaries = "\n".join(
            f"[{i}] {doc.get('content', '')[:_RERANK_SUMMARY_CHARS]}"
            for i, doc in enumerate(documents)
        )

        messages = [
            SystemMessage(content="你是一个文档相关性排序专家。"),
            HumanMessage(content=(
                f"用户查询: {query}\n\n"
                f"候选文档:\n{doc_summaries}\n\n"
                f"请返回最相关的{top_k}个文档的索引号，用逗号分隔，如: 0,2,4"
            )),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            indices = [int(i.strip()) for i in response.content.split(",")]
        except ValueError:
            # 解析失败回落：按检索排序取前 top_k，检索得分仍有参考价值
            return documents[:top_k]

        # 两处防护：
        # 1) 边界 0 <= i < len(documents)：只查上界挡不住负索引穿透
        # 2) dict.fromkeys 去重保序 + 截断 top_k：重复索引不重复取
        unique_valid = dict.fromkeys(i for i in indices if 0 <= i < len(documents))
        return [documents[i] for i in unique_valid][:top_k]

    @trace_agent_call("rag_generate")
    async def generate_answer(self, query: str, context_docs: list[dict]) -> str:
        """基于检索文档生成回答（用原始问题，见模块 docstring）。"""
        if not context_docs:
            # 空检索兜底：固定话术，不调 LLM
            return "抱歉，知识库中暂未找到与您问题相关的信息。建议您联系人工客服获取帮助。"

        context = "\n\n---\n\n".join(
            f"来源: {doc.get('source', '未知')}\n内容: {doc.get('content', '')}"
            for doc in context_docs
        )

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题: {query}\n\n"
                f"检索到的参考文档:\n{context}"
            )),
        ]

        response = await self.llm.ainvoke(messages)
        return response.content

    @trace_agent_call("knowledge_rag_process")
    async def process(self, state: AgentState) -> dict[str, Any]:
        """作为 Graph 节点处理状态：串联四步流水线。"""
        messages = state.get("messages", [])
        if not messages:
            return {}

        # 隐含假设：最后一条消息是本轮用户消息。
        # 成立前提是 synthesize 在所有业务节点之后才写入回复（当前拓扑
        # 如此）；若图结构变化（插入新的消息写入点），这里会取错
        original_query = messages[-1].content

        # 检索用改写后的查询，生成用原始问题（理由见模块 docstring）
        rewritten_query = await self.rewrite_query(original_query)

        raw_docs = await self.retrieve_documents(rewritten_query, top_k=5)

        reranked_docs = await self.rerank_documents(rewritten_query, raw_docs, top_k=3)

        answer = await self.generate_answer(original_query, reranked_docs)

        return {
            "sub_results": {
                **state.get("sub_results", {}),
                "knowledge_rag": answer,
            },
        }
