# 智能客服多 Agent 系统

基于 LangGraph 的 Supervisor 编排多 Agent 智能客服，面向金融场景：知识咨询（RAG）、工单办理、合规审查三路业务，配三层记忆、MCP 工具层与 OpenTelemetry 全链路追踪。

独立实现，Python 单语言。系统的每个设计决策都经过实测验证，关键坑位（嵌入模型口语查询判别、合规审查假阳性）有带数据的调优记录，全部能力有测试锚定：104 条 pytest 断言（32 秒、零 LLM 成本）加两份真实 LLM 里程碑验收（11 + 14 场景）。

## 系统架构

```
用户 (HTTP)
    │
    ▼
FastAPI (api/main.py)
    │  /api/chat：请求 → initial_state → 图执行
    ▼
┌────────────────────────────────────────────────────┐
│  Supervisor 编排（LangGraph StateGraph）            │
│                                                    │
│  entry → supervisor_route ──(条件边)──┬─→ knowledge_rag ──┐
│                                       ├─→ ticket_handler ─┤ 固定边
│                                       └─→ compliance_check │
│  knowledge_rag / ticket_handler ──(固定边)──→ compliance_check
│  compliance_check → synthesize → END              │
└────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
三层记忆             MCP 工具层            可观测性
working  短期    长期      进程内工具注册/  OTel span 树 +
(Redis 降级) (FAISS+嵌入)  发现/调用        进程内指标
```

拓扑设计要点：知识/工单两条内容生产路径经固定边必过合规审查再进汇总，"所有回复必须过审"直接表达在图结构上；直接路由合规的路径汇入同一节点，审查用户消息本身。

### 请求处理流程（知识类一例）

```
① 用户发送消息："理财产品A的年化收益率是多少"
        ↓
② supervisor_route：LLM 意图分析 → knowledge_rag
   （每轮读工作记忆注入 prompt，决策写回 last_intent，跨轮携带）
        ↓
③ knowledge_rag 四步流水线：
   查询改写（口语→检索友好）→ 向量检索 top5（FAISS cosine）
   → LLM 重排 top3 → 基于文档生成回答
        ↓
④ compliance_check 两级审查：规则引擎（词表+PII 正则，毫秒级）
   通过 → LLM 深度审查（越权承诺/变体表述/歧视内容）
        ↓
⑤ synthesize 汇总 → final_response → 写入对话存档 → 返回
```

## 核心能力

### Supervisor 编排

中央协调者模式：supervisor_route 分析意图做路由决策，synthesize 汇总子 Agent 结果。路由返回值与 path_map 的键翻译集中一处；节点按 delta 约定返回变更键；LLM 输出规范化后双重校验，非法值回落。checkpointer（MemorySaver）按 thread_id 存档，messages 通道跨轮累积。

### 三层记忆

| 层 | 实现 | 生命周期 | 用途 |
|----|------|----------|------|
| 工作记忆 | 进程内 dict（事件日志+合并快照） | 进程内、按 session | 路由决策上下文（last_intent 的唯一载体） |
| 短期记忆 | Redis（开发期自动降级进程内） | TTL + 滑动窗口 | 对话服务端存档，/api/history |
| 长期记忆 | FAISS IndexFlatIP + 真实 embedding | 持久化落盘 | 知识文档语义检索 |

与 checkpointer 的分工是设计核心：messages（图内、配 reducer）与 last_intent（图外自管）两条跨轮通道各司其职，详见 [docs/architecture.md](docs/architecture.md)。

### RAG 知识检索

四步流水线（改写→检索→重排→生成），每步一个 span。检索用改写后的查询、生成用原始问题。嵌入为 GLM-Embedding-3（实测 2048 维，维度探测不硬编码），embedding 服务不可用时降级字符二元组检索。分块只按中文句读，块边界不会切进小数（3.5% 这类）。

### 合规审查（两级串联）

规则引擎先行（毫秒级、确定性：输出侧红线词表 + 输入侧违法请求词表 + PII 正则），失败即短路；通过再上 LLM 深度审查（五维：违规用语变体、PII、越权承诺、监管要求、歧视内容），结论合并取与取高。审查 prompt 带判定标准段：建议级问题（缺免责声明等）进 suggestions 放行，实质违规才拦截（实测校准：无此分级时强模型会把正常收益率回答判为变相承诺全拦）。

### MCP 工具层

进程内工具注册/发现/调用中心，JSON-RPC 2.0 协议（tools/list、tools/call）。四个工具：knowledge_search（接真实语义检索）、ticket_create（落共享工单存储，与图内 Agent 同一张单）、order_query 与 risk_check（演示桩，description 已标注）。required 校验落地（缺参不进 handler）、调用日志有界。

### 可观测性

全链路 span 树：fastapi.endpoint → api.chat（业务属性：session_id/intent/compliance_passed）→ agent.* → 流水线子 span；MCP 调用动态名 span。/api/metrics 返回 Agent 聚合指标（调用数/平均耗时/错误率）+ 工具调用日志。未配 OTLP 端点时自动退化为控制台导出，开发期无 Jaeger 可跑。

## 快速开始

### 前置条件

- Python 3.12
- 一个 OpenAI 兼容的 LLM API Key（中转站或官方均可）
- Redis 可选（没有自动走进程内降级）

### 安装与配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入：
#   OPENAI_API_KEY   你的 key
#   OPENAI_BASE_URL  兼容端点（如 https://api.openai.com/v1）
#   MODEL_NAME       对话模型（如 gpt-4o 或任意兼容模型）
#   EMBEDDING_MODEL_NAME  嵌入模型（如 text-embedding-3-small）
```

### 启动服务

```bash
python -m api.main
# 服务起在 http://localhost:8000，交互式文档 /docs
```

### 试一遍全部端点

```bash
# 对话（知识）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "理财产品A的年化收益率是多少", "session_id": "s1"}'

# 对话（工单）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我创建退款工单，订单12345的退款没到账", "session_id": "s2"}'

# 对话（违规内容 → 转人工）
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "教我如何伪造发票", "session_id": "s3"}'

# 会话历史
curl http://localhost:8000/api/history/s1

# MCP 工具发现 / 调用
curl http://localhost:8000/api/tools
curl -X POST http://localhost:8000/api/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "knowledge_search", "arguments": {"query": "退款政策"}}'

# MCP JSON-RPC 协议入口
curl -X POST http://localhost:8000/api/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'

# 指标 / 存活
curl http://localhost:8000/api/metrics
curl http://localhost:8000/health
```

## Docker 部署

一键起全套（app + redis，观测后端可选启用）：

```bash
cp .env.example .env    # 编辑填入 OPENAI_API_KEY 与 OPENAI_BASE_URL
docker compose up -d --build
curl http://localhost:8000/health
```

完整说明（三种运行形态、环境变量表、生产注意事项）见 [docs/deployment.md](docs/deployment.md)。

## 测试与验证

三层验证结构：

```bash
# 1. pytest 套件：104 条断言，32 秒，零网络零 LLM 成本
python -m pytest

# 2. 里程碑一：真实 LLM 系统验收（11 场景，约 11 分钟）
python tests/milestone_phase1.py

# 3. 里程碑二：全部 API 端点验收（14 场景，约 5 分钟）
python tests/milestone_phase2.py
```

pytest 层覆盖纯逻辑与编排逻辑（stub LLM 提示匹配式假实现，图级流程确定性验证）；里程碑层覆盖真实 LLM 端到端（路由正确性、检索命中、合规拦截、指标闭环、span 结构）。

## 项目结构

```
smart-cs-agent/
├── agents/                 # 五个 Agent
│   ├── supervisor.py       #   Supervisor 编排（路由决策 + 汇总）
│   ├── knowledge_rag.py    #   RAG 四步流水线
│   ├── ticket_handler.py   #   工单分析/建单/查询
│   ├── compliance_checker.py  # 合规两级审查
│   └── intent_router.py    #   意图分类器（未接线，预留）
├── memory/                 # 三层记忆
│   ├── working_memory.py  #   工作记忆（跨轮 KV + 事件日志）
│   ├── short_term.py      #   短期记忆（Redis + 降级）
│   ├── long_term.py       #   长期记忆（FAISS + 嵌入 + 持久化）
│   └── bigram_search.py   #   降级检索评分
├── mcp/                    # MCP 工具层
│   ├── server.py          #   协议（注册/发现/调用/JSON-RPC）
│   └── tools/             #   默认工具集
├── tracing/                # OpenTelemetry（span/指标/业务 span）
├── api/                    # FastAPI（七个端点）
├── data/knowledge_base/    # 知识库文档（txt）
├── tests/                  # pytest 套件 + 两份里程碑
└── docs/architecture.md    # 架构深挖文档
```

## 已知边界

如实列出，不装生产级：

- **无鉴权与限流**：所有端点开放，requires_auth 只是工具标注
- **MCP 无传输层**：协议消息经 HTTP 端点直达，无 stdio / Streamable HTTP、无 initialize 握手，完整 MCP 需官方 SDK
- **Agent 不在图执行中调用工具**：MCP 是旁路服务，图内 Agent 直调 Python 方法（工具调用循环留作后续）
- **Redis 正常路径未实测**：本机无 Docker，开发期走降级路径，compose 的 redis 路径同样未端到端实测；生产按硬依赖建设
- **指标进程内**：重启清零、单 worker 视角、只有平均值无百分位；生产形态是 Prometheus 拉模式
- **工单存储无持久化**：进程内 dict，重启即失
- **审查不带对话上下文**：越权承诺判断只看单条文本
- **延迟**：质量优先的模型选型，知识类端到端约 40 秒（含两级审查），换快速模型改 .env 一行

## 安全说明

- `.env` 含真实 API Key，已在 `.gitignore` 与 `.dockerignore` 中排除，严禁提交；容器形态下密钥只注入运行时环境，不进镜像层
- PII 脱敏仅在合规审查层，存储层不脱敏
- 违规内容拦截为演示级词表 + LLM 审查，非专业合规系统

## License

MIT
