# 架构说明

本文是设计层的深挖文档：每个关键决策的为什么。功能与用法见 [README](../README.md)。

## 1. StateGraph 的状态语义

状态契约 `AgentState`（agents/state.py）八字段，只有 `messages` 带 `add_messages` reducer，其余全是 LastValue 通道。这决定了整个系统的信息流规则：

- **messages 配 reducer**：按消息 ID 合并、累积，配合 checkpointer 按 thread 存档，对话历史因此跨轮存活。全系统只有两个写入点：端点写用户消息、synthesize 写最终回复
- **其余字段后写覆盖**：端点每轮传入的 initial_state 直接覆盖旧值，intent / sub_results 只反映当轮

由此推出跨轮存活的通道只有两条：messages（checkpointer）与 working_memory（图外自管）。任何"想跨轮传递"的信息都必须落进这两条通道之一，这是本系统最重要的结构约束。

**条件边的键翻译**：`path_map` 以路由函数返回值为键查表。路由函数返回合法意图名，意图名到节点名的翻译（compliance_checker → compliance_check）集中在 path_map 一处。路由函数直接返回节点名而 path_map 以意图名为键，两者错位会在条件边处 KeyError。

**delta 约定**：节点只返回变更键，未提到的字段保持原值。全量回写 messages 会触发按 ID 替换，净效果为零纯开销。

## 2. 三层记忆与 checkpointer 的四通道分工

四个存储组件，四个消费者，边界如下：

| 通道 | 数据 | 消费者 | 生命周期 |
|------|------|--------|----------|
| messages (checkpointer) | 完整对话（含回复） | 模型上下文（路由 prompt、生成 prompt） | thread 级、进程内 |
| working_memory | 路由决策等推理状态 | supervisor 每轮读注入 prompt | session 级、进程内 |
| short_term_memory | user/assistant 双写存档 | /api/history、前端 | TTL + 滑动窗口 |
| long_term_memory | 知识文档 + 向量 | RAG 检索后端 | 持久化落盘 |

**为什么 last_intent 不进 messages**：路由调用是独立的 LLM 调用，其输出走 AgentState 而非对话通道；塞进 messages 会污染对话历史（下一轮的路由 prompt 与生成 prompt 都会看到）。

**为什么 working_memory 不用 checkpointer 存**：checkpointer 按 thread 存 superstep 级全量快照，字段要跨轮存活需逐个配 reducer，且 state 会广播给全部节点。working_memory 任意键零合约写入即跨轮存活、引用只发给选定组件、节点执行中途的多次写入各自留痕，这些形态图机制覆盖不到。

**短期记忆与 checkpointer 为什么双写**：同一份对话、两个消费者。checkpointer 是图执行的内部机制（无查询接口），短期记忆是服务层资产（有 /api/history）。双写是消费者分离的代价，失败窗口行为"有问无答"符合事实。

## 3. RAG 流水线与实测发现

四步：改写 → 检索 top5 → LLM 重排 top3 → 生成。检索用改写后的查询，生成用原始问题（生成要贴用户原话）。

**实测发现一（嵌入判别）**：GLM-Embedding-3 对无词汇重合的短口语查询 top-1 判别弱。"钱放你们那能赚多少"的实测排序：开户指南 0.45 > 退款政策 0.43 > 理财文档 0.35，五种口语变体模式一致。排查确认非管线问题（query 与 document 两种嵌入入口向量 cos=1.0000，绕开管线直接复算复现排序）。BGE 式指令前缀可救回口语查询（0.49 升第一）但压塌精确查询区分度（"怎么开通账户"领先幅度从 0.13 缩到 0.006），无净收益未采用。

**结论**：嵌入层只承担召回，精度由上游的查询改写（口语转规范表达后检索）加 LLM 重排兜底，端到端口语用例验证命中。

**实测发现二（分块边界）**：按英文句点切分会让块边界落进小数（3.5% 被切成两块），本系统只按中文句读（。！？）与换行切，句子分支与段落分支共用 overlap，超长句硬切兜底。

**rerank 的输出防护**：LLM 返回索引列表，三个坑：负索引穿透（-1 从列表尾部取）、重复索引重复取、超量不截断。`dict.fromkeys` 保序去重加边界过滤加截断。

## 4. 合规两级审查与降级决策

规则引擎先行（毫秒级），失败即短路；通过再上 LLM 深度审查。

**级联位置决定降级设计**：走到 LLM 的前提是规则硬门已过（PII 与红线词的确定性拦截已完成）。LLM 层解析残余失败时按规则水位放行并显式留痕（violations 带降级标记 + 日志警告）。选择放行而非拦截的推理：假阳性拦截伤害确定发生且用户无法自救（温度 0 下重试也无解），漏检需要两个小概率相乘（解析失败 × 该内容恰好语义违规），且 LLM 审查本身即有假阴性率，属渐进损失组件。

**实测发现三（审查假阳性）**：审查标准不区分建议级与拦截级问题时，DeepSeek-V4-Pro 按监管口径从严执行，把"年化收益率为3.5%-5.2%"缺免责声明的正常知识回复判为变相承诺拦截（medium + passed=false），知识路径全挂。修复：prompt 增判定标准段，建议级问题（缺风险提示/免责声明/测算依据）进 suggestions 且放行，实质违规才拦截。

**LLM 输出防护清单**（合规层与意图路由、工单提取同模式）：

- 解析走共享 parse_llm_json（截首尾大括号，围栏与解释文字天然兼容；直解析在围栏输出时静默失败）
- passed 字段：字符串 "false" 是 truthy，直接判定会翻转拦截结论；无法解析时按违规证据推断
- 枚举字段（risk_level 等）：非法值回落，裸转换抛 ValueError
- 列表字段（violations/suggestions）：非列表丢弃、非字符串元素过滤

## 5. MCP 的定位与共享存储

MCP 是图之外的旁路服务：图内 Agent 直调 Python 方法，MCP 面向外部客户端（运维面板、其他服务）。完整接入（Agent 在图执行中经工具调用循环使用工具）需要 function calling 支持，留作后续。

**共享工单存储**：TicketStore 单例由 API lifespan 构造，同时注入图工厂与 MCP server。两处各建各的会形成两套互不相通的工单体系（图内建的单 MCP 查不到，反向亦然）。

**校验策略的分层**：图内 Agent 面对的是 LLM 输出，非法值回落默认保持可用；MCP 工具面对程序化调用方，schema 即契约，非法取值明确报错。合法集合复用同一份模块常量（单一事实源）。

## 6. 可观测性结构

span 树三层：fastapi.endpoint（框架请求周期，含序列化）→ api.chat（业务边界，含图执行）→ agent.*（各 Agent 与流水线子 span）。

**api.chat 为什么显式自建而非挂 fastapi 内置服务端 span**：挂内置依赖框架 instrumentation 的内部行为（是否把 span 设为当前激活、是否记录态），任一不成立则属性设置静默无操作，观测功能消失且无信号。显式自建每步自控，树层级与代码职责对齐。

**业务属性**：chat.session_id / chat.user_id / chat.intent / chat.compliance_passed 挂在 api.chat 上，trace 可按会话与路由结果检索。

**span 与指标的分工**：span 是单次调用的完整时间线（排查用），AgentMetrics 是跨调用聚合（监控用，/api/metrics）。工具调用不进 AgentMetrics（tool_call_log 已承载逐条记录，重复聚合无增量）。

## 7. 降级策略族

系统内四个降级点，同一哲学（可用性优先、降级显式可观测），各自适配组件性质：

| 组件 | 失败形态 | 降级行为 | 理由 |
|------|----------|----------|------|
| 短期记忆 | Redis 断线 | 进程内存储，逐操作重试重连 | 会话资产可容忍短暂分裂 |
| 长期检索 | 嵌入服务不可用 | 二元组重合度检索 | 语义质量回落但服务可用 |
| 工单入库 | 嵌入失败 | 抛异常零残留 | 无向量的文档是假入库 |
| 合规深审 | LLM 解析失败 | 按规则水位放行留标记 | 增强层失效降级到硬门水位 |

读写不对称的原则：读路径降级保可用，写路径失败保一致（宁可不写，不写坏的）。

## 8. 测试架构

三层金字塔：

1. **pytest（104 条，32 秒）**：纯逻辑单测（记忆、分块、评分、校验、协议）+ 图级编排（StubLLM 提示匹配式假 LLM，按收到的系统提示返回脚本响应，不依赖调用次序）+ HTTP 包装层（假图对象 monkeypatch 模块级 graph 全局，验证写透位置、thread_id 绑定、失败窗口语义）
2. **里程碑（真实 LLM，11+14 场景）**：路由正确性、检索命中、合规拦截、指标闭环（compliance 调用数 == 总轮次的拓扑断言）、span 结构
3. **真实运行**：人工验证

stub 与里程碑对同一行为的断言互为交叉验证：stub 锁契约（LLM 返回 X 时系统做 Y），里程碑锁真实行为（LLM 会不会返回 X）。2.4 的审查假阳性问题对 stub 不可见、被里程碑暴露，正是这个分层的价值。

**StubLLM 的设计**：提示匹配式而非调用次序式。按收到的系统提示关键词匹配返回对应脚本响应，图结构改动不会连带测试全崩。

## 9. 已知边界与后续路线

按影响排序：

1. **MCP 规范完整接入**：官方 SDK（FastMCP）、stdio/Streamable HTTP 传输、initialize 握手；图内工具调用循环（需 function calling）
2. **指标生产化**：Prometheus 导出器拉模式、直方图（p95/p99）、跨 worker 聚合
3. **短期记忆生产路线**：禁用降级、Redis 硬依赖（AOF everysec、主从高可用、noeviction）
4. **父文档检索**：小块定位（检索精度）、整篇交付（上下文完整），现有 source 元数据已支撑
5. **意图路由整合**：Supervisor 内部委托 classify（方案已定，每轮仍一次 LLM 调用）
6. **工单 update 流转与跨轮跟踪**：working memory 存 session 工单号，查询无号时兜底
7. **RAG 重排换专用模型**：GLM-Rerank（当前用对话 LLM，占知识流水线耗时一半）
