# 部署说明

系统有三种运行形态，按环境选择。架构与设计决策见 [architecture.md](architecture.md)，本地开发的最短路径见 README 的快速开始。

| 形态 | 适用场景 | 依赖 |
|------|----------|------|
| 本地裸跑 | 开发调试 | Python 3.12、LLM API Key；Redis 可选（无则自动降级进程内） |
| 单容器 | 已有 Redis 的环境 | Docker、外部 Redis |
| compose 编排 | 一键起全套 | Docker Compose（自带 Redis，观测后端可选启用） |

## 环境变量

应用从环境读取配置（本地开发时经 `.env` 加载，`.env.example` 是完整模板）：

| 变量 | 作用 | 默认值 | 必填 |
|------|------|--------|------|
| `OPENAI_API_KEY` | LLM 与嵌入模型的密钥 | 无 | 是 |
| `OPENAI_BASE_URL` | OpenAI 兼容端点地址 | 无 | 是 |
| `MODEL_NAME` | 对话模型名 | DeepSeek-V4-Pro-0813 | 建议 |
| `EMBEDDING_MODEL_NAME` | 嵌入模型名 | GLM-Embedding-3 | 否 |
| `REDIS_URL` | 短期记忆的 Redis 地址 | redis://localhost:6379/0 | 否 |
| `FAISS_INDEX_PATH` | 向量索引的落盘路径 | ./vector_store/faiss_index | 否 |
| `OTEL_SERVICE_NAME` | 链路追踪的服务名 | smart-cs-agent | 否 |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC 收集器地址，空则追踪打控制台 | 空 | 否 |
| `HOST` / `PORT` | 仅本地 `python -m api.main` 启动时生效 | 0.0.0.0 / 8000 | 否 |

## compose 部署（推荐）

```bash
# 1. 准备配置（.env 不入库，.dockerignore 同时保证它不进镜像层）
cp .env.example .env
#    编辑 .env 填入 OPENAI_API_KEY 与 OPENAI_BASE_URL

# 2. 构建并启动（app + redis 两个服务）
docker compose up -d --build

# 3. 验证（lifespan 里要先嵌入或加载知识文档，首次启动需等一会）
curl http://localhost:8000/health
#    {"status": "healthy", "version": "1.0.0"}
```

compose 内的关键行为：

- **密钥注入路径**：`.env` 经 `env_file` 进入容器运行时环境，镜像构建时被 `.dockerignore` 排除，密钥只存在于运行环境，不存在于镜像层与镜像仓库。
- **Redis 地址覆盖**：容器网络内经服务名寻址，compose 的 `environment` 用 `redis://redis:6379/0` 覆盖 `.env` 里的默认地址（compose 的 environment 优先级高于 env_file）。
- **索引持久化**：命名卷 `vector_store` 挂到 `/app/vector_store`。首次启动嵌入知识文档（一次真实 API 调用，有成本），此后索引落卷，容器重建后直接加载。
- **启动顺序**：`depends_on: condition: service_healthy`，app 等 redis 健康检查通过后才启动。
- **健康检查**：redis 用 `redis-cli ping`；app 打 `/health`（slim 镜像无 curl，用 Python 标准库实现）。

### 启用观测后端

```bash
# Jaeger 为可选服务，默认不启动
docker compose --profile observability up -d

# 在 .env 中设置导出目标（应用侧走 OTLP gRPC，对应 4317 端口）
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
# 然后重启 app 使环境变量生效
docker compose up -d

# UI
http://localhost:16686
```

## 生产注意事项

**Redis 淘汰策略**。compose 里配置为 `volatile-ttl` 加 256mb 上限。本系统的 Redis 键全部是带 TTL 的会话存档，写满时 volatile-ttl 优先清剩余存活时间最短（距上次活跃最久）的会话，保住活跃会话。避免 `allkeys-lru`：它从全部键里静默逐出，会话历史无感知丢失。容量按「峰值会话数 × 单会话消息条数上限 × 单条大小」估算，写满前扩容。

**单 worker 边界**。系统有三处进程内状态：working memory（图外推理状态）、MemorySaver checkpointer（图执行存档）、TicketStore（工单）。部署必须单进程单 worker（uvicorn 默认即是）。多 worker 或多副本时，同一会话的请求会被路由到不同进程，上下文与工单数据互相不可见。横向扩展的前置工作是状态外置：短期记忆已在 Redis（天然支持），checkpointer 换持久化后端，工单存储换数据库。

**未实测声明**。本仓库开发环境无 Docker，镜像构建与 compose 编排未做过端到端实测。交付前的验证是：干净 venv 完整跑一遍依赖安装与全量测试（模拟镜像的依赖层）、Dockerfile 的启动命令在本地等效执行并验证 `/health`、compose 的服务名、端口、环境变量名与代码逐项静态核对。首次真实构建若暴露问题（例如 faiss-cpu 在 slim 镜像的运行库依赖，预期 manylinux wheel 自带，未证实），按报错定位即可，文件本身保持最小改动成本。
