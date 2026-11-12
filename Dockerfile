# 智能客服多 Agent 系统的服务镜像
# 设计说明与环境变量表见 docs/deployment.md

FROM python:3.12-slim

# Python 输出不经缓冲，日志实时进 docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 依赖层与代码层分离：requirements 不变时命中构建缓存，改代码不重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码（.dockerignore 已裁掉密钥、本地索引与缓存）
COPY . .

# 非 root 运行。vector_store 显式建目录并授权：
# compose 把命名卷挂到这个路径，目录必须在镜像内存在且属主正确，
# 命名卷首次挂载才会以该属主初始化，否则应用无写权限
RUN useradd --create-home appuser \
    && mkdir -p /app/vector_store \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 生产启动形式。api/main.py 的 __main__ 块是开发入口（带热重载），
# 容器不走它，由 uvicorn 命令行直接启动
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
