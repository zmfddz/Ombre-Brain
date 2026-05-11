# ============================================================
# Ombre Brain Docker Build
# Docker 构建文件
#
# Build: docker build -t ombre-brain .
# Run:   docker run -e OMBRE_API_KEY=your-key -p 8000:8000 ombre-brain
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (leverage Docker cache)
# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend code + config / 后端代码 + 配置
COPY *.py .
COPY config.yaml .

# Frontend assets (memory-garden) / 前端资源
COPY index.html .
COPY index_legacy.html .
COPY tweaks-panel.jsx .
COPY js/ ./js/

# Persistent mount point: bucket data from host
# 持久化挂载点：记忆数据从宿主机挂进来

# Default to streamable-http for container (remote access)
# 容器场景默认用 streamable-http
ENV OMBRE_TRANSPORT=streamable-http

EXPOSE 8000

CMD ["python", "server.py"]
