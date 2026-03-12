# =============================================================================
# AgentDesk RAG Platform — Multi-Stage Dockerfile
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — bağımlılıkları yükle
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime — minimal çalışma ortamı
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (curl sağlık kontrolü için)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Builder aşamasından yüklü paketleri kopyala
COPY --from=builder /install /usr/local

# Kaynak kodu kopyala
COPY src/ ./src/
COPY config.yaml .
COPY mcp_servers/ ./mcp_servers/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
