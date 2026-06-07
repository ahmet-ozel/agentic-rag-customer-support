# AgentDesk: Agentic RAG Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-224%20passing-brightgreen)]()

A production-grade **Agentic RAG** (Retrieval-Augmented Generation) platform for customer support automation. A reference implementation that brings together LLMs, MCP servers, vector databases and a document pipeline in a config-driven architecture.

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Supported Providers](#supported-providers)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ Intent Router│───▶│         Agent Loop            │  │
│  │  (TF-IDF)    │    │  ┌────────┐  ┌────────────┐  │  │
│  └──────────────┘    │  │  LLM   │◀▶│ MCP Manager│  │  │
│                      │  │ Client │  │            │  │  │
│  ┌──────────────┐    │  └────────┘  └─────┬──────┘  │  │
│  │Session Mgr   │    └─────────────────────┼────────┘  │
│  └──────────────┘                          │            │
│                                            ▼            │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │Reference     │    │         MCP Servers           │  │
│  │Store (TTL)   │    │  postgres-mcp │ qdrant-mcp   │  │
│  └──────────────┘    │  docling-mcp  │ paddleocr-mcp│  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PostgreSQL  │    │   Qdrant    │    │  vLLM / API │
│  (Customer   │    │  (Vector    │    │  (LLM       │
│   Database)  │    │   Store)    │    │   Backend)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Design Decisions

- **Unified LLM Interface:** All LLM providers (vLLM, OpenAI, Anthropic, Google, Ollama) are accessed through a single OpenAI-compatible client
- **MCP Protocol:** All external I/O is managed through MCP servers - the agent loop stays pure
- **Reference Store:** Large tool results are kept in a TTL-based store to prevent token overflow
- **Config-Driven:** All providers can be swapped via `config.yaml` with no code changes
- **Tiered LLM:** A cheap model is used for routing and a strong model for response generation, optimizing cost

---

## Features

- **Flexible LLM Backend** - vLLM (local), OpenAI, Anthropic, Google, Ollama; tiered routing for cost optimization
- **Multiple Vector Stores** - Qdrant (default), extensible to Milvus, Chroma, pgvector
- **MCP Server Management** - stdio and SSE transport, auto-restart, health monitoring
- **Document Pipeline** - upload → parse → chunk → embed → store, with configurable chunking strategies
- **Intent Routing** - TF-IDF semantic classification; chitchat bypasses the agent loop
- **Session Management** - in-memory conversation history with TTL and message limits
- **Reference Store** - large tool results stored under a `ref_xxx` key to prevent context overflow
- **Gradio UI** - chat interface, document upload, MCP status and statistics panels
- **Full Observability** - structured JSON logs for conversations, token usage and tool calls
- **Multi-Database Support** - PostgreSQL (self-hosted), Neon (serverless), Supabase (BaaS)

---

## Quick Start

### Requirements

- Python 3.11+
- Docker & Docker Compose
- (Optional) NVIDIA GPU - for local vLLM

### 1. Clone and configure

```bash
git clone https://github.com/ahmet-ozel/agentic-rag-customer-support.git
cd agentic-rag-customer-support
cp .env.example .env
# Edit .env with your API keys and database passwords
```

### 2. Start the services

```bash
# CPU / cloud LLM mode
docker compose up -d

# GPU mode (includes vLLM)
docker compose --profile gpu up -d
```

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### 4. Open the UI

Go to `http://localhost:7860` for the Gradio interface, or use the REST API directly.

---

## Installation

### Local Development Environment

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env

# Start the server (Qdrant + PostgreSQL must be running)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Start the Gradio UI
python frontend/gradio_app.py
```

### Makefile Commands

```bash
make help          # List all commands
make dev           # Start the FastAPI server with hot-reload
make ui            # Start the Gradio UI
make test          # Run all tests
make test-unit     # Unit tests only
make test-int      # Integration tests only
make lint          # Run the Ruff linter
make docker-up     # Start all services (CPU mode)
make docker-gpu    # Start all services including vLLM (GPU mode)
make docker-down   # Stop all services
make clean         # Clean __pycache__ and .pytest_cache
```

---

## Configuration

All behavior is controlled from `config.yaml`. Secrets are loaded from `.env` via `${ENV_VAR}` placeholders.

### LLM Configuration

```yaml
llm:
  default_provider: openai    # vllm | openai | anthropic | google | ollama

  providers:
    openai:
      base_url: https://api.openai.com/v1
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
      max_tokens: 4096
      temperature: 0.7
      timeout: 60

    vllm:
      base_url: ${VLLM_BASE_URL}
      model: Qwen/Qwen3-32B

  tiered:
    enabled: true
    routing_provider: openai
    routing_model: gpt-4o-mini    # Cheap model for routing
```

### Database Configuration

AgentDesk supports three different database providers. Select the active one via `config.yaml` → `database.db_provider`.

#### Option 1: PostgreSQL (Self-Hosted / Docker)

Started automatically via Docker Compose. No extra configuration needed.

```yaml
# config.yaml
database:
  db_provider: postgresql
  providers:
    postgresql:
      host: ${POSTGRES_HOST}
      port: ${POSTGRES_PORT}
      database: ${POSTGRES_DB}
      user: ${POSTGRES_READONLY_USER}
      password: ${POSTGRES_READONLY_PASSWORD}
      ssl_mode: prefer
```

```bash
# .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=agentdesk
POSTGRES_READONLY_USER=agentdesk_readonly
POSTGRES_READONLY_PASSWORD=secure_password
DB_PASSWORD=changeme
```

```bash
# Start services (PostgreSQL + Qdrant + AgentDesk)
docker compose up -d
```

#### Option 2: Neon (Serverless, Free Plan)

[Neon](https://neon.tech) is a serverless PostgreSQL service. Its free plan offers 0.5 GB storage and auto-scaling.

Setup steps:

1. Create an account at [neon.tech](https://neon.tech)
2. Create a new project
3. Dashboard → Connection Details → copy the connection string
4. Fill in `.env`:

```bash
# .env
NEON_DATABASE_URL=postgresql://neondb_owner:abc123@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
NEON_API_KEY=napi_abc123...
```

5. Update `config.yaml`:

```yaml
# config.yaml
database:
  db_provider: neon
  providers:
    neon:
      connection_string: ${NEON_DATABASE_URL}
      ssl_mode: require

# Update MCP servers  -  enable neon_mcp, disable postgres_mcp
mcp_servers:
  postgres_mcp:
    enabled: false
    # ...
  neon_mcp:
    enabled: true
    transport: stdio
    command: npx
    args: ["-y", "@neondatabase/mcp-server-neon", "start"]
    env:
      NEON_API_KEY: ${NEON_API_KEY}
```

6. Create the database schema (via the Neon SQL Editor or psql):

```bash
psql "${NEON_DATABASE_URL}" -f db/migrations/001_initial_schema.sql
psql "${NEON_DATABASE_URL}" -f db/seed/seed_data.sql
```

#### Option 3: Supabase (BaaS, Free Plan)

[Supabase](https://supabase.com) is an open-source Firebase alternative. Its free plan offers a 500 MB database, unlimited API requests and built-in auth.

Setup steps:

1. Create an account at [supabase.com](https://supabase.com)
2. Create a new project (choose a region, set a database password)
3. Collect the required values:
   - Project Settings → Database → Connection string (Transaction pooler) → `SUPABASE_DATABASE_URL`
   - Project Settings → API → `service_role` key → `SUPABASE_ACCESS_TOKEN`
   - Project Settings → General → Reference ID → `SUPABASE_PROJECT_REF`
4. Fill in `.env`:

```bash
# .env
SUPABASE_DATABASE_URL=postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
SUPABASE_ACCESS_TOKEN=sbp_abc123...
SUPABASE_PROJECT_REF=abcdefghijklmnop
```

5. Update `config.yaml`:

```yaml
# config.yaml
database:
  db_provider: supabase
  providers:
    supabase:
      connection_string: ${SUPABASE_DATABASE_URL}
      access_token: ${SUPABASE_ACCESS_TOKEN}
      project_ref: ${SUPABASE_PROJECT_REF}
      ssl_mode: require

# Update MCP servers  -  enable supabase_mcp, disable postgres_mcp
mcp_servers:
  postgres_mcp:
    enabled: false
    # ...
  supabase_mcp:
    enabled: true
    transport: stdio
    command: npx
    args: ["-y", "@supabase/mcp-server-supabase@latest"]
    env:
      SUPABASE_ACCESS_TOKEN: ${SUPABASE_ACCESS_TOKEN}
```

6. Create the database schema (via the Supabase SQL Editor or psql):

```bash
psql "${SUPABASE_DATABASE_URL}" -f db/migrations/001_initial_schema.sql
psql "${SUPABASE_DATABASE_URL}" -f db/seed/seed_data.sql
```

#### Database Provider Comparison

| Feature | PostgreSQL | Neon | Supabase |
|---------|-----------|------|----------|
| Type | Self-hosted | Serverless | BaaS |
| Free Plan | Via Docker | 0.5 GB | 500 MB |
| Auto-scaling | No | Yes | Yes |
| Zero Cold Start | Yes | ~0.5s | Yes |
| Built-in Auth | No | No | Yes |
| MCP Server | postgres_mcp | neon_mcp | supabase_mcp |
| Setup Difficulty | Easy (Docker) | Easy | Easy |
| Production Ready | Yes | Yes | Yes |

### Vector Store and Embedding

```yaml
vector_store:
  provider: qdrant
  host: localhost
  port: 6333
  collection_name: documents
  hybrid_search: false
  reranker:
    enabled: false
    model: cross-encoder/ms-marco-MiniLM-L-6-v2

embedding:
  model: bge-m3
  provider: local    # local | openai | voyage | cohere
  dimension: 1024

chunking:
  strategy: recursive    # recursive | semantic | document_aware
  chunk_size: 512
  chunk_overlap: 50
```

See [`config.yaml`](config.yaml) for all options.

---

## API Reference

| Method | Endpoint | Description |
|-------|----------|----------|
| `GET` | `/health` | Health check |
| `GET` | `/info` | System info (version, active providers) |
| `POST` | `/api/v1/chat` | Synchronous chat |
| `WS` | `/api/v1/chat/stream` | Streaming chat (WebSocket) |
| `POST` | `/api/v1/documents` | Upload and process a document |
| `GET` | `/api/v1/documents` | List documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `GET` | `/api/v1/customers` | Query customer info |
| `GET` | `/api/v1/config` | View configuration |
| `PUT` | `/api/v1/config` | Update configuration at runtime |
| `GET` | `/api/v1/mcp/status` | MCP server statuses |
| `GET` | `/api/v1/stats` | Token usage and statistics |

Interactive API docs: `http://localhost:8000/docs`

### Example Requests

#### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the subscription status of customer #1?"}'
```

#### Document Upload

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@user_manual.pdf"
```

#### System Statistics

```bash
curl http://localhost:8000/api/v1/stats
```

---

## Project Structure

```
agentdesk/
├── src/
│   ├── agent/          # AgentLoop  -  iterative LLM ↔ MCP tool-call loop
│   ├── api/            # FastAPI routers (chat, documents, customers, config, stats)
│   ├── chunking/       # ChunkingEngine (recursive, semantic, document_aware)
│   ├── config/         # ConfigManager + Pydantic models
│   ├── llm/            # LLMClient  -  unified OpenAI-compatible interface
│   ├── mcp/            # MCPManager  -  stdio/SSE lifecycle management
│   ├── models/         # API request/response schemas
│   ├── router/         # IntentRouter  -  TF-IDF semantic classification
│   ├── session/        # SessionManager  -  conversation history
│   ├── store/          # ReferenceStore  -  TTL-based memory store
│   ├── vectorstore/    # VectorStoreAdapter + QdrantVectorStore
│   ├── logging_config.py  # Structured logging
│   └── main.py         # FastAPI application entry point
├── frontend/
│   └── gradio_app.py   # Gradio chat interface
├── db/
│   ├── migrations/     # PostgreSQL schema migrations
│   └── seed/           # Sample data
├── mcp_servers/
│   └── postgres_mcp/   # Read-only PostgreSQL MCP server configuration
├── tests/
│   ├── unit/           # Unit tests (224 tests)
│   ├── property/       # Property-based tests (Hypothesis)
│   └── integration/    # End-to-end flow tests
├── config.yaml         # Main configuration file
├── docker-compose.yml  # Multi-service orchestration
├── Dockerfile          # Multi-stage build
├── Makefile            # Development commands
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

---

## Development

### Requirements

- Python 3.11+
- Docker & Docker Compose (for services)
- Git

### Environment Setup

```bash
# Clone the repo
git clone https://github.com/ahmet-ozel/agentic-rag-customer-support.git
cd agentic-rag-customer-support

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Environment variables
cp .env.example .env
```

### Code Style

The project uses the [Ruff](https://docs.astral.sh/ruff/) linter:

```bash
make lint
# or
ruff check src/ tests/
```

---

## Testing

The project has a comprehensive test suite:

```bash
# Run all tests
make test
# or
pytest tests/ -v

# Unit tests only
make test-unit

# Integration tests only
make test-int
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Isolated tests for each component
  - `test_database_config.py` - Database provider configuration (PostgreSQL, Neon, Supabase)
  - `test_config.py` - Configuration loading and validation
  - `test_llm_client.py` - LLM client functionality
  - `test_mcp_manager.py` - MCP server management
  - `test_intent_router.py` - Intent classification
  - `test_agent_loop.py` - Agent loop workflow
  - `test_reference_store.py` - Reference store operations
  - `test_session_manager.py` - Session management
  - `test_chunking.py` - Document chunking
  - `test_vectorstore.py` - Vector store operations
  - `test_api_endpoints.py` - REST API endpoints
  - `test_schemas.py` - API schema validation
  - `test_logging_config.py` - Logging setup

- **Integration Tests** (`tests/integration/`): End-to-end flow tests
  - Chat flow (Intent Router → Agent Loop → LLM → MCP → Response)
  - Document processing pipeline
  - Customer query flow

- **Property Tests** (`tests/property/`): Correctness properties with Hypothesis

### Test Coverage

| Module | Test Count | Status |
|-------|-------------|-------|
| Unit Tests | 216 | Yes Passing |
| Integration Tests | 8 | Yes Passing |
| **Total** | **224** | **Yes All Passing** |

---

## Docker Deployment

### Services

| Service | Port | Description |
|--------|------|----------|
| `agentdesk` | 8000 | FastAPI main server |
| `postgres` | 5432 | PostgreSQL customer database |
| `qdrant` | 6333, 6334 | Qdrant vector database |
| `vllm` (GPU) | 8080 | vLLM local inference server |

### Startup

```bash
# CPU mode (uses cloud LLM)
docker compose up -d

# GPU mode (includes local vLLM)
docker compose --profile gpu up -d

# Follow logs
docker compose logs -f agentdesk

# Stop
docker compose down
```

### Persistent Data

Data is preserved via Docker volumes:
- `postgres_data` - PostgreSQL database data
- `qdrant_data` - Qdrant vector indexes

---

## Supported Providers

| Category | Options |
|----------|------------|
| **LLM** | vLLM, OpenAI, Anthropic, Google, Ollama |
| **Vector Store** | Qdrant (default), extensible to Milvus, Chroma, pgvector |
| **Embedding** | bge-m3 (local), OpenAI, Voyage, Cohere |
| **Document Parser** | docling-mcp, paddleocr-mcp (via MCP configuration) |
| **Customer DB** | PostgreSQL, Neon, Supabase |

---

## Environment Variables

| Variable | Description | Required |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | If LLM provider=openai |
| `ANTHROPIC_API_KEY` | Anthropic API key | If LLM provider=anthropic |
| `GOOGLE_API_KEY` | Google API key | If LLM provider=google |
| `VLLM_BASE_URL` | vLLM server URL | If LLM provider=vllm |
| `POSTGRES_HOST` | PostgreSQL host | If db_provider=postgresql |
| `POSTGRES_PORT` | PostgreSQL port | If db_provider=postgresql |
| `POSTGRES_DB` | Database name | If db_provider=postgresql |
| `POSTGRES_READONLY_USER` | Read-only user | If db_provider=postgresql |
| `POSTGRES_READONLY_PASSWORD` | User password | If db_provider=postgresql |
| `DB_PASSWORD` | PostgreSQL admin password | For Docker Compose |
| `NEON_DATABASE_URL` | Neon connection string | If db_provider=neon |
| `NEON_API_KEY` | Neon API key | If db_provider=neon |
| `SUPABASE_DATABASE_URL` | Supabase connection string | If db_provider=supabase |
| `SUPABASE_ACCESS_TOKEN` | Supabase access token | If db_provider=supabase |
| `SUPABASE_PROJECT_REF` | Supabase project reference ID | If db_provider=supabase |

See [`.env.example`](.env.example) for all variables.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the [MIT License](LICENSE).