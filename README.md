# AgentDesk — Agentic RAG Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-224%20passing-brightgreen)]()

Müşteri destek otomasyonu için üretim kalitesinde bir **Agentic RAG** (Retrieval-Augmented Generation) platformu. LLM'ler, MCP sunucuları, vektör veritabanları ve doküman pipeline'ını yapılandırma odaklı (config-driven) bir mimaride birleştiren referans implementasyon.

---

## İçindekiler

- [Mimari](#mimari)
- [Özellikler](#özellikler)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [API Referansı](#api-referansı)
- [Proje Yapısı](#proje-yapısı)
- [Geliştirme](#geliştirme)
- [Test](#test)
- [Docker ile Dağıtım](#docker-ile-dağıtım)
- [Desteklenen Sağlayıcılar](#desteklenen-sağlayıcılar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

---

## Mimari

```
Kullanıcı İsteği
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Sunucusu                      │
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
│  │Reference     │    │         MCP Sunucuları        │  │
│  │Store (TTL)   │    │  postgres-mcp │ qdrant-mcp   │  │
│  └──────────────┘    │  docling-mcp  │ paddleocr-mcp│  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PostgreSQL  │    │   Qdrant    │    │  vLLM / API │
│  (Müşteri    │    │  (Vektör    │    │  (LLM       │
│   Veritabanı)│    │   Deposu)   │    │   Backend)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Temel Tasarım Kararları

- **Birleşik LLM Arayüzü:** Tüm LLM sağlayıcıları (vLLM, OpenAI, Anthropic, Google, Ollama) tek bir OpenAI-uyumlu istemci üzerinden erişilir
- **MCP Protokolü:** Tüm harici I/O işlemleri MCP sunucuları üzerinden yönetilir — agent loop saf kalır
- **Reference Store:** Büyük araç sonuçları TTL'li bir bellekte saklanarak token taşması önlenir
- **Config-Driven:** `config.yaml` üzerinden tüm sağlayıcılar kod değişikliği olmadan değiştirilebilir
- **Kademeli LLM:** Maliyet optimizasyonu için yönlendirmede ucuz, yanıt üretiminde güçlü model kullanılır

---

## Özellikler

- **Esnek LLM Backend** — vLLM (yerel), OpenAI, Anthropic, Google, Ollama; maliyet optimizasyonu için kademeli yönlendirme
- **Çoklu Vektör Deposu** — Qdrant (varsayılan), Milvus, Chroma, pgvector'e genişletilebilir
- **MCP Sunucu Yönetimi** — stdio ve SSE transport, otomatik yeniden başlatma, sağlık izleme
- **Doküman Pipeline** — yükleme → ayrıştırma → parçalama → gömme → depolama, yapılandırılabilir chunking stratejileri
- **Niyet Yönlendirme** — TF-IDF semantik sınıflandırma, chitchat agent loop'u atlar
- **Oturum Yönetimi** — bellekte konuşma geçmişi, TTL ve mesaj limiti
- **Reference Store** — büyük araç sonuçları `ref_xxx` koduyla saklanarak bağlam taşması önlenir
- **Gradio UI** — sohbet arayüzü, doküman yükleme, MCP durumu ve istatistik panelleri
- **Tam İzlenebilirlik** — konuşmalar, token kullanımı ve araç çağrıları için yapılandırılmış JSON logları
- **Çoklu Veritabanı Desteği** — PostgreSQL (self-hosted), Neon (serverless), Supabase (BaaS)

---

## Hızlı Başlangıç

### Gereksinimler

- Python 3.11+
- Docker & Docker Compose
- (Opsiyonel) NVIDIA GPU — yerel vLLM için

### 1. Klonla ve yapılandır

```bash
git clone https://github.com/ahmet-ozel/agentdesk.git
cd agentdesk
cp .env.example .env
# .env dosyasını API anahtarlarınız ve veritabanı şifrelerinizle düzenleyin
```

### 2. Servisleri başlat

```bash
# CPU / bulut LLM modu
docker compose up -d

# GPU modu (vLLM dahil)
docker compose --profile gpu up -d
```

### 3. Doğrula

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### 4. Arayüzü aç

Gradio arayüzü için `http://localhost:7860` adresine gidin veya REST API'yi doğrudan kullanın.

---

## Kurulum

### Yerel Geliştirme Ortamı

```bash
# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Ortam değişkenlerini ayarla
cp .env.example .env
# .env dosyasını düzenle

# Sunucuyu başlat (Qdrant + PostgreSQL çalışıyor olmalı)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Gradio UI'ı başlat
python frontend/gradio_app.py
```

### Makefile Komutları

```bash
make help          # Tüm komutları listele
make dev           # FastAPI sunucusunu hot-reload ile başlat
make ui            # Gradio UI'ı başlat
make test          # Tüm testleri çalıştır
make test-unit     # Sadece birim testleri
make test-int      # Sadece entegrasyon testleri
make lint          # Ruff linter çalıştır
make docker-up     # Tüm servisleri başlat (CPU modu)
make docker-gpu    # vLLM dahil tüm servisleri başlat (GPU modu)
make docker-down   # Tüm servisleri durdur
make clean         # __pycache__ ve .pytest_cache temizle
```

---

## Yapılandırma

Tüm davranış `config.yaml` dosyasından kontrol edilir. Gizli bilgiler `.env` dosyasından `${ENV_VAR}` yer tutucuları ile yüklenir.

### LLM Yapılandırması

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
    routing_model: gpt-4o-mini    # Yönlendirme için ucuz model
```

### Veritabanı Yapılandırması

AgentDesk üç farklı veritabanı sağlayıcısını destekler. `config.yaml` → `database.db_provider` ile aktif sağlayıcıyı seçin.

#### Seçenek 1: PostgreSQL (Self-Hosted / Docker)

Docker Compose ile otomatik başlatılır. Ek yapılandırma gerekmez.

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
POSTGRES_READONLY_PASSWORD=güvenli_şifre
DB_PASSWORD=changeme
```

```bash
# Servisleri başlat (PostgreSQL + Qdrant + AgentDesk)
docker compose up -d
```

#### Seçenek 2: Neon (Serverless — Ücretsiz Plan Mevcut)

[Neon](https://neon.tech), serverless PostgreSQL hizmetidir. Ücretsiz planı 0.5 GB depolama ve otomatik ölçeklendirme sunar.

Kurulum adımları:

1. [neon.tech](https://neon.tech) adresinden hesap oluşturun
2. Yeni bir proje oluşturun
3. Dashboard → Connection Details → Connection string'i kopyalayın
4. `.env` dosyasını doldurun:

```bash
# .env
NEON_DATABASE_URL=postgresql://neondb_owner:abc123@ep-cool-name-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
NEON_API_KEY=napi_abc123...
```

5. `config.yaml`'ı güncelleyin:

```yaml
# config.yaml
database:
  db_provider: neon
  providers:
    neon:
      connection_string: ${NEON_DATABASE_URL}
      ssl_mode: require

# MCP sunucularını güncelle — neon_mcp'yi etkinleştir, postgres_mcp'yi devre dışı bırak
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

6. Veritabanı şemasını oluşturun (Neon SQL Editor veya psql ile):

```bash
psql "${NEON_DATABASE_URL}" -f db/migrations/001_initial_schema.sql
psql "${NEON_DATABASE_URL}" -f db/seed/seed_data.sql
```

#### Seçenek 3: Supabase (BaaS — Ücretsiz Plan Mevcut)

[Supabase](https://supabase.com), açık kaynak Firebase alternatifidir. Ücretsiz planı 500 MB veritabanı, sınırsız API isteği ve dahili auth sunar.

Kurulum adımları:

1. [supabase.com](https://supabase.com) adresinden hesap oluşturun
2. Yeni bir proje oluşturun (bölge seçin, veritabanı şifresi belirleyin)
3. Gerekli bilgileri toplayın:
   - Project Settings → Database → Connection string (Transaction pooler) → `SUPABASE_DATABASE_URL`
   - Project Settings → API → `service_role` key → `SUPABASE_ACCESS_TOKEN`
   - Project Settings → General → Reference ID → `SUPABASE_PROJECT_REF`
4. `.env` dosyasını doldurun:

```bash
# .env
SUPABASE_DATABASE_URL=postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
SUPABASE_ACCESS_TOKEN=sbp_abc123...
SUPABASE_PROJECT_REF=abcdefghijklmnop
```

5. `config.yaml`'ı güncelleyin:

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

# MCP sunucularını güncelle — supabase_mcp'yi etkinleştir, postgres_mcp'yi devre dışı bırak
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

6. Veritabanı şemasını oluşturun (Supabase SQL Editor veya psql ile):

```bash
psql "${SUPABASE_DATABASE_URL}" -f db/migrations/001_initial_schema.sql
psql "${SUPABASE_DATABASE_URL}" -f db/seed/seed_data.sql
```

#### Veritabanı Sağlayıcı Karşılaştırması

| Özellik | PostgreSQL | Neon | Supabase |
|---------|-----------|------|----------|
| Tür | Self-hosted | Serverless | BaaS |
| Ücretsiz Plan | Docker ile | ✅ 0.5 GB | ✅ 500 MB |
| Otomatik Ölçeklendirme | ❌ | ✅ | ✅ |
| Sıfır Soğuk Başlatma | ✅ | ~0.5s | ✅ |
| Dahili Auth | ❌ | ❌ | ✅ |
| MCP Sunucusu | postgres_mcp | neon_mcp | supabase_mcp |
| Kurulum Zorluğu | Kolay (Docker) | Kolay | Kolay |
| Üretim İçin | ✅ | ✅ | ✅ |

### Vektör Deposu ve Embedding

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

Tüm seçenekler için [`config.yaml`](config.yaml) dosyasına bakın.

---

## API Referansı

| Metod | Endpoint | Açıklama |
|-------|----------|----------|
| `GET` | `/health` | Sağlık kontrolü |
| `GET` | `/info` | Sistem bilgisi (versiyon, aktif sağlayıcılar) |
| `POST` | `/api/v1/chat` | Senkron sohbet |
| `WS` | `/api/v1/chat/stream` | Akışlı sohbet (WebSocket) |
| `POST` | `/api/v1/documents` | Doküman yükle ve işle |
| `GET` | `/api/v1/documents` | Dokümanları listele |
| `DELETE` | `/api/v1/documents/{id}` | Doküman sil |
| `GET` | `/api/v1/customers` | Müşteri bilgisi sorgula |
| `GET` | `/api/v1/config` | Yapılandırmayı görüntüle |
| `PUT` | `/api/v1/config` | Çalışma zamanında yapılandırma güncelle |
| `GET` | `/api/v1/mcp/status` | MCP sunucu durumları |
| `GET` | `/api/v1/stats` | Token kullanımı ve istatistikler |

İnteraktif API dokümantasyonu: `http://localhost:8000/docs`

### Örnek İstekler

#### Sohbet

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Müşteri 1 numaralı müşterinin abonelik durumu nedir?"}'
```

#### Doküman Yükleme

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@kullanim_kilavuzu.pdf"
```

#### Sistem İstatistikleri

```bash
curl http://localhost:8000/api/v1/stats
```

---

## Proje Yapısı

```
agentdesk/
├── src/
│   ├── agent/          # AgentLoop — iteratif LLM ↔ MCP araç çağrısı döngüsü
│   ├── api/            # FastAPI router'ları (chat, documents, customers, config, stats)
│   ├── chunking/       # ChunkingEngine (recursive, semantic, document_aware)
│   ├── config/         # ConfigManager + Pydantic modelleri
│   ├── llm/            # LLMClient — birleşik OpenAI-uyumlu arayüz
│   ├── mcp/            # MCPManager — stdio/SSE yaşam döngüsü yönetimi
│   ├── models/         # API istek/yanıt şemaları
│   ├── router/         # IntentRouter — TF-IDF semantik sınıflandırma
│   ├── session/        # SessionManager — konuşma geçmişi
│   ├── store/          # ReferenceStore — TTL'li bellek deposu
│   ├── vectorstore/    # VectorStoreAdapter + QdrantVectorStore
│   ├── logging_config.py  # Yapılandırılmış loglama
│   └── main.py         # FastAPI uygulama giriş noktası
├── frontend/
│   └── gradio_app.py   # Gradio sohbet arayüzü
├── db/
│   ├── migrations/     # PostgreSQL şema migration'ları
│   └── seed/           # Örnek veriler
├── mcp_servers/
│   └── postgres_mcp/   # Salt okunur PostgreSQL MCP sunucu yapılandırması
├── tests/
│   ├── unit/           # Birim testleri (224 test)
│   ├── property/       # Property-based testler (Hypothesis)
│   └── integration/    # Uçtan uca akış testleri
├── config.yaml         # Ana yapılandırma dosyası
├── docker-compose.yml  # Çoklu servis orkestrasyonu
├── Dockerfile          # Multi-stage build
├── Makefile            # Geliştirme komutları
├── requirements.txt    # Python bağımlılıkları
└── .env.example        # Ortam değişkeni şablonu
```

---

## Geliştirme

### Gereksinimler

- Python 3.11+
- Docker & Docker Compose (servisler için)
- Git

### Ortam Kurulumu

```bash
# Repo'yu klonla
git clone https://github.com/ahmet-ozel/agentdesk.git
cd agentdesk

# Sanal ortam
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıklar
pip install -r requirements.txt

# Ortam değişkenleri
cp .env.example .env
```

### Kod Stili

Proje [Ruff](https://docs.astral.sh/ruff/) linter kullanır:

```bash
make lint
# veya
ruff check src/ tests/
```

---

## Test

Proje kapsamlı bir test altyapısına sahiptir:

```bash
# Tüm testleri çalıştır
make test
# veya
pytest tests/ -v

# Sadece birim testleri
make test-unit

# Sadece entegrasyon testleri
make test-int
```

### Test Yapısı

- **Birim Testleri** (`tests/unit/`): Her bileşen için izole testler
  - `test_database_config.py` — Veritabanı sağlayıcı yapılandırması (PostgreSQL, Neon, Supabase)
  - `test_config.py` — Yapılandırma yükleme ve doğrulama
  - `test_llm_client.py` — LLM istemci işlevselliği
  - `test_mcp_manager.py` — MCP sunucu yönetimi
  - `test_intent_router.py` — Niyet sınıflandırma
  - `test_agent_loop.py` — Agent loop iş akışı
  - `test_reference_store.py` — Reference store işlemleri
  - `test_session_manager.py` — Oturum yönetimi
  - `test_chunking.py` — Doküman parçalama
  - `test_vectorstore.py` — Vektör deposu işlemleri
  - `test_api_endpoints.py` — REST API endpoint'leri
  - `test_schemas.py` — API şema doğrulama
  - `test_logging_config.py` — Loglama kurulumu

- **Entegrasyon Testleri** (`tests/integration/`): Uçtan uca akış testleri
  - Sohbet akışı (Intent Router → Agent Loop → LLM → MCP → Yanıt)
  - Doküman işleme pipeline'ı
  - Müşteri sorgu akışı

- **Property Testleri** (`tests/property/`): Hypothesis ile doğruluk özellikleri

### Test Kapsamı

| Modül | Test Sayısı | Durum |
|-------|-------------|-------|
| Birim Testleri | 216 | ✅ Geçiyor |
| Entegrasyon Testleri | 8 | ✅ Geçiyor |
| **Toplam** | **224** | **✅ Tümü Geçiyor** |

---

## Docker ile Dağıtım

### Servisler

| Servis | Port | Açıklama |
|--------|------|----------|
| `agentdesk` | 8000 | FastAPI ana sunucu |
| `postgres` | 5432 | PostgreSQL müşteri veritabanı |
| `qdrant` | 6333, 6334 | Qdrant vektör veritabanı |
| `vllm` (GPU) | 8080 | vLLM yerel çıkarım sunucusu |

### Başlatma

```bash
# CPU modu (bulut LLM kullanır)
docker compose up -d

# GPU modu (yerel vLLM dahil)
docker compose --profile gpu up -d

# Logları izle
docker compose logs -f agentdesk

# Durdur
docker compose down
```

### Kalıcı Veriler

Docker volume'ları ile veriler korunur:
- `postgres_data` — PostgreSQL veritabanı verileri
- `qdrant_data` — Qdrant vektör indeksleri

---

## Desteklenen Sağlayıcılar

| Kategori | Seçenekler |
|----------|------------|
| **LLM** | vLLM, OpenAI, Anthropic, Google, Ollama |
| **Vektör Deposu** | Qdrant (varsayılan), Milvus, Chroma, pgvector'e genişletilebilir |
| **Embedding** | bge-m3 (yerel), OpenAI, Voyage, Cohere |
| **Doküman Parser** | docling-mcp, paddleocr-mcp (MCP yapılandırması ile) |
| **Müşteri DB** | PostgreSQL, Neon, Supabase |

---

## Ortam Değişkenleri

| Değişken | Açıklama | Zorunlu |
|----------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API anahtarı | LLM provider=openai ise |
| `ANTHROPIC_API_KEY` | Anthropic API anahtarı | LLM provider=anthropic ise |
| `GOOGLE_API_KEY` | Google API anahtarı | LLM provider=google ise |
| `VLLM_BASE_URL` | vLLM sunucu URL'i | LLM provider=vllm ise |
| `POSTGRES_HOST` | PostgreSQL host | db_provider=postgresql ise |
| `POSTGRES_PORT` | PostgreSQL port | db_provider=postgresql ise |
| `POSTGRES_DB` | Veritabanı adı | db_provider=postgresql ise |
| `POSTGRES_READONLY_USER` | Salt okunur kullanıcı | db_provider=postgresql ise |
| `POSTGRES_READONLY_PASSWORD` | Kullanıcı şifresi | db_provider=postgresql ise |
| `DB_PASSWORD` | PostgreSQL admin şifresi | Docker Compose için |
| `NEON_DATABASE_URL` | Neon bağlantı dizesi | db_provider=neon ise |
| `NEON_API_KEY` | Neon API anahtarı | db_provider=neon ise |
| `SUPABASE_DATABASE_URL` | Supabase bağlantı dizesi | db_provider=supabase ise |
| `SUPABASE_ACCESS_TOKEN` | Supabase erişim token'ı | db_provider=supabase ise |
| `SUPABASE_PROJECT_REF` | Supabase proje referans ID'si | db_provider=supabase ise |

Tüm değişkenler için [`.env.example`](.env.example) dosyasına bakın.

---

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Detaylar için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.
