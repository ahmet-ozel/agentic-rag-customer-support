"""Pydantic configuration models for AgentDesk RAG Platform.

Mirrors the structure of config.yaml exactly.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class LLMProviderConfig(BaseModel):
    base_url: str = ""
    model: str = ""
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60


class TieredLLMConfig(BaseModel):
    enabled: bool = False
    routing_provider: str = "openai"
    routing_model: str = ""


class LLMConfig(BaseModel):
    default_provider: str = "openai"
    providers: dict[str, LLMProviderConfig] = {}
    tiered: TieredLLMConfig = TieredLLMConfig()

    def get_active(self) -> LLMProviderConfig:
        cfg = self.providers.get(self.default_provider)
        if cfg is None:
            raise ValueError(
                f"LLM provider '{self.default_provider}' not found in providers. "
                f"Available: {list(self.providers.keys())}"
            )
        return cfg

    def get_routing(self) -> LLMProviderConfig:
        if not self.tiered.enabled:
            return self.get_active()
        cfg = self.providers.get(self.tiered.routing_provider)
        if cfg is None:
            raise ValueError(
                f"Routing provider '{self.tiered.routing_provider}' not found in providers."
            )
        if self.tiered.routing_model:
            return cfg.model_copy(update={"model": self.tiered.routing_model})
        return cfg


class PostgreSQLConfig(BaseModel):
    host: str = "localhost"
    port: str = "5432"
    database: str = "agentdesk"
    user: str = "agentdesk_readonly"
    password: str = ""
    ssl_mode: str = "prefer"

    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


class NeonConfig(BaseModel):
    connection_string: str = ""
    ssl_mode: str = "require"

    def get_connection_string(self) -> str:
        """Return the Neon connection string with SSL mode appended if needed."""
        cs = self.connection_string
        if cs and "sslmode" not in cs:
            separator = "&" if "?" in cs else "?"
            cs = f"{cs}{separator}sslmode={self.ssl_mode}"
        return cs



class SupabaseConfig(BaseModel):
    connection_string: str = ""
    access_token: str = ""
    project_ref: str = ""
    ssl_mode: str = "require"

    def get_connection_string(self) -> str:
        """Return the Supabase connection string with SSL mode appended if needed."""
        cs = self.connection_string
        if cs and "sslmode" not in cs:
            separator = "&" if "?" in cs else "?"
            cs = f"{cs}{separator}sslmode={self.ssl_mode}"
        return cs


class DatabaseConfig(BaseModel):
    """Customer database configuration.

    db_provider selects the active backend: postgresql | neon | supabase
    """

    db_provider: str = "postgresql"
    providers: dict = {}

    postgresql: PostgreSQLConfig = PostgreSQLConfig()
    neon: NeonConfig = NeonConfig()
    supabase: SupabaseConfig = SupabaseConfig()

    @model_validator(mode="before")
    @classmethod
    def _parse_providers(cls, values: dict) -> dict:
        providers = values.get("providers", {})
        if "postgresql" in providers:
            values["postgresql"] = providers["postgresql"]
        if "neon" in providers:
            values["neon"] = providers["neon"]
        if "supabase" in providers:
            values["supabase"] = providers["supabase"]
        return values

    def get_active(self) -> PostgreSQLConfig | NeonConfig | SupabaseConfig:
        mapping = {
            "postgresql": self.postgresql,
            "neon": self.neon,
            "supabase": self.supabase,
        }
        cfg = mapping.get(self.db_provider)
        if cfg is None:
            raise ValueError(
                f"Database provider '{self.db_provider}' not supported. "
                f"Supported: {list(mapping.keys())}"
            )
        return cfg


class MCPServerConfig(BaseModel):
    enabled: bool = True
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "documents"
    hybrid_search: bool = False
    reranker: RerankerConfig | None = None


class EmbeddingConfig(BaseModel):
    model: str = "bge-m3"
    provider: str = "local"
    base_url: str | None = None
    api_key: str | None = None
    dimension: int = 1024


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 50


class IntentCategory(BaseModel):
    utterances: list[str] = []
    action: str = "agent_loop"
    response: str | None = None


class IntentRouterConfig(BaseModel):
    categories: dict[str, IntentCategory] = {}


class SessionConfig(BaseModel):
    max_messages: int = 50
    timeout_minutes: int = 30


class LoggingConfig(BaseModel):
    log_conversations: bool = True
    log_tool_calls: bool = True
    log_file: str = "./logs/agentdesk.log"
    track_token_usage: bool = True
    cost_log_file: str = "./logs/cost.log"


class AppSettings(BaseModel):
    name: str = "AgentDesk"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    max_agent_iterations: int = 10


class AppConfig(BaseModel):
    """Root configuration model — mirrors config.yaml."""

    app: AppSettings = AppSettings()
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}
    intent_router: IntentRouterConfig = IntentRouterConfig()
    session: SessionConfig = SessionConfig()
    logging: LoggingConfig = LoggingConfig()
