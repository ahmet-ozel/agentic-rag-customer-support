"""Unit tests for database configuration models.

Covers PostgreSQL, Neon, and Supabase provider configs,
DatabaseConfig provider selection, connection strings,
and config.yaml loading with database sections.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config.manager import ConfigManager
from src.config.models import (
    DatabaseConfig,
    NeonConfig,
    PostgreSQLConfig,
    SupabaseConfig,
)


# ---------------------------------------------------------------------------
# PostgreSQLConfig
# ---------------------------------------------------------------------------


class TestPostgreSQLConfig:
    def test_defaults(self) -> None:
        cfg = PostgreSQLConfig()
        assert cfg.host == "localhost"
        assert cfg.port == "5432"
        assert cfg.database == "agentdesk"
        assert cfg.user == "agentdesk_readonly"
        assert cfg.ssl_mode == "prefer"

    def test_connection_string(self) -> None:
        cfg = PostgreSQLConfig(
            host="db.example.com",
            port="5433",
            database="mydb",
            user="reader",
            password="s3cret",
            ssl_mode="require",
        )
        cs = cfg.connection_string()
        assert cs == "postgresql://reader:s3cret@db.example.com:5433/mydb?sslmode=require"

    def test_connection_string_empty_password(self) -> None:
        cfg = PostgreSQLConfig(host="localhost", password="")
        cs = cfg.connection_string()
        assert "localhost" in cs
        assert ":@" in cs  # empty password


# ---------------------------------------------------------------------------
# NeonConfig
# ---------------------------------------------------------------------------


class TestNeonConfig:
    def test_defaults(self) -> None:
        cfg = NeonConfig()
        assert cfg.connection_string == ""
        assert cfg.ssl_mode == "require"

    def test_get_connection_string_appends_ssl(self) -> None:
        cfg = NeonConfig(
            connection_string="postgresql://user:pass@ep-cool-123.us-east-2.aws.neon.tech/neondb"
        )
        cs = cfg.get_connection_string()
        assert "sslmode=require" in cs
        assert cs.startswith("postgresql://")

    def test_get_connection_string_preserves_existing_ssl(self) -> None:
        cfg = NeonConfig(
            connection_string="postgresql://u:p@host/db?sslmode=verify-full"
        )
        cs = cfg.get_connection_string()
        assert "sslmode=verify-full" in cs
        assert cs.count("sslmode") == 1  # not duplicated

    def test_get_connection_string_empty(self) -> None:
        cfg = NeonConfig(connection_string="")
        assert cfg.get_connection_string() == ""

    def test_get_connection_string_with_existing_params(self) -> None:
        cfg = NeonConfig(
            connection_string="postgresql://u:p@host/db?connect_timeout=10"
        )
        cs = cfg.get_connection_string()
        assert "&sslmode=require" in cs


# ---------------------------------------------------------------------------
# SupabaseConfig
# ---------------------------------------------------------------------------


class TestSupabaseConfig:
    def test_defaults(self) -> None:
        cfg = SupabaseConfig()
        assert cfg.connection_string == ""
        assert cfg.access_token == ""
        assert cfg.project_ref == ""
        assert cfg.ssl_mode == "require"

    def test_get_connection_string_appends_ssl(self) -> None:
        cfg = SupabaseConfig(
            connection_string="postgresql://postgres.abc123:pass@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        )
        cs = cfg.get_connection_string()
        assert "sslmode=require" in cs

    def test_get_connection_string_preserves_existing_ssl(self) -> None:
        cfg = SupabaseConfig(
            connection_string="postgresql://u:p@host/db?sslmode=verify-ca"
        )
        cs = cfg.get_connection_string()
        assert cs.count("sslmode") == 1

    def test_get_connection_string_empty(self) -> None:
        cfg = SupabaseConfig(connection_string="")
        assert cfg.get_connection_string() == ""

    def test_full_config(self) -> None:
        cfg = SupabaseConfig(
            connection_string="postgresql://u:p@host/db",
            access_token="sbp_token123",
            project_ref="abcdefghijklmnop",
        )
        assert cfg.access_token == "sbp_token123"
        assert cfg.project_ref == "abcdefghijklmnop"


# ---------------------------------------------------------------------------
# DatabaseConfig — provider selection
# ---------------------------------------------------------------------------


class TestDatabaseConfigProviderSelection:
    def test_default_provider_is_postgresql(self) -> None:
        cfg = DatabaseConfig()
        assert cfg.db_provider == "postgresql"
        active = cfg.get_active()
        assert isinstance(active, PostgreSQLConfig)

    def test_select_neon(self) -> None:
        cfg = DatabaseConfig(
            db_provider="neon",
            neon=NeonConfig(connection_string="postgresql://neon-host/db"),
        )
        active = cfg.get_active()
        assert isinstance(active, NeonConfig)
        assert "neon-host" in active.connection_string

    def test_select_supabase(self) -> None:
        cfg = DatabaseConfig(
            db_provider="supabase",
            supabase=SupabaseConfig(
                connection_string="postgresql://supabase-host/db",
                access_token="token",
                project_ref="ref123",
            ),
        )
        active = cfg.get_active()
        assert isinstance(active, SupabaseConfig)
        assert active.access_token == "token"

    def test_unsupported_provider_raises(self) -> None:
        cfg = DatabaseConfig(db_provider="mysql")
        with pytest.raises(ValueError, match="not supported"):
            cfg.get_active()

    def test_providers_dict_populates_postgresql(self) -> None:
        cfg = DatabaseConfig(
            db_provider="postgresql",
            providers={
                "postgresql": {
                    "host": "custom-host",
                    "port": "5433",
                    "database": "custom_db",
                }
            },
        )
        assert cfg.postgresql.host == "custom-host"
        assert cfg.postgresql.port == "5433"

    def test_providers_dict_populates_neon(self) -> None:
        cfg = DatabaseConfig(
            db_provider="neon",
            providers={
                "neon": {
                    "connection_string": "postgresql://neon/db",
                }
            },
        )
        assert cfg.neon.connection_string == "postgresql://neon/db"

    def test_providers_dict_populates_supabase(self) -> None:
        cfg = DatabaseConfig(
            db_provider="supabase",
            providers={
                "supabase": {
                    "connection_string": "postgresql://supa/db",
                    "access_token": "tok",
                    "project_ref": "ref",
                }
            },
        )
        assert cfg.supabase.connection_string == "postgresql://supa/db"
        assert cfg.supabase.access_token == "tok"


# ---------------------------------------------------------------------------
# Loading from config.yaml
# ---------------------------------------------------------------------------


class TestDatabaseConfigFromYAML:
    def test_load_postgresql_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: postgresql
                  providers:
                    postgresql:
                      host: myhost
                      port: "5433"
                      database: mydb
                      user: myuser
                      password: mypass
                      ssl_mode: require
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        db = config.database
        assert db.db_provider == "postgresql"
        pg = db.get_active()
        assert isinstance(pg, PostgreSQLConfig)
        assert pg.host == "myhost"
        assert pg.port == "5433"
        assert "myuser" in pg.connection_string()

    def test_load_neon_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: neon
                  providers:
                    neon:
                      connection_string: "postgresql://user:pass@ep-cool.neon.tech/neondb"
                      ssl_mode: require
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        db = config.database
        assert db.db_provider == "neon"
        neon = db.get_active()
        assert isinstance(neon, NeonConfig)
        assert "neon.tech" in neon.connection_string

    def test_load_supabase_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: supabase
                  providers:
                    supabase:
                      connection_string: "postgresql://postgres.abc:pass@pooler.supabase.com:6543/postgres"
                      access_token: "sbp_test_token"
                      project_ref: "abcdefghijklmnop"
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        db = config.database
        assert db.db_provider == "supabase"
        supa = db.get_active()
        assert isinstance(supa, SupabaseConfig)
        assert "supabase.com" in supa.connection_string
        assert supa.access_token == "sbp_test_token"
        assert supa.project_ref == "abcdefghijklmnop"

    def test_load_with_env_vars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_NEON_URL", "postgresql://neon-env/db")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: neon
                  providers:
                    neon:
                      connection_string: "${TEST_NEON_URL}"
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        neon = config.database.get_active()
        assert isinstance(neon, NeonConfig)
        assert neon.connection_string == "postgresql://neon-env/db"

    def test_switch_provider_via_reload(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: postgresql
                  providers:
                    postgresql:
                      host: localhost
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        mgr.load(str(cfg_file))
        assert isinstance(mgr.get_config().database.get_active(), PostgreSQLConfig)

        # Switch to neon
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: neon
                  providers:
                    neon:
                      connection_string: "postgresql://neon/db"
            """),
            encoding="utf-8",
        )
        mgr.reload()
        assert isinstance(mgr.get_config().database.get_active(), NeonConfig)


# ---------------------------------------------------------------------------
# MCP server config alignment with db_provider
# ---------------------------------------------------------------------------


class TestMCPServerDBAlignment:
    def test_neon_mcp_config_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: neon
                  providers:
                    neon:
                      connection_string: "postgresql://neon/db"
                mcp_servers:
                  neon_mcp:
                    enabled: true
                    transport: stdio
                    command: npx
                    args: ["-y", "@neondatabase/mcp-server-neon", "start"]
                    env:
                      NEON_API_KEY: "test-key"
                  postgres_mcp:
                    enabled: false
                    transport: stdio
                    command: python
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        assert config.mcp_servers["neon_mcp"].enabled is True
        assert config.mcp_servers["postgres_mcp"].enabled is False

    def test_supabase_mcp_config_in_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
                database:
                  db_provider: supabase
                  providers:
                    supabase:
                      connection_string: "postgresql://supa/db"
                      access_token: "tok"
                      project_ref: "ref"
                mcp_servers:
                  supabase_mcp:
                    enabled: true
                    transport: stdio
                    command: npx
                    args: ["-y", "@supabase/mcp-server-supabase@latest"]
                    env:
                      SUPABASE_ACCESS_TOKEN: "tok"
                  postgres_mcp:
                    enabled: false
                    transport: stdio
                    command: python
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg_file))
        assert config.mcp_servers["supabase_mcp"].enabled is True
        assert config.mcp_servers["postgres_mcp"].enabled is False
