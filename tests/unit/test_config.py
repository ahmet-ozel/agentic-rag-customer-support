"""Unit tests for ConfigManager."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from src.config.manager import ConfigError, ConfigManager, get_config


@pytest.fixture()
def valid_yaml(tmp_path: Path) -> Path:
    """Create a minimal valid config.yaml."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            app:
              name: TestApp
              port: 9000
            llm:
              default_provider: openai
              providers:
                openai:
                  base_url: http://localhost:8080/v1
                  model: gpt-4
        """),
        encoding="utf-8",
    )
    return cfg


@pytest.fixture()
def invalid_yaml(tmp_path: Path) -> Path:
    """Create a config.yaml with an invalid field type."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            app:
              port: not_a_number
        """),
        encoding="utf-8",
    )
    return cfg


class TestConfigManagerLoad:
    def test_load_valid_config(self, valid_yaml: Path) -> None:
        mgr = ConfigManager()
        config = mgr.load(str(valid_yaml))
        assert config.app.name == "TestApp"
        assert config.app.port == 9000
        assert config.llm.default_provider == "openai"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        mgr = ConfigManager()
        with pytest.raises(ConfigError, match="Config file not found"):
            mgr.load(str(tmp_path / "nonexistent.yaml"))

    def test_load_invalid_config_raises_with_field_name(self, invalid_yaml: Path) -> None:
        mgr = ConfigManager()
        with pytest.raises(ConfigError, match="port"):
            mgr.load(str(invalid_yaml))

    def test_load_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("", encoding="utf-8")
        mgr = ConfigManager()
        config = mgr.load(str(cfg))
        assert config.app.name == "AgentDesk"


class TestEnvVarResolution:
    def test_resolve_simple_env_var(self, valid_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "secret123")
        cfg = valid_yaml
        cfg.write_text(
            textwrap.dedent("""\
                llm:
                  default_provider: openai
                  providers:
                    openai:
                      base_url: http://localhost/v1
                      model: gpt-4
                      api_key: ${MY_API_KEY}
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg))
        assert config.llm.providers["openai"].api_key == "secret123"

    def test_unset_env_var_resolves_to_empty(self, valid_yaml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        cfg = valid_yaml
        cfg.write_text(
            textwrap.dedent("""\
                llm:
                  default_provider: openai
                  providers:
                    openai:
                      base_url: http://localhost/v1
                      model: gpt-4
                      api_key: ${NONEXISTENT_VAR}
            """),
            encoding="utf-8",
        )
        mgr = ConfigManager()
        config = mgr.load(str(cfg))
        assert config.llm.providers["openai"].api_key == ""

    def test_resolve_env_vars_in_nested_dict(self) -> None:
        os.environ["TEST_HOST"] = "myhost"
        result = ConfigManager._resolve_env_vars({"server": {"host": "${TEST_HOST}"}})
        assert result == {"server": {"host": "myhost"}}
        del os.environ["TEST_HOST"]

    def test_resolve_env_vars_in_list(self) -> None:
        os.environ["TEST_ARG"] = "hello"
        result = ConfigManager._resolve_env_vars(["${TEST_ARG}", "static"])
        assert result == ["hello", "static"]
        del os.environ["TEST_ARG"]

    def test_non_string_values_pass_through(self) -> None:
        assert ConfigManager._resolve_env_vars(42) == 42
        assert ConfigManager._resolve_env_vars(True) is True
        assert ConfigManager._resolve_env_vars(None) is None


class TestReload:
    def test_reload_picks_up_changes(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("app:\n  name: Before\n", encoding="utf-8")
        mgr = ConfigManager()
        mgr.load(str(cfg))
        assert mgr.get_config().app.name == "Before"

        cfg.write_text("app:\n  name: After\n", encoding="utf-8")
        mgr.reload()
        assert mgr.get_config().app.name == "After"

    def test_reload_without_load_raises(self) -> None:
        mgr = ConfigManager()
        with pytest.raises(ConfigError, match="Cannot reload"):
            mgr.reload()


class TestGetConfig:
    def test_get_config_before_load_raises(self) -> None:
        mgr = ConfigManager()
        with pytest.raises(ConfigError, match="not loaded"):
            mgr.get_config()

    def test_get_config_returns_loaded(self, valid_yaml: Path) -> None:
        mgr = ConfigManager()
        mgr.load(str(valid_yaml))
        config = mgr.get_config()
        assert config.app.name == "TestApp"


class TestModuleSingleton:
    def test_module_get_config_after_load(self, valid_yaml: Path) -> None:
        mgr = ConfigManager()
        mgr.load(str(valid_yaml))
        config = get_config()
        assert config.app.name == "TestApp"
