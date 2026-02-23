"""Tests for configuration loading."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import Config, load_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self) -> None:
        """Loading a valid YAML config returns a Config instance."""
        config = load_config("configs/config.yaml")
        assert isinstance(config, Config)
        assert "duckdb_path" in config.database
        assert "roberta_model_name" in config.model

    def test_config_database_section(self) -> None:
        """Database section contains expected keys."""
        config = load_config("configs/config.yaml")
        assert config.database["duckdb_path"] == "data/sentiment.duckdb"
        assert config.database["batch_size"] == 1000

    def test_config_model_section(self) -> None:
        """Model section contains expected keys."""
        config = load_config("configs/config.yaml")
        assert config.model["roberta_model_name"] == "roberta-base"
        assert config.model["max_length"] == 128
        assert config.model["num_labels"] == 3

    def test_config_training_section(self) -> None:
        """Training section contains expected keys."""
        config = load_config("configs/config.yaml")
        assert config.training["epochs"] == 3
        assert config.training["learning_rate"] == 2.0e-5

    def test_config_missing_file(self) -> None:
        """Loading from a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent/config.yaml")

    def test_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Loading malformed YAML raises a ParserError."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": :\n  - :\n    invalid")
        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_file))

    def test_config_empty_file(self, tmp_path: Path) -> None:
        """Loading an empty YAML file raises ValueError."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ValueError, match="Invalid configuration format"):
            load_config(str(empty))

    def test_config_partial_sections(self, tmp_path: Path) -> None:
        """Config with only some sections loads with defaults for missing ones."""
        partial = tmp_path / "partial.yaml"
        data = {"database": {"duckdb_path": "test.duckdb"}}
        partial.write_text(yaml.dump(data))
        config = load_config(str(partial))
        assert config.database["duckdb_path"] == "test.duckdb"
        assert config.model == {}

    def test_config_extra_keys_ignored(self, tmp_path: Path) -> None:
        """Extra keys in the YAML are silently ignored."""
        extra = tmp_path / "extra.yaml"
        data = {
            "database": {"duckdb_path": "x.db"},
            "unknown_section": {"key": "value"},
        }
        extra.write_text(yaml.dump(data))
        config = load_config(str(extra))
        assert not hasattr(config, "unknown_section")


class TestConfigDataclass:
    """Tests for the Config dataclass defaults."""

    def test_default_config(self) -> None:
        """Default Config has empty dicts for all fields."""
        config = Config()
        assert config.database == {}
        assert config.model == {}
        assert config.training == {}
        assert config.dashboard == {}
        assert config.api == {}
        assert config.simulator == {}
