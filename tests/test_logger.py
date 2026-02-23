"""Tests for the logging utility."""

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_logger(self) -> None:
        """get_logger returns a logging.Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self) -> None:
        """Logger has the expected name."""
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_logger_level(self) -> None:
        """Logger level matches the requested level."""
        logger = get_logger("test_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_logger_has_handler(self) -> None:
        """Logger has at least one handler attached."""
        logger = get_logger("test_handler")
        assert len(logger.handlers) >= 1

    def test_no_duplicate_handlers(self) -> None:
        """Calling get_logger twice doesn't duplicate handlers."""
        logger1 = get_logger("test_no_dup")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("test_no_dup")
        assert len(logger2.handlers) == handler_count

    def test_default_level_is_info(self) -> None:
        """Default log level is INFO."""
        logger = get_logger("test_default_level")
        assert logger.level == logging.INFO
