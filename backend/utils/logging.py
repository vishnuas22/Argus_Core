"""
Argus Core - Structured Logging
===============================
Configures structured logging with correlation IDs for distributed tracing.

Implements: PRIME_ARGUS_DOCUMENT.md - Section 2.2 - utils/logging.py
"""

import logging
import sys
from typing import Optional
from contextvars import ContextVar
from datetime import datetime, timezone
import json

# Context variable for correlation ID (request tracing)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Output format compatible with ELK stack and cloud logging services.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console formatter for development.
    """
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        correlation_id = correlation_id_var.get()
        correlation_str = f"[{correlation_id[:8]}]" if correlation_id else ""
        
        return (
            f"{timestamp} {color}{record.levelname:8}{self.RESET} "
            f"{correlation_str} {record.name}: {record.getMessage()}"
        )


def setup_logging(
    level: str = "INFO",
    json_format: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (True for production, False for development)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter based on format preference
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())
    
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for the current request context.
    
    Args:
        correlation_id: Unique request identifier
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """
    Get correlation ID for the current request context.
    
    Returns:
        Current correlation ID or None
    """
    return correlation_id_var.get()


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes extra context fields.
    """
    
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        
        # Add correlation ID
        correlation_id = correlation_id_var.get()
        if correlation_id:
            extra["correlation_id"] = correlation_id
        
        kwargs["extra"] = extra
        return msg, kwargs
