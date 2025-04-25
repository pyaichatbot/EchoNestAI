import logging
import sys
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel


class LoggingSettings(BaseModel):
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FORMAT: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(name: str):
    """Configure logging with loguru"""
    logging_settings = LoggingSettings()
    
    # Remove default loggers
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        enqueue=True,
        backtrace=True,
        level=logging_settings.LOGGING_LEVEL,
        format=logging_settings.LOGGING_FORMAT,
    )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    # Intercept uvicorn logging
    """ for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
        _logger = logging.getLogger(_log)
        _logger.handlers = [InterceptHandler()] """
    _logger = logging.getLogger(name)
    _logger.handlers = [InterceptHandler()]

    return logger
