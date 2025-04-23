import logging
import sys
from typing import Any, Dict, Optional

class CustomFormatter(logging.Formatter):
    """
    Custom formatter with colors for different log levels.
    """
    
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.COLORS.get(record.levelno, self.RESET) + self._fmt + self.RESET
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with custom formatter.
    
    Args:
        name: Logger name (default: None, which returns the root logger)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = CustomFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger
