import logging
import os
import sys
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler

class JSONFormatter(logging.Formatter):
    """
    Formatter to output logs in JSON format.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "path": record.pathname
        }
        
        # Merge extra attributes if present (e.g. passed via extra={...})
        # Note: Standard logging attributes are skipped to avoid clutter
        # Custom attributes added to the record will be included
        skip_keys = {
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
        }
        
        for key, value in record.__dict__.items():
            if key not in skip_keys and not key.startswith('_'):
                log_record[key] = value
            
        # Handle exceptions
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record, ensure_ascii=False)

def setup_logger(name="autoviewmem_memory", log_file=None, level=None, format_type=None):
    """
    Sets up a logger with console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file. If None, defaults to logs/YYYY-MM-DD/{name}_YYYY-MM-DD.log.
        level (int): Logging level. Defaults to env LOG_LEVEL or logging.INFO.
        format_type (str): 'text' or 'json'. Defaults to env LOG_FORMAT or 'text'.
    
    Returns:
        logging.Logger: Configured logger.
    """
    # Load defaults from environment variables
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
        
    if format_type is None:
        format_type = os.getenv("LOG_FORMAT", "text").lower()

    if log_file is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H%M%S')
        log_dir = os.path.join(current_dir, "logs", today_str)
        log_filename = f"{name}_{today_str}_{time_str}.log"
        log_file = os.path.join(log_dir, log_filename)

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            # Fallback to current directory if permission denied or other error
            log_file = f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times if logger is already configured
    if logger.handlers:
        return logger

    # Formatters
    if format_type == 'json':
        formatter = JSONFormatter()
    else:
        # Enhanced text format with file and line info
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    # 10MB per file, max 5 backup files
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging: {e}")

    return logger

# Create a default logger instance for easy import
logger = setup_logger()
