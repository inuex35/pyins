# Copyright 2024 inuex35
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging configuration for RTK+INS system"""

import logging
import sys
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """Log levels for the system"""
    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Add TRACE level to logging
logging.addLevelName(LogLevel.TRACE.value, "TRACE")


def trace(self, message, *args, **kwargs):
    """Add trace method to logger"""
    if self.isEnabledFor(LogLevel.TRACE.value):
        self._log(LogLevel.TRACE.value, message, args, **kwargs)


# Add trace method to Logger class
logging.Logger.trace = trace


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    # Color codes
    COLORS = {
        'TRACE': '\033[36m',     # Cyan
        'DEBUG': '\033[34m',     # Blue
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str = "rtk_ins",
                level: str = "INFO",
                log_file: Optional[str] = None,
                console: bool = True) -> logging.Logger:
    """
    Setup logger with specified configuration

    Parameters:
    -----------
    name : str
        Logger name
    level : str
        Log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str]
        Log file path (if None, no file logging)
    console : bool
        Enable console output

    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(LogLevel, level.upper()).value)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(LogLevel, level.upper()).value)

        # Use colored formatter for console
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(LogLevel, level.upper()).value)

        # Plain formatter for file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger by name"""
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary log level change"""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(LogLevel, level.upper()).value
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


# Module level functions for different log levels
def trace(msg, *args, **kwargs):
    """Log trace message"""
    logger = logging.getLogger("rtk_ins")
    logger.trace(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Log debug message"""
    logger = logging.getLogger("rtk_ins")
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log info message"""
    logger = logging.getLogger("rtk_ins")
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log warning message"""
    logger = logging.getLogger("rtk_ins")
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log error message"""
    logger = logging.getLogger("rtk_ins")
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log critical message"""
    logger = logging.getLogger("rtk_ins")
    logger.critical(msg, *args, **kwargs)


class LoggerConfig:
    """Logger configuration manager for module-specific log levels"""

    def __init__(self):
        self.module_levels = {}
        self.default_level = "INFO"
        self.log_file = None
        self.console = True

    def set_module_level(self, module_name: str, level: str):
        """Set log level for specific module"""
        self.module_levels[module_name] = level
        # Update logger if it already exists
        logger = logging.getLogger(module_name)
        if logger.handlers:
            logger.setLevel(getattr(LogLevel, level.upper()).value)
            for handler in logger.handlers:
                handler.setLevel(getattr(LogLevel, level.upper()).value)

    def set_default_level(self, level: str):
        """Set default log level for all modules"""
        self.default_level = level

    def get_level_for_module(self, module_name: str) -> str:
        """Get log level for specific module"""
        return self.module_levels.get(module_name, self.default_level)

    def configure_from_dict(self, config: dict):
        """Configure from dictionary"""
        if 'default_level' in config:
            self.default_level = config['default_level']
        if 'log_file' in config:
            self.log_file = config['log_file']
        if 'console' in config:
            self.console = config['console']
        if 'module_levels' in config:
            for module, level in config['module_levels'].items():
                self.set_module_level(module, level)

    def setup_all_loggers(self):
        """Setup all configured loggers"""
        # Setup default logger
        setup_logger("rtk_ins", self.default_level, self.log_file, self.console)

        # Setup module-specific loggers
        for module, level in self.module_levels.items():
            setup_logger(module, level, self.log_file, self.console)


# Global logger configuration
logger_config = LoggerConfig()


def setup_logger_from_config(config: dict):
    """Setup loggers from configuration dictionary

    Example config:
    {
        'default_level': 'INFO',
        'log_file': 'app.log',
        'console': True,
        'module_levels': {
            'OhkamiValidator': 'DEBUG',
            'GTSAMRTKINSEstimator': 'TRACE',
            'IMUProcessor': 'WARNING'
        }
    }
    """
    logger_config.configure_from_dict(config)
    logger_config.setup_all_loggers()
