import logging
import os
from threading import Lock


class LoggerSingleton:
    """
    Singleton Logger class to log messages to file or console selectively.

    Use:
        Initialize once in any file as:
            ```
            LoggerSingleton(name=LOGGER_NAME, log_dir=LOG_DIR, log_file_name=LOG_FNAME)
            ```
        In other files import and use as:
            ```
            from ood_detection.utils.logger import LoggerSingleton

            logger = LoggerSingleton.get_logger(LOGGER_NAME)
            logger.info("msg")
            ```
    """

    _instances = {}  # Dictionary to hold multiple named loggers
    _lock = Lock()  # Ensures thread safety

    def __new__(
        cls,
        name: str = "default",
        log_dir: str = "logs",
        log_file_name: str = "train.txt",
        fmt: str = "%(message)s",
    ):
        with cls._lock:
            if name not in cls._instances:
                instance = super(LoggerSingleton, cls).__new__(cls)
                instance._init_logger(name, log_dir, log_file_name, fmt)
                cls._instances[name] = instance
        return cls._instances[name]

    def _init_logger(self, name: str, log_dir: str, log_file_name: str, fmt: str):
        """Initialize logger for the given name and file."""
        self.log_dir = log_dir
        self.log_file_path = os.path.join(log_dir, log_file_name)

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt)

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(formatter)

        # Create file handler
        self.file_handler = logging.FileHandler(self.log_file_path, mode="w", encoding="utf-8")
        self.file_handler.setFormatter(formatter)

        # Attach handlers
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

    @staticmethod
    def get_logger(name: str = "default"):
        """Retrieve the singleton logger instance."""
        if name not in LoggerSingleton._instances:
            raise ValueError(f"Logger '{name}' has not been initialized. Initialize it in train.py first.")
        return LoggerSingleton._instances[name]

    def log(self, message, level="info", to_console=True, to_file=True):
        """Handles the logging logic based on provided settings."""
        if not to_console and not to_file:
            return  # If neither logging method is selected, do nothing

        if level == "info":
            log_func = self.logger.info
        elif level == "error":
            log_func = self.logger.error
        elif level == "warning":
            log_func = self.logger.warning
        elif level == "debug":
            log_func = self.logger.debug
        else:
            raise ValueError(f"Invalid log level: {level}")

        if to_console:
            self.console_handler.setLevel(logging.INFO)
        else:
            self.console_handler.setLevel(logging.CRITICAL)  # Disable lower-level logging

        if to_file:
            self.file_handler.setLevel(logging.INFO)
        else:
            self.file_handler.setLevel(logging.CRITICAL)  # Disable lower-level logging

        log_func(message)

    def info(self, message, to_console=True, to_file=True):
        self.log(message, level="info", to_console=to_console, to_file=to_file)

    def error(self, message, to_console=True, to_file=True):
        self.log(message, level="error", to_console=to_console, to_file=to_file)

    def warning(self, message, to_console=True, to_file=True):
        self.log(message, level="warning", to_console=to_console, to_file=to_file)

    def debug(self, message, to_console=True, to_file=True):
        self.log(message, level="debug", to_console=to_console, to_file=to_file)


EXP_LOGGER_NAME = "experiment"
TRAIN_METRICS_LOGGER_NAME = "train_metrics"
VAL_METRICS_LOGGER_NAME = "val_metrics"
TEST_METRICS_LOGGER_NAME = "test_metrics"
