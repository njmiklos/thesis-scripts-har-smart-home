"""
This script sets up a custom logger for individual modules, storing log outputs in a structured log file. 
It applies different formatting styles based on log levels and ensures each module gets a separate log file 
under a common logging directory.

Environment Configuration:
- Set `BASE_PATH` in your `.env` file to define the root project directory.
- Log files are saved to a `logs/` subdirectory inside the base path with the module name as the filename.
- Logging includes support for all standard levels: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
import logging

from utils.get_env import get_path_from_env


def get_logger(module_name: str):
    """
    Creates and returns a logger specific to the module.

    Args:
        module_name (str): The name of the module for which logs are generated.

    Returns:
        logging.Logger: Configured logger instance.
    """
    base_path = get_path_from_env('BASE_PATH')
    log_file = base_path / 'logs' / f'{module_name}.log'

    logger = logging.getLogger(module_name)

    # If logger handler does not exist, configure it
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Capture all log levels

        # File handler for writing logs to file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Custom formatter for different log levels
        class CustomFormatter(logging.Formatter):
            formats = {
                logging.INFO: logging.Formatter('%(message)s'),  # INFO: Only message
                logging.WARNING: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
                logging.ERROR: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
                logging.CRITICAL: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
                logging.DEBUG: logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'),
            }

            def format(self, record):
                formatter = self.formats.get(record.levelno, self.formats[logging.INFO])
                return formatter.format(record)

        # Set custom formatter to the file handler
        file_handler.setFormatter(CustomFormatter())

        # Add handler to the logger
        logger.addHandler(file_handler)

        logger.info(f"Logger initialized for module: {module_name}")
        print(f"Logs for {module_name} are being written to {log_file}")

    return logger
