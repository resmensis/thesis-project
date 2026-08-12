"""
I/O utilities for the equity prediction project.

This module provides utilities for:
- Directory management
- File I/O (parquet, pickle)
- Random seed management
- Logging setup
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import random
import torch


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str, enabled: bool = True) -> None:
    """Save DataFrame to parquet if enabled."""
    if enabled:
        ensure_dir(os.path.dirname(path))
        df.to_parquet(path, index=False)


def maybe_load_parquet(path: str, enabled: bool = True) -> Optional[pd.DataFrame]:
    """Load DataFrame from parquet if enabled and file exists."""
    if enabled and os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save_pickle(obj: object, path: str, enabled: bool = True) -> None:
    """Save object to pickle if enabled."""
    if enabled:
        ensure_dir(os.path.dirname(path))
        pd.to_pickle(obj, path)


def maybe_load_pickle(path: str, enabled: bool = True) -> Optional[object]:
    """Load object from pickle if enabled and file exists."""
    if enabled and os.path.exists(path):
        return pd.read_pickle(path)
    return None


def set_global_seed(seed: int, torch_deterministic: bool = True) -> None:
    """
    Set global random seed for reproducibility.
    
    This should be called ONCE at the start of the program to ensure
    all random operations are reproducible across runs.
    
    Args:
        seed: Random seed (e.g., 42)
        torch_deterministic: If True, set torch to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)
        if torch_deterministic:
            torch.use_deterministic_algorithms(True)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def setup_project_logger(
    logger_name: str = "eap_ml",
    log_dir: str = "logs",
    regime_mode: str = "full",
    overwrite_log: bool = False,
    file_level_full: str = "INFO",
    file_level_coding: str = "DEBUG",
    console_level_full: str = "INFO",
    console_level_coding: str = "DEBUG",
) -> Tuple[logging.Logger, str]:
    """
    Setup and return the project logger.
    
    This should be called ONCE at the start of main() to configure logging.
    All other modules should use get_project_logger() to access the same logger.
    
    Args:
        logger_name: Name of the logger
        log_dir: Directory to save log files
        regime_mode: "full" or "coding_reduced" (affects log levels)
        overwrite_log: If True, overwrite existing log file
        file_level_full: Log level for file handler in full mode
        file_level_coding: Log level for file handler in coding mode
        console_level_full: Log level for console handler in full mode
        console_level_coding: Log level for console handler in coding mode
    
    Returns:
        logger: Configured logger instance
        log_path: Path to the log file
    """
    # Determine log levels based on mode
    if regime_mode == "coding_reduced":
        file_level = file_level_coding
        console_level = console_level_coding
    else:  # full mode
        file_level = file_level_full
        console_level = console_level_full
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
    
    # Clear existing handlers (important for avoiding duplicates)
    logger.handlers.clear()
    
    # Ensure log directory exists
    ensure_dir(log_dir)
    
    # Create log file path
    log_filename = f"{logger_name}_{regime_mode}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Remove old log file if overwrite
    if overwrite_log and os.path.exists(log_path):
        os.remove(log_path)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, file_level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_path


def get_project_logger(name: str = "eap_ml") -> logging.Logger:
    """
    Return the project logger.
    
    Assumes setup_project_logger() was called once at program start.
    Falls back to getLogger(name) if not configured.
    
    Args:
        name: Logger name (default: "eap_ml")
    
    Returns:
        logging.Logger: The project logger instance
    
    Example:
        >>> from io_utils import get_project_logger
        >>> logger = get_project_logger()
        >>> logger.info("This will log to file and console")
    """
    return logging.getLogger(name)