from pathlib import Path
import os
import pickle
import random
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def maybe_load_parquet(path, enabled=True):
    path = Path(path)
    if enabled and path.exists():
        return pd.read_parquet(path)
    return None


def save_parquet(df, path, enabled=True):
    if enabled:
        ensure_dir(Path(path).parent)
        df.to_parquet(path, index=False)


def maybe_load_pickle(path, enabled=True):
    path = Path(path)
    if enabled and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_pickle(obj, path, enabled=True):
    if enabled:
        ensure_dir(Path(path).parent)
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def set_global_seed(random_state: int = 42, torch_deterministic: bool = True):
    random.seed(random_state)
    np.random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    try:
        import torch
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _get_level(level_name: str) -> int:
    return getattr(logging, str(level_name).upper(), logging.INFO)


def setup_project_logger(
    logger_name: str,
    log_dir: str,
    regime_mode: str,
    overwrite_log: bool = False,
    file_level_full: str = "INFO",
    file_level_coding: str = "DEBUG",
    console_level_full: str = "INFO",
    console_level_coding: str = "DEBUG",
):
    """
    Create a project logger that writes both to console and to a log file.

    coding mode -> detailed DEBUG logs
    full / extended mode -> higher-level INFO logs
    """
    ensure_dir(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    
    file_level = _get_level(file_level_coding)
    console_level = _get_level(console_level_coding)
    
    """
    if regime_mode == "coding":
        file_level = _get_level(file_level_coding)
        console_level = _get_level(console_level_coding)
    else:
        file_level = _get_level(file_level_full)
        console_level = _get_level(console_level_full)
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{logger_name}_{regime_mode}_{timestamp}.log"

    file_mode = "w" if overwrite_log else "a"

    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized.")
    logger.info(f"Regime mode: {regime_mode}")
    logger.info(f"Log file: {log_path}")

    return logger, log_path


def get_logger(logger_name: str = "eap_ml"):
    return logging.getLogger(logger_name)