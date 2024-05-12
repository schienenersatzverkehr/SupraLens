from datetime import date
from venv import logger

import pandas as pd
import psutil


def log_memory_details(nbytes: int) -> (int, int):
    mem = psutil.virtual_memory()
    logger.info(
        f"\nDEM memory usage:"
        f"\n\t{nbytes / 1024 ** 3:.2f}/{mem.total / 1024 ** 3}GB"
        f"\nTotal Memory usage: {mem.percent}%"
    )
    return mem.total, mem.percent


def extract_timestamp_from_stem(stem: str) -> date:
    datetime_str = stem.split("_")[3]
    return pd.to_datetime(datetime_str).date()
