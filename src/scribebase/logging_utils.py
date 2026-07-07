from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(data_dir: Path) -> logging.Logger:
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("scribebase")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(logs / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
