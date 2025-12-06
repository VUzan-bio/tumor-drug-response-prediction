from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO", filename: Optional[str] = None) -> None:
    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        filename=filename,
    )
