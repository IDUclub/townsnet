"""Shared type aliases and constants for provision model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import pandas as pd

CityInfoSource = Union[str, Path, pd.DataFrame]
ServiceInput = Union[str, Path, pd.DataFrame]

REQUIRED_SERVICE_COLUMNS: Tuple[str, ...] = (
    "demand",
    "demand_within",
    "demand_without",
    "capacity",
    "capacity_left",
)

__all__ = ["CityInfoSource", "ServiceInput", "REQUIRED_SERVICE_COLUMNS"]
