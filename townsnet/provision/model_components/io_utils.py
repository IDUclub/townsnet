"""I/O helpers for provision model."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd


def _as_path(value: Union[str, Path]) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq", ".pqt"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".geojson":
        return gpd.read_file(path)
    raise ValueError(f"Unsupported file format: {path}")


def _ensure_numeric(series: pd.Series, *, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _drop_geometry(df: pd.DataFrame) -> pd.DataFrame:
    if gpd is not None and isinstance(df, gpd.GeoDataFrame):
        return pd.DataFrame(df.drop(columns=["geometry"], errors="ignore"))
    return df.drop(columns=["geometry"], errors="ignore")


__all__ = ["_as_path", "_read_table", "_ensure_numeric", "_drop_geometry"]
