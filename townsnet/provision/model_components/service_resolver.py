"""Helpers to resolve service names and ids."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .io_utils import _as_path
from .service_catalog import (
    _canonical_service_name,
    _extract_service_id,
    _service_name_from_id,
    _service_name_from_path,
)


def _infer_service_name(payload) -> Optional[str]:
    if isinstance(payload, pd.DataFrame):
        df = payload
        columns = {str(col).strip().lower(): col for col in df.columns}
        if "service_name" in columns:
            values = df[columns["service_name"]].dropna().unique()
            if len(values) == 1:
                name = str(values[0])
                canonical = _canonical_service_name(name)
                return canonical or name
        for key in ("service_id", "service_code"):
            if key in columns:
                values = df[columns[key]].dropna().unique()
                if len(values) == 1:
                    service_id = _extract_service_id(str(values[0]))
                    if service_id is not None:
                        name = _service_name_from_id(service_id)
                        if name:
                            return name
        return None
    return _as_path(payload).stem or None


__all__ = ["_infer_service_name"]
