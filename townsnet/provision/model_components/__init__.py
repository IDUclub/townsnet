"""Shared components for townsnet provision model."""

from .service_catalog import (
    SERVICE_GROUPS,
    SERVICE_ID_TO_NAME,
    SERVICE_NAME_ALIASES,
    _canonical_service_name,
    _extract_service_id,
    _normalize_service_key,
    _service_name_from_id,
    _service_name_from_path,
)
from .io_utils import _as_path, _drop_geometry, _ensure_numeric, _read_table
from .service_resolver import _infer_service_name
from .types import CityInfoSource, ServiceInput, REQUIRED_SERVICE_COLUMNS

__all__ = [
    "SERVICE_GROUPS",
    "SERVICE_ID_TO_NAME",
    "SERVICE_NAME_ALIASES",
    "_canonical_service_name",
    "_extract_service_id",
    "_normalize_service_key",
    "_service_name_from_id",
    "_service_name_from_path",
    "_as_path",
    "_drop_geometry",
    "_ensure_numeric",
    "_read_table",
    "_infer_service_name",
    "CityInfoSource",
    "ServiceInput",
    "REQUIRED_SERVICE_COLUMNS",
]
