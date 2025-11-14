"""Aggregate pre-computed provision results into grouped city profiles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from townsnet.provision.validation import GeoDfSchema


# Public mapping of services by thematic group.
def _normalize_service_key(value: str) -> str:
    """Prepare service names for case-insensitive lookup."""
    normalized = value.strip().lower().replace("ё", "е")
    normalized = re.sub(r"[\s_\-]+", " ", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized

SERVICE_ID_TO_NAME: Dict[int, str] = {
    1: "Парк",
    21: "Детский сад",
    22: "Школа",
    23: "Дом детского творчества",
    25: "Детские лагеря",
    26: "Среднее специальное учебное заведение",
    27: "Высшее учебное заведение",
    28: "Поликлиника",
    29: "Детская поликлиника",
    30: "Стоматологическая клиника",
    31: "Фельдшерско-акушерский пункт",
    32: "Женская консультация",
    33: "Реабилитационный центр",
    34: "Аптека",
    35: "Больница",
    36: "Роддом",
    37: "Детская больница",
    38: "Хоспис",
    39: "Станция скорой медицинской помощи",
    40: "Травматологические пункты",
    41: "Морг",
    42: "Диспансер",
    44: "Дом престарелых",
    45: "Центр занятости населения",
    46: "Детские дома-интернаты",
    47: "Многофункциональные центры (МФЦ)",
    48: "Библиотека",
    49: "Дворец культуры",
    50: "Музей",
    51: "Театр",
    56: "Кинотеатр",
    57: "Торговый центр",
    58: "Аквапарк",
    59: "Стадион",
    60: "Ледовая арена",
    61: "Кафе",
    62: "Ресторан",
    63: "Бар/Паб",
    64: "Столовая",
    65: "Булочная",
    67: "Бассейн",
    68: "Спортивный зал",
    69: "Каток",
    73: "Скалодром",
    78: "Полицейский участок",
    79: "Пожарная станция",
    81: "Железнодорожный вокзал",
    86: "Автовокзал",
    88: "Выход метро",
    89: "Супермаркет",
    90: "Продукты (магазин у дома)",
    91: "Рынок",
    92: "Хозяйственные товары",
    93: "Одежда и обувь",
    94: "Бытовая техника",
    95: "Книжный магазин",
    96: "Детские товары",
    97: "Спортивный магазин",
    98: "Почтовое отделение",
    99: "Пункт выдачи",
    100: "Отделение банка",
    101: "Банкомат",
    102: "Адвокат",
    103: "Нотариальная контора",
    104: "Парикмахер",
    105: "Салон красоты",
    106: "Общественная баня",
    107: "Ветеринарная клиника",
    108: "Зоомагазин",
    110: "Гостиница",
    111: "Хостел",
    112: "База отдыха",
    113: "Памятник",
    114: "Церковь",
    143: "Санаторий",
}

SERVICE_NAME_ALIASES: Dict[str, str] = {
    _normalize_service_key(name): name for name in SERVICE_ID_TO_NAME.values()
}

SERVICE_GROUPS: Dict[str, List[str]] = {
    "Образование": [
        SERVICE_ID_TO_NAME[21],  # Детский сад
        SERVICE_ID_TO_NAME[22],  # Школа
        SERVICE_ID_TO_NAME[23],  # Дом детского творчества
        SERVICE_ID_TO_NAME[25],  # Детские лагеря
        SERVICE_ID_TO_NAME[26],  # Среднее специальное учебное заведение
        SERVICE_ID_TO_NAME[27],  # Высшее учебное заведение
    ],
    "Здравоохранение": [
        SERVICE_ID_TO_NAME[28],  # Поликлиника
        SERVICE_ID_TO_NAME[29],  # Детская поликлиника
        SERVICE_ID_TO_NAME[30],  # Стоматологическая клиника
        SERVICE_ID_TO_NAME[31],  # Фельдшерско-акушерский пункт
        SERVICE_ID_TO_NAME[32],  # Женская консультация
        SERVICE_ID_TO_NAME[33],  # Реабилитационный центр
        SERVICE_ID_TO_NAME[34],  # Аптека
        SERVICE_ID_TO_NAME[35],  # Больница
        SERVICE_ID_TO_NAME[36],  # Роддом
        SERVICE_ID_TO_NAME[37],  # Детская больница
        SERVICE_ID_TO_NAME[38],  # Хоспис
        SERVICE_ID_TO_NAME[39],  # Станция скорой медицинской помощи
        SERVICE_ID_TO_NAME[40],  # Травматологические пункты
        SERVICE_ID_TO_NAME[41],  # Морг
        SERVICE_ID_TO_NAME[42],  # Диспансер
        SERVICE_ID_TO_NAME[143], # Санаторий
    ],
    "Спорт": [
        SERVICE_ID_TO_NAME[67],  # Бассейн
        SERVICE_ID_TO_NAME[68],  # Спортивный зал
        SERVICE_ID_TO_NAME[59],  # Стадион
        SERVICE_ID_TO_NAME[60],  # Ледовая арена
        SERVICE_ID_TO_NAME[69],  # Каток
        SERVICE_ID_TO_NAME[73],  # Скалодром
    ],
    "Социальная помощь": [
        SERVICE_ID_TO_NAME[44],  # Дом престарелых
        SERVICE_ID_TO_NAME[46],  # Детские дома-интернаты
        SERVICE_ID_TO_NAME[45],  # Центр занятости населения
    ],
    "Услуги": [
        SERVICE_ID_TO_NAME[98],  # Почтовое отделение
        SERVICE_ID_TO_NAME[99],  # Пункт выдачи
        SERVICE_ID_TO_NAME[100], # Отделение банка
        SERVICE_ID_TO_NAME[101], # Банкомат
        SERVICE_ID_TO_NAME[47],  # Многофункциональные центры (МФЦ)
        SERVICE_ID_TO_NAME[102], # Адвокат
        SERVICE_ID_TO_NAME[103], # Нотариальная контора
        SERVICE_ID_TO_NAME[104], # Парикмахер
        SERVICE_ID_TO_NAME[105], # Салон красоты
        SERVICE_ID_TO_NAME[106], # Общественная баня
        SERVICE_ID_TO_NAME[107], # Ветеринарная клиника
        SERVICE_ID_TO_NAME[108], # Зоомагазин
    ],
    "Культура и отдых": [
        SERVICE_ID_TO_NAME[48],  # Библиотека
        SERVICE_ID_TO_NAME[49],  # Дворец культуры
        SERVICE_ID_TO_NAME[50],  # Музей
        SERVICE_ID_TO_NAME[51],  # Театр
        SERVICE_ID_TO_NAME[56],  # Кинотеатр
        SERVICE_ID_TO_NAME[57],  # Торговый центр
        SERVICE_ID_TO_NAME[58],  # Аквапарк
        SERVICE_ID_TO_NAME[59],  # Стадион (как объект массовых мероприятий)
        SERVICE_ID_TO_NAME[60],  # Ледовая арена
        SERVICE_ID_TO_NAME[1],   # Парк
    ],
    "Безопасность": [
        SERVICE_ID_TO_NAME[78],  # Полицейский участок
        SERVICE_ID_TO_NAME[79],  # Пожарная станция
    ],
    "Туризм": [
        SERVICE_ID_TO_NAME[110], # Гостиница
        SERVICE_ID_TO_NAME[111], # Хостел
        SERVICE_ID_TO_NAME[112], # База отдыха
        SERVICE_ID_TO_NAME[61],  # Кафе
        SERVICE_ID_TO_NAME[62],  # Ресторан
        SERVICE_ID_TO_NAME[63],  # Бар/Паб
        SERVICE_ID_TO_NAME[64],  # Столовая
        SERVICE_ID_TO_NAME[65],  # Булочная
    ],
}

_SERVICE_ID_PATTERN = re.compile(r"\d+")


def _canonical_service_name(candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    normalized = _normalize_service_key(str(candidate))
    return SERVICE_NAME_ALIASES.get(normalized)


def _extract_service_id(token: str) -> Optional[int]:
    matches = _SERVICE_ID_PATTERN.findall(token)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _service_name_from_id(service_id: int) -> Optional[str]:
    try:
        return SERVICE_ID_TO_NAME[int(service_id)]
    except (KeyError, ValueError):
        return None


def _service_name_from_path(path: Path) -> Optional[str]:
    service_id = _extract_service_id(path.stem)
    if service_id is None:
        return None
    return _service_name_from_id(service_id)


CityInfoSource = Union[str, Path, pd.DataFrame]
ServiceInput = Union[str, Path, pd.DataFrame]

REQUIRED_SERVICE_COLUMNS: Tuple[str, ...] = (
    "demand",
    "demand_within",
    "demand_without",
    "capacity",
    "capacity_left",
)


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


@dataclass
class UrbanFunctionCalculator:
    """Combine per-service results into grouped city provision profiles."""

    city_info: Optional[pd.DataFrame] = field(default=None, init=False)
    service_results: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    group_aggregates: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    service_aggregates: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False)
    city_json: Dict[int, Dict[str, object]] = field(default_factory=dict, init=False)

    # ML-related caches (optional; populated by compute_ml_features)
    ml_features: Optional[pd.DataFrame] = field(default=None, init=False)
    ml_models: Dict[str, object] = field(default_factory=dict, init=False)

    _service_results_lower: Dict[str, pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _external_supply_total: Optional[pd.Series] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Get data
    # ------------------------------------------------------------------ #

    def get_profilies(self) -> gpd.GeoDataFrame:
        rows = []
        for cid, prof in self.city_json.items():
            rows.append({"city_id": int(cid), **prof})
        df = pd.DataFrame(rows, index=self.city_info.index)
        keys = ["city_id", "Название", "Население", "Опорный пункт", "Потенциальный опорный пункт", "Лучшая градообразующая функция", "Лучшая градообразующая функция, чел", "geometry"]
        df_small = df[keys].copy()

        gdf = gpd.GeoDataFrame(
            df_small,
            geometry='geometry',
            crs="EPSG:4326",
            index=self.city_info.index,
        )
        return gdf

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_city_info(self, source: CityInfoSource) -> None:
        """Load city metadata with strict schema [name, is_city, population, geometry]."""
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        else:
            data = _read_table(_as_path(source))

        if data.empty:
            raise ValueError("City info table is empty.")
        
        data = GeoDfSchema.validate(data)
        prepared = gpd.GeoDataFrame(index=data.index, geometry=data['geometry'], crs=data.crs)
        prepared["city_name"] = data["name"].astype(str)
        prepared["is_anchor"] = data["is_city"].astype(bool)
        prepared["population"] = _ensure_numeric(data["population"]).fillna(0.0)

        self.city_info = prepared
        self.service_results.clear()
        self._service_results_lower.clear()
        self.group_aggregates.clear()
        self.service_aggregates.clear()
        self.city_json.clear()
        self._external_supply_total = None

    def load_service_results(
        self,
        services: Union[Mapping[str, ServiceInput], Sequence[ServiceInput], str, Path],
    ) -> None:
        """Load already calculated provision outputs for individual services."""
        if self.city_info is None:
            raise RuntimeError("Load city info before service results.")

        if isinstance(services, Mapping):
            raw_items: List[Tuple[Optional[str], ServiceInput]] = list(services.items())
        elif isinstance(services, (str, Path)):
            base_path = _as_path(services)
            if base_path.is_dir():
                candidates = sorted(
                    path for path in base_path.glob("*.parquet") if path.is_file()
                )
                if not candidates:
                    raise ValueError(f"No .parquet files found in directory: {base_path}")
                raw_items = [(None, candidate) for candidate in candidates]
            else:
                raw_items = [(None, base_path)]
        else:
            raw_items = [(_infer_service_name(item), item) for item in services]

        self.service_results.clear()
        self._service_results_lower.clear()

        loaded = 0
        skipped: List[str] = []
        for provided_name, payload in raw_items:
            canonical_name = self._resolve_service_name(provided_name, payload)
            if canonical_name is None:
                if isinstance(payload, pd.DataFrame):
                    skipped.append("<dataframe>")
                else:
                    skipped.append(str(_as_path(payload)))
                continue
            frame = self._load_service_frame(payload)
            prepared = self._prepare_service_frame(frame)
            self.service_results[canonical_name] = prepared
            self._service_results_lower[_normalize_service_key(canonical_name)] = prepared
            loaded += 1

        if loaded == 0:
            if skipped:
                skipped_display = ", ".join(skipped)
                raise ValueError(
                    f"No service results were loaded. Skipped unsupported services: {skipped_display}"
                )
            raise ValueError("No service results were loaded.")

        self.group_aggregates.clear()
        self.service_aggregates.clear()
        self.city_json.clear()
        self._external_supply_total = None

    # ------------------------------------------------------------------ #
    # Aggregation & export
    # ------------------------------------------------------------------ #

    def build_profiles(self, min_export_threshold: int = 0) -> Dict[int, Dict[str, object]]:
        """
        Aggregate service-level results into grouped city profiles with urban function metrics.

        This method performs the full pipeline:
        - Groups individual services into thematic categories (e.g., Education, Healthcare)
        - Computes aggregated metrics per group and service
        - Calculates key urban functions:
            * City self-sufficiency (градообслуживающая функция)
            * Service export to other cities (градообразующая функция)
        - Identifies best-performing groups in both dimensions
        - Flags potential anchor cities based on service export above the threshold
        - Assembles a structured JSON-ready dictionary of city profiles

        Args:
            min_export_threshold (int): Minimum number of people served from other cities 
                                    required to classify a non-anchor city as a 
                                    potential anchor point. Default is 0.

        Returns:
            Dict[int, Dict[str, object]]: A dictionary mapping city IDs to their complete 
                                        profiles containing demographic data, provision 
                                        metrics, best-performing functions, and classification.

        Raises:
            RuntimeError: If city info or service results have not been loaded prior to calling.
            RuntimeError: If none of the configured service groups match available data.
        """
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")
        if not self.service_results:
            raise RuntimeError("Service results are not loaded.")

        external_supply = pd.Series(0.0, index=self.city_info.index, dtype=float)
        aggregates: Dict[str, pd.DataFrame] = {}
        has_matching_service = False

        for group_name, service_names in SERVICE_GROUPS.items():
            if any(
                service_name in self.service_results
                or service_name.lower() in self._service_results_lower
                for service_name in service_names
            ):
                has_matching_service = True
            aggregates[group_name] = self._aggregate_group(group_name, service_names, external_supply)

        if not has_matching_service:
            raise RuntimeError("No services match the configured service groups.")

        self.group_aggregates = aggregates
        self.service_aggregates = self._build_service_metrics()
        self._external_supply_total = external_supply
        self.city_json = self._assemble_city_json(min_export_threshold=min_export_threshold)
        return self.city_json

    # ------------------------------------------------------------------ #
    # ML features (optional)
    # ------------------------------------------------------------------ #

    def _build_city_feature_matrix(self) -> pd.DataFrame:
        """Assemble wide city x service feature matrix for ML.

        Uses per-service aggregates: for each service, takes provision_pct and external_pct.
        Missing values are filled with 0.0 and rows are aligned to city_info index.
        """
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")

        if not self.service_aggregates:
            # Ensure service metrics are available
            self.service_aggregates = self._build_service_metrics()

        parts: List[pd.DataFrame] = []
        for service_name, df in self.service_aggregates.items():
            sub = pd.DataFrame(
                {
                    f"{service_name}__provision_pct": pd.to_numeric(df["provision_pct"], errors="coerce"),
                    f"{service_name}__external_pct": pd.to_numeric(df["external_pct"], errors="coerce"),
                },
                index=df.index,
            )
            parts.append(sub)

        if not parts:
            raise RuntimeError("No service aggregates available to build ML features.")

        features = pd.concat(parts, axis=1).reindex(self.city_info.index).fillna(0.0)
        return features

    def compute_ml_features(
        self,
        *,
        n_clusters: int = 5,
        components: int = 2,
        with_anomaly: bool = True,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Compute optional ML features per city.

        - Standardizes features and applies PCA to get embeddings (pca1..pcaN).
        - Builds a composite index from PC1 scaled to [0, 100].
        - Clusters cities with KMeans (cluster labels as ints).
        - Optionally, scores anomalies via IsolationForest (higher = more anomalous).

        Returns the features DataFrame and stores it in self.ml_features.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
            import numpy as _np  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "scikit-learn is required for ML features (pip install scikit-learn)."
            ) from exc

        X = self._build_city_feature_matrix()
        X = pd.concat([X, self.city_info['population']], axis=1)
        self.X = X
        if X.shape[0] < 2 or X.shape[1] < 2:
            raise RuntimeError("Not enough data to compute ML features.")

        scaler = StandardScaler()
        Z = scaler.fit_transform(X.values)
        self.Z = Z

        n_comp = int(max(1, min(int(components), X.shape[1])))
        pca = PCA(n_components=n_comp, random_state=random_state)
        emb = pca.fit_transform(Z)

        pca_cols = [f"pca{i+1}" for i in range(n_comp)]
        df_ml = pd.DataFrame(emb, index=X.index, columns=pca_cols)

        pc1 = df_ml[pca_cols[0]].values
        pc1_min = float(_np.min(pc1))
        pc1_ptp = float(_np.ptp(pc1)) if float(_np.ptp(pc1)) > 0 else 1.0
        df_ml["index"] = (pc1 - pc1_min) / pc1_ptp * 100.0

        km = KMeans(n_clusters=int(max(2, n_clusters)), n_init=10, random_state=random_state)
        clusters = km.fit_predict(Z)
        df_ml["cluster"] = clusters.astype(int)

        models: Dict[str, object] = {"scaler": scaler, "pca": pca, "kmeans": km}

        if with_anomaly and X.shape[0] >= 10:
            iso = IsolationForest(random_state=random_state, contamination="auto")
            iso.fit(Z)
            # decision_function: higher -> more normal, lower -> more anomalous; invert
            df_ml["anomaly_score"] = (-iso.decision_function(Z)).astype(float)
            models["isolation_forest"] = iso

        self.ml_features = pd.concat([self.city_info, df_ml], axis=1)
        self.ml_models = models
        return df_ml

    def save_city_json(self, path: Union[str, Path], *, by: str = "id") -> None:
        """Save the assembled profiles to disk."""
        if not self.city_json:
            self.build_profiles()

        if by == "name":
            payload: MutableMapping[str, Dict[str, object]] = {
                profile["Название"]: profile for profile in self.city_json.values()
            }
        elif by == "id":
            payload = {str(city_id): profile for city_id, profile in self.city_json.items()}
        else:
            raise ValueError("Parameter 'by' must be either 'id' or 'name'.")

        target = _as_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_service_frame(self, payload: ServiceInput) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        else:
            frame = _read_table(_as_path(payload))
        if frame.empty:
            raise ValueError("Service result table is empty.")
        return _drop_geometry(frame)

    def _resolve_service_name(
        self,
        provided_name: Optional[str],
        payload: ServiceInput,
    ) -> Optional[str]:
        candidates: List[str] = []
        if provided_name:
            candidates.append(str(provided_name))
        inferred = _infer_service_name(payload)
        if inferred:
            candidates.append(str(inferred))
        if not isinstance(payload, pd.DataFrame):
            path = _as_path(payload)
            candidates.extend([path.stem, path.name])

        for candidate in candidates:
            canonical = _canonical_service_name(candidate)
            if canonical:
                return canonical
            service_id = _extract_service_id(candidate)
            if service_id is not None:
                mapped = _service_name_from_id(service_id)
                if mapped:
                    return mapped

        if isinstance(payload, pd.DataFrame):
            columns = {str(col).strip().lower(): col for col in payload.columns}
            for key in ("service_id", "service_code"):
                if key in columns:
                    values = payload[columns[key]].dropna().unique()
                    if len(values) == 1:
                        service_id = _extract_service_id(str(values[0]))
                        if service_id is not None:
                            mapped = _service_name_from_id(service_id)
                            if mapped:
                                return mapped

        if not isinstance(payload, pd.DataFrame):
            from_path = _service_name_from_path(_as_path(payload))
            if from_path:
                return from_path

        return None

    def _prepare_service_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.city_info is None:
            raise RuntimeError("City info must be loaded before services.")

        frame.columns = [str(col).strip() for col in frame.columns]
        lower_map = {col.lower(): col for col in frame.columns}

        if "city_id" in lower_map:
            frame = frame.set_index(lower_map["city_id"])
        elif "id" in lower_map:
            frame = frame.set_index(lower_map["id"])
        elif "town_id" in lower_map:
            frame = frame.set_index(lower_map["town_id"])
        elif frame.index.name and str(frame.index.name).lower() in {"city_id", "id", "town_id"}:
            pass
        else:
            frame.index.name = "city_id"

        frame.index = pd.Index(pd.to_numeric(frame.index, errors="coerce"), name="city_id")
        if frame.index.hasnans:
            raise ValueError("Service result index must be numeric (city ids).")

        numeric: Dict[str, pd.Series] = {}
        for column in REQUIRED_SERVICE_COLUMNS + ("population",):
            if column in frame.columns:
                numeric[column] = _ensure_numeric(frame[column])

        for column in REQUIRED_SERVICE_COLUMNS:
            if column not in numeric:
                raise KeyError(f"Service result must contain column '{column}'.")

        prepared = pd.DataFrame(numeric, index=frame.index)
        prepared = prepared.reindex(self.city_info.index).fillna(0.0)

        if "population" in prepared.columns:
            self._update_population(prepared["population"])

        return prepared

    def _update_population(self, population: pd.Series) -> None:
        if self.city_info is None:
            return
        aligned = population.reindex(self.city_info.index).fillna(0.0)
        mask = (self.city_info["population"] <= 0) & (aligned > 0)
        if mask.any():
            self.city_info.loc[mask, "population"] = aligned[mask]

    def _get_service_frame(self, service_name: str) -> Optional[pd.DataFrame]:
        frame = self.service_results.get(service_name)
        if frame is not None:
            return frame
        return self._service_results_lower.get(_normalize_service_key(service_name))

    def _build_service_metrics(self) -> Dict[str, pd.DataFrame]:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")

        city_ids = self.city_info.index
        metrics: Dict[str, pd.DataFrame] = {}

        seen_services: set[str] = set()

        for service_names in SERVICE_GROUPS.values():
            for service_name in service_names:
                if service_name in seen_services:
                    continue
                seen_services.add(service_name)

                frame = self._get_service_frame(service_name)
                if frame is None:
                    continue

                aligned = frame.reindex(city_ids).fillna(0.0)
                demand = aligned["demand"]
                served = aligned["demand_within"]
                external = aligned["demand_without"]

                mask = demand > 0
                # Default to 0.0 so missing demand doesn't produce NaN -> null in JSON
                provision_pct = pd.Series(0.0, index=city_ids, dtype=float)
                provision_pct.loc[mask] = (
                    (served.loc[mask] / demand.loc[mask]).clip(0.0, 1.0) * 100.0
                )

                external_pct = pd.Series(0.0, index=city_ids, dtype=float)
                external_pct.loc[mask] = (
                    (external.loc[mask] / demand.loc[mask]).clip(lower=0.0) * 100.0
                )

                metrics[service_name] = pd.DataFrame(
                    {
                        "provision_pct": provision_pct,
                        "served_population": served,
                        "external_demand": external,
                        "external_pct": external_pct,
                    },
                    index=city_ids,
                )

        return metrics

    def _aggregate_group(
        self,
        group_name: str,
        service_names: Sequence[str],
        external_supply_acc: pd.Series,
    ) -> pd.DataFrame:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")

        city_ids = self.city_info.index
        demand = pd.Series(0.0, index=city_ids, dtype=float)
        served = pd.Series(0.0, index=city_ids, dtype=float)
        external_demand = pd.Series(0.0, index=city_ids, dtype=float)
        capacity_used = pd.Series(0.0, index=city_ids, dtype=float)
        supplied_to_others_group = pd.Series(0.0, index=city_ids, dtype=float)
        has_data = False

        for service_name in service_names:
            frame = self._get_service_frame(service_name)
            if frame is None:
                continue

            has_data = True
            aligned = frame.reindex(city_ids).fillna(0.0)
            demand += aligned["demand"]
            served += aligned["demand_within"]
            external_demand += aligned["demand_without"]

            used_capacity = aligned["capacity"] - aligned["capacity_left"]
            capacity_used += used_capacity

            # --- новая логика обеспечения других городов ---
            capacity_after_self = aligned["capacity"] - aligned["demand"]

            # город может обеспечить других (есть запас)
            can_supply = capacity_after_self > 0

            # и реально обеспечивает (остаток меньше потенциального запаса)
            actually_supplies = aligned["capacity_left"] < capacity_after_self

            mask = can_supply & actually_supplies

            supplied_to_others = pd.Series(0.0, index=aligned.index, dtype=float)
            supplied_to_others[mask] = (
                capacity_after_self[mask] - aligned.loc[mask, "capacity_left"]
            ).clip(lower=0.0)

            supplied_to_others_group += supplied_to_others
            external_supply_acc += supplied_to_others


        result = pd.DataFrame(
            {
                "city_id": city_ids,
                "group_name": group_name,
                "demand": demand,
                "served": served,
                "external_demand": external_demand,
                "supplied_to_others": supplied_to_others_group,
            }
        ).set_index("city_id")

        if not has_data:
            # No underlying services matched this group; treat as zeros so
            # the group participates in averages instead of being skipped.
            result.loc[
                :,
                [
                    "demand",
                    "served",
                    "external_demand",
                    "provision_pct",
                    "served_population",
                    "external_pct",
                    "capacity_used",
                ],
            ] = 0.0
            return result

        mask = demand > 0

        result["provision_pct"] = 0.0
        result.loc[mask, "provision_pct"] = (
            (served.loc[mask] / demand.loc[mask]).clip(0.0, 1.0) * 100.0
        )

        result["served_population"] = served
        result["external_pct"] = 0.0
        result.loc[mask, "external_pct"] = (
            (external_demand.loc[mask] / demand.loc[mask]).clip(lower=0.0) * 100.0
        )

        result["capacity_used"] = capacity_used
        return result

    def _assemble_city_json(self, min_export_threshold: int = 0) -> Dict[int, Dict[str, object]]:
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")
        if not self.group_aggregates:
            raise RuntimeError("Group aggregates are not available.")
        if not self.service_aggregates:
            raise RuntimeError("Service aggregates are not available.")

        profiles: Dict[int, Dict[str, object]] = {}
        if self._external_supply_total is None:
            external_supply = pd.Series(0.0, index=self.city_info.index, dtype=float)
        else:
            external_supply = self._external_supply_total.reindex(self.city_info.index).fillna(0.0)

        def _as_optional_float(value: object) -> Optional[float]:
            if value is None:
                return None
            try:
                if pd.isna(value):  # type: ignore[arg-type]
                    return None
            except TypeError:
                pass
            return float(value)

        def _round_optional(value: Optional[float], digits: int = 2) -> Optional[float]:
            if value is None:
                return None
            return round(float(value), digits)

        def _int_optional(value: Optional[float]) -> Optional[int]:
            if value is None:
                return None
            return int(round(value))

        for city_id, city_row in self.city_info.iterrows():
            group_provision: Dict[str, Dict[str, object]] = {}
            group_mobility: Dict[str, Dict[str, object]] = {}
            service_provision: Dict[str, Dict[str, object]] = {}
            service_mobility: Dict[str, Dict[str, object]] = {}

            top_group = None
            top_value = 0.0
            has_served_group = False

            top_mobility_group = None
            top_mobility_value = 0.0
            has_mobility_group = False

            for group_name, service_names in SERVICE_GROUPS.items():
                group_df = self.group_aggregates.get(group_name)
                if group_df is not None and city_id in group_df.index:
                    metrics = group_df.loc[city_id]
                    provision_pct = _as_optional_float(metrics.get("provision_pct"))
                    served_population = _as_optional_float(metrics.get("served_population"))
                    external_demand = _as_optional_float(metrics.get("external_demand"))
                    external_pct = _as_optional_float(metrics.get("external_pct"))
                    supplied_to_others = _as_optional_float(metrics.get("supplied_to_others"))
                else:
                    provision_pct = None
                    served_population = None
                    external_demand = None
                    external_pct = None

                group_provision[group_name] = {
                    "Обеспеченность, %": _round_optional(provision_pct),
                    "Обслуженное население": _int_optional(served_population),
                }
                group_mobility[group_name] = {
                    "Обслуженное население": _round_optional(supplied_to_others),
                }

                if supplied_to_others is not None and supplied_to_others > 0:
                    has_mobility_group = True
                    if supplied_to_others > top_mobility_value or top_mobility_group is None:
                        top_mobility_value = supplied_to_others
                        top_mobility_group = group_name

                if served_population is not None and served_population > 0:
                    has_served_group = True
                    if provision_pct is not None and (provision_pct > top_value or top_group is None):
                        top_value = provision_pct
                        top_group = group_name

            seen_services: set[str] = set()
            for service_names in SERVICE_GROUPS.values():
                for service_name in service_names:
                    if service_name in seen_services:
                        continue
                    seen_services.add(service_name)

                    metrics_df = self.service_aggregates.get(service_name)
                    if metrics_df is not None and city_id in metrics_df.index:
                        metrics = metrics_df.loc[city_id]
                        provision_pct = _as_optional_float(metrics.get("provision_pct"))
                        served_population = _as_optional_float(metrics.get("served_population"))
                        external_demand = _as_optional_float(metrics.get("external_demand"))
                        external_pct = _as_optional_float(metrics.get("external_pct"))
                    else:
                        # If service metrics are missing for this city, use zeros so
                        # the UI shows 0 instead of null and averages include it.
                        provision_pct = 0.0
                        served_population = 0.0
                        external_demand = 0.0
                        external_pct = 0.0

                    service_provision[service_name] = {
                        "Обеспеченность, %": _round_optional(provision_pct),
                        "Обслуженное население": _int_optional(served_population),
                    }
                    service_frame = self._get_service_frame(service_name)
                    if service_frame is not None and city_id in service_frame.index:
                        row = service_frame.loc[city_id]
                        capacity_after_self = row["capacity"] - row["demand"]
                        can_supply = capacity_after_self > 0
                        actually_supplies = row["capacity_left"] < capacity_after_self
                        export_value = max(0.0, capacity_after_self - row["capacity_left"]) if (can_supply and actually_supplies) else 0.0
                    else:
                        export_value = 0.0

                    service_mobility[service_name] = {
                        "Обслуженное население": _round_optional(export_value)
                    }

            best_provision = _round_optional(top_value) if has_served_group else None
            best_group = top_group if has_served_group else None
            best_mobility_pct = _round_optional(top_mobility_value) if has_mobility_group else None
            best_mobility_group = top_mobility_group if has_mobility_group else None


            population = int(round(float(city_row.get("population", 0.0) or 0.0)))
            is_anchor = bool(city_row.get("is_anchor", False))
            # Город считается потенциальным опорным, только если:
            #   - он сам не помечен как опорный;
            #   - он действительно обслуживает другие города (есть внешний объём);
            #   - у него есть хотя бы одна группа, где обслуживается своё население.
            potential_anchor = bool(
                (not is_anchor)
                and external_supply.get(city_id, 0.0) > min_export_threshold
                and has_served_group
            )
            geometry = city_row.get("geometry")

            profiles[int(city_id)] = {
                "Название": str(city_row.get("city_name", city_id)),
                "geometry": geometry,
                "Опорный пункт": is_anchor,
                "Потенциальный опорный пункт": potential_anchor,
                "Население": population,
                "Лучшая обеспеченность, %": best_provision,
                "Сервисы: градообслуживающая функция": service_provision,
                "Сервисы: градообразующая функция": service_mobility,
                "Градообслуживающие функции": group_provision,
                "Градообразующие функции": group_mobility,
                "Лучшая градообслуживающая функция": best_group,
                "Лучшая градообслуживающая функция, %": best_provision,
                "Лучшая градообразующая функция": best_mobility_group,
                "Лучшая градообразующая функция, чел": _round_optional(top_mobility_value),
            }

        # Attach optional ML block per city if available
        if getattr(self, "ml_features", None) is not None:
            try:
                ml_index = set(self.ml_features.index)  # type: ignore[union-attr]
            except Exception:
                ml_index = set()
            for cid in list(profiles.keys()):
                if cid in ml_index:  # type: ignore[operator]
                    row = self.ml_features.loc[cid]  # type: ignore[index]
                    ml_block: Dict[str, object] = {
                        "cluster": int(row.get("cluster")) if "cluster" in row else None,
                        "index": float(row.get("index")) if "index" in row else None,
                        "embedding": [
                            float(row.get("pca1")) if "pca1" in row else None,
                            float(row.get("pca2")) if "pca2" in row else None,
                        ],
                    }
                    if "anomaly_score" in row:
                        ml_block["anomaly"] = float(row.get("anomaly_score"))
                    profiles[cid]["ML"] = ml_block

        return profiles


def _infer_service_name(payload: ServiceInput) -> Optional[str]:
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


__all__ = ["UrbanFunctionCalculator", "SERVICE_GROUPS", "SERVICE_ID_TO_NAME"]
