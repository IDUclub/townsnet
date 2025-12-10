"""Aggregate pre-computed provision results into grouped city profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from townsnet.provision.model_components import (
    SERVICE_GROUPS,
    SERVICE_ID_TO_NAME,
    CityInfoSource,
    REQUIRED_SERVICE_COLUMNS,
    ServiceInput,
    _as_path,
    _canonical_service_name,
    _drop_geometry,
    _ensure_numeric,
    _extract_service_id,
    _infer_service_name,
    _normalize_service_key,
    _read_table,
    _service_name_from_id,
    _service_name_from_path,
)
from townsnet.provision.validation import GeoDfSchema


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
        keys = [
            "city_id",
            "Название",
            "Население",
            "Опорный пункт",
            "Потенциальный опорный пункт",
            "Лучшая градообразующая функция",
            "Лучшая градообразующая функция, чел",
            "geometry",
        ]
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
        """Load city metadata with strict schema [name, is_anchor, population, geometry]."""
        if isinstance(source, pd.DataFrame):
            data = source.copy()
        else:
            data = _read_table(_as_path(source))

        if data.empty:
            raise ValueError("City info table is empty.")
        
        data = GeoDfSchema.validate(data)
        prepared = gpd.GeoDataFrame(index=data.index, geometry=data['geometry'], crs=data.crs)
        prepared["city_name"] = data["name"].astype(str)
        prepared["is_anchor"] = data["is_anchor"].astype(bool)
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

    def build_profiles(self, min_export_threshold: int = 1000) -> Dict[int, Dict[str, object]]:
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
        """
        Assemble wide city x service feature matrix for ML.
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
                    f"{service_name}__self_supply_pct": pd.to_numeric(df["self_supply_pct"], errors="coerce"),
                },
                index=df.index,
            )
            parts.append(sub)

        if not parts:
            raise RuntimeError("No service aggregates available to build ML features.")

        features = pd.concat(parts, axis=1).reindex(self.city_info.index).fillna(0.0)
        features = features.fillna(0.0)
        features = features.loc[:, ~(features == 0.0).all()]
        return features

    def compute_ml_features(
        self,
        *,
        n_clusters: int = 5,
        components: int = 2,
        with_anomaly: bool = True,
        random_state: int = 0,
    ) -> pd.DataFrame:
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest
            import numpy as np
        except Exception as exc:
            raise RuntimeError(
                "scikit-learn is required (pip install scikit-learn)."
            ) from exc

        self.city_feature_matrix = self._build_city_feature_matrix()
        if self.city_feature_matrix.shape[0] < 2 or self.city_feature_matrix.shape[1] < 2:
            raise RuntimeError("Not enough data to compute ML features.")

        scaler = StandardScaler()
        Z = scaler.fit_transform(self.city_feature_matrix.values)
        self.Z = Z

        n_comp = max(1, min(components, self.city_feature_matrix.shape[1]))
        pca = PCA(n_components=n_comp, random_state=random_state)
        emb = pca.fit_transform(Z)

        pca_cols = [f"pca{i+1}" for i in range(n_comp)]
        df_ml = pd.DataFrame(emb, index=self.city_feature_matrix.index, columns=pca_cols)

        pc1 = df_ml[pca_cols[0]].values
        pc1_min = float(np.min(pc1))
        pc1_ptp = float(np.ptp(pc1)) if np.ptp(pc1) > 0 else 1.0
        df_ml["service_provision_index"] = (pc1 - pc1_min) / pc1_ptp * 100.0

        km = KMeans(n_clusters=max(2, n_clusters), n_init=10, random_state=random_state)
        clusters = km.fit_predict(Z)
        df_ml["cluster"] = clusters.astype(int)

        models: dict = {"scaler": scaler, "pca": pca, "kmeans": km}

        if with_anomaly and self.city_feature_matrix.shape[0] >= 10:
            anomaly_scores_local = np.zeros(self.city_feature_matrix.shape[0])

            for cluster_id in np.unique(clusters):
                mask = (clusters == cluster_id)
                Z_cluster = Z[mask]

                # Пропускаем маленькие кластеры (<10) или используем нейтральное значение
                if len(Z_cluster) < 10:
                    anomaly_scores_local[mask] = 0.0
                    continue

                iso = IsolationForest(random_state=random_state, contamination="auto")
                iso.fit(Z_cluster)
                raw_scores = -iso.decision_function(Z_cluster)  # Чем выше — тем аномальнее
                anomaly_scores_local[mask] = raw_scores

            df_ml["anomaly_score_in_cluster"] = anomaly_scores_local.astype(float)
            models["isolation_forest_per_cluster"] = iso

        self.ml_features = pd.concat([self.city_info, df_ml], axis=1)
        self.ml_models = models
        return self.ml_features

    def analyze_cluster(
        self,
        cluster_id: int,
        *,
        anomaly_quantile: float = 0.95,
        pop_provision_quantile: float = 0.95,
        pop_anomaly_quantile: float = 0.95,
        typical_provision_quantile: float = 0.1,
        figsize: Tuple[int, int] = (14, 6),
        show_plots: bool = False,
        show_map: bool = False,
    ) -> Dict[str, object]:
        """
        Compare anomalous vs. typical cities inside one cluster.

        Visual artefacts (bar charts and folium map) are created only when the
        matching flags are enabled so the method stays API-friendly.

        Returns:
            Dict[str, object]: Detailed analysis including pandas objects and
                a JSON-ready `api_payload` for easy serialization.
        """
        if self.ml_features is None:
            raise RuntimeError("Compute ML features before analyzing clusters.")
        if self.city_info is None:
            raise RuntimeError("City info is not loaded.")
        if "cluster" not in self.ml_features:
            raise RuntimeError("Cluster labels are missing in ML features.")
        if "anomaly_score_in_cluster" not in self.ml_features:
            raise RuntimeError(
                "Anomaly scores are missing; run compute_ml_features(with_anomaly=True)."
            )

        if not self.service_aggregates:
            if self.service_results:
                self.service_aggregates = self._build_service_metrics()
            else:
                raise RuntimeError("Service results are required to analyze clusters.")

        if (not self.group_aggregates) or (not self.city_json) or (self._external_supply_total is None):
            if self.service_results:
                self.build_profiles()
            else:
                raise RuntimeError("Service results are required to analyze clusters.")

        ml_data = self.ml_features
        cluster_mask = ml_data["cluster"] == cluster_id
        if not bool(cluster_mask.any()):
            raise ValueError(f"Cluster {cluster_id} not found or empty.")

        if self._external_supply_total is None:
            external_supply = pd.Series(0.0, index=ml_data.index, dtype=float)
        else:
            external_supply = self._external_supply_total.reindex(ml_data.index).fillna(0.0)

        scores = ml_data.loc[cluster_mask, "anomaly_score_in_cluster"]
        threshold = float(scores.quantile(anomaly_quantile))

        high_anomaly = ml_data.loc[cluster_mask & (ml_data["anomaly_score_in_cluster"] >= threshold)]
        low_anomaly = ml_data.loc[cluster_mask & (ml_data["anomaly_score_in_cluster"] < threshold)]

        idx_high = high_anomaly.index.tolist()
        idx_low = low_anomaly.index.tolist()

        def collect_service_metrics(indices: List[int]) -> pd.DataFrame:
            parts: List[pd.DataFrame] = []
            keys: List[str] = []
            for group_name, df in self.service_aggregates.items():
                subset = df[["provision_pct", "self_supply_pct"]].loc[
                    df.index.intersection(indices)
                ]
                if not subset.empty:
                    parts.append(subset)
                    keys.append(group_name)
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, keys=keys, names=["services", "city_id"])

        metrics_high = collect_service_metrics(idx_high)
        metrics_low = collect_service_metrics(idx_low)

        wide_high = metrics_high.unstack("services") if not metrics_high.empty else pd.DataFrame()
        wide_low = metrics_low.unstack("services") if not metrics_low.empty else pd.DataFrame()

        summary = pd.DataFrame(
            {
                "city_name": self.city_info.loc[cluster_mask, "city_name"],
                "cluster": ml_data.loc[cluster_mask, "cluster"],
                "anomaly_score": ml_data.loc[cluster_mask, "anomaly_score_in_cluster"],
                "service_provision_index": ml_data.loc[cluster_mask, "service_provision_index"],
                "external_supply": external_supply.loc[cluster_mask],
                "population": self.city_info.loc[cluster_mask, "population"],
                "is_anchor": self.city_info.loc[cluster_mask, "is_anchor"],
            }
        ).sort_values("anomaly_score", ascending=False)

        potential_anchor_keys = (
            "Потенциальный опорный пункт",
            "Потенциальный опорный пункт (legacy)",
        )
        sample_profile = next(iter(self.city_json.values()), {})
        potential_anchor_key = next(
            (key for key in potential_anchor_keys if key in sample_profile), potential_anchor_keys[-1]
        )
        analytical_ids = [
            cid for cid, prof in self.city_json.items() if bool(prof.get(potential_anchor_key))
        ]
        summary["analytical_pop"] = summary.index.isin(analytical_ids)
        analytical_gdf = self.get_profilies().loc[analytical_ids]

        diff_provision: Optional[pd.Series] = None
        diff_self_supply: Optional[pd.Series] = None
        if not metrics_high.empty and not metrics_low.empty:
            diff_provision = (
                metrics_high.groupby("services")["provision_pct"].mean()
                - metrics_low.groupby("services")["provision_pct"].mean()
            ).sort_values()

            diff_self_supply = (
                metrics_high.groupby("services")["self_supply_pct"].mean()
                - metrics_low.groupby("services")["self_supply_pct"].mean()
            ).sort_values()

        anomalies: List[Dict[str, object]] = []
        if self.group_aggregates:
            cluster_mean = summary[["external_supply", "service_provision_index"]].mean()
            for city_id in idx_high:
                best_services: List[Tuple[str, float]] = []
                for group_name, df in self.group_aggregates.items():
                    if city_id in df.index:
                        best_services.append((group_name, float(df.loc[city_id, "provision_pct"])))
                best_services.sort(key=lambda x: x[1], reverse=True)

                anomalies.append(
                    {
                        "city_id": int(city_id),
                        "city_name": str(summary.loc[city_id, "city_name"]),
                        "anomaly_score": float(summary.loc[city_id, "anomaly_score"]),
                        "external_supply": float(summary.loc[city_id, "external_supply"]),
                        "service_provision_index": float(summary.loc[city_id, "service_provision_index"]),
                        "is_anchor": bool(summary.loc[city_id, "is_anchor"]),
                        "analytical_pop": bool(summary.loc[city_id, "analytical_pop"]),
                        "cluster_mean_external_supply": float(cluster_mean["external_supply"]),
                        "cluster_mean_service_provision_index": float(cluster_mean["service_provision_index"]),
                        "top_service_groups": [
                            {"group": g, "provision_pct": float(pct)} for g, pct in best_services[:3]
                        ],
                    }
                )

        ml_pops = ml_data[
            (external_supply >= external_supply.quantile(pop_provision_quantile))
            & (ml_data["service_provision_index"] >= ml_data["service_provision_index"].quantile(pop_provision_quantile))
            & (ml_data["anomaly_score_in_cluster"] >= ml_data["anomaly_score_in_cluster"].quantile(pop_anomaly_quantile))
            & (~ml_data["is_anchor"])
        ]

        typical_low_cities: List[Dict[str, object]] = []
        if len(low_anomaly) > 0:
            provision_threshold = float(
                low_anomaly["service_provision_index"].quantile(typical_provision_quantile)
            )
            typical_worst = low_anomaly[
                low_anomaly["service_provision_index"] <= provision_threshold
            ].sort_values("service_provision_index")
            for city_id, row in typical_worst.iterrows():
                typical_low_cities.append(
                    {
                        "city_id": int(city_id),
                        "city_name": str(summary.loc[city_id, "city_name"]),
                        "anomaly_score": float(row["anomaly_score_in_cluster"]),
                        "service_provision_index": float(row["service_provision_index"]),
                        "threshold": provision_threshold,
                    }
                )

        common = set(analytical_ids).intersection(set(ml_pops.index))
        only_analytical = set(analytical_ids).difference(set(ml_pops.index))
        only_ml = set(ml_pops.index).difference(set(analytical_ids))

        fig = None
        if show_plots and diff_provision is not None and diff_self_supply is not None:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 2, figsize=figsize)
                diff_provision.plot(kind="barh", ax=ax[0], color="salmon", edgecolor="k", alpha=0.8)
                ax[0].set_title(f"Δ provision_pct\n(High quantile - Low quantile), q={anomaly_quantile}")
                ax[0].axvline(0, color="red", linestyle="--", linewidth=1)
                ax[0].grid(axis="x", alpha=0.3)
                ax[0].set_xlim(-100, 100)
                ax[0].set_xlabel("Разница (%)")

                diff_self_supply.plot(kind="barh", ax=ax[1], color="skyblue", edgecolor="k", alpha=0.8)
                ax[1].set_title(f"Δ self_supply_pct\n(High quantile - Low quantile), q={anomaly_quantile}")
                ax[1].axvline(0, color="red", linestyle="--", linewidth=1)
                ax[1].grid(axis="x", alpha=0.3)
                ax[1].set_xlim(-100, 100)
                ax[1].set_xlabel("Разница (%)")

                plt.tight_layout()
                plt.show()
            except ImportError:
                fig = None

        map_widget = None
        if show_map:
            try:
                import folium
                map_widget = analytical_gdf.explore(
                    color="#ff00007a",
                    name="Аналитический ПОП",
                    tooltip=["Название"],
                    tiles="CartoDB positron",
                )
                if len(ml_pops) > 0:
                    gpd.GeoDataFrame(ml_pops, geometry=ml_pops.geometry, crs=analytical_gdf.crs).explore(
                        color="orange", marker_kwds={"radius": 6}, name="ML ПОП", m=map_widget
                    )
                if len(common) > 0:
                    analytical_gdf.loc[list(common)].explore(
                        color="green", marker_kwds={"radius": 8}, name="Совпадения", m=map_widget
                    )
                folium.LayerControl().add_to(map_widget)
            except ImportError:
                map_widget = None

        summary_records = summary.reset_index().rename(columns={"index": "city_id"}).to_dict(orient="records")
        api_payload = {
            "cluster_id": int(cluster_id),
            "anomaly_threshold": threshold,
            "counts": {
                "total": int(cluster_mask.sum()),
                "high_anomaly": len(idx_high),
                "low_anomaly": len(idx_low),
            },
            "high_anomaly_city_ids": [int(x) for x in idx_high],
            "low_anomaly_city_ids": [int(x) for x in idx_low],
            "diff_provision_pct": diff_provision.round(2).to_dict() if diff_provision is not None else {},
            "diff_self_supply_pct": diff_self_supply.round(2).to_dict() if diff_self_supply is not None else {},
            "summary": summary_records,
            "anomalies": anomalies,
            "typical_low_provision": typical_low_cities,
            "ml_vs_analytical": {
                "common": [int(x) for x in common],
                "only_analytical": [int(x) for x in only_analytical],
                "only_ml": [int(x) for x in only_ml],
            },
        }

        return {
            "summary": summary,
            "metrics_high": metrics_high,
            "metrics_low": metrics_low,
            "stats_high": wide_high.describe().round(2) if not wide_high.empty else None,
            "stats_low": wide_low.describe().round(2) if not wide_low.empty else None,
            "diff_provision": diff_provision,
            "diff_self_supply": diff_self_supply,
            "ml_pops": ml_pops,
            "analytical_pops": analytical_gdf,
            "anomalies": anomalies,
            "typical_low_provision": typical_low_cities,
            "figure": fig,
            "map": map_widget,
            "api_payload": api_payload,
        }

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
                capacity = aligned["capacity"]
                capacity_left = aligned["capacity_left"]
                exported_to_others = ((capacity - demand).clip(lower=0.0) - capacity_left).clip(lower=0.0)
                self_supply_pct = (np.minimum(demand, capacity) / demand) * 100.0
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
                        "exported": exported_to_others,
                        "self_supply_pct": self_supply_pct,
                        "capacity_left": capacity_left,
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
        capacity_left = pd.Series(0.0, index=city_ids, dtype=float)
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
            capacity_left += aligned["capacity_left"]

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
                "capacity_left": capacity_left
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
                    supplied_to_others = _as_optional_float(metrics.get("supplied_to_others"))
                else:
                    provision_pct = None
                    served_population = None

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
                    else:
                        # If service metrics are missing for this city, use zeros so
                        # the UI shows 0 instead of null and averages include it.
                        provision_pct = 0.0
                        served_population = 0.0

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
                        "service provision index": float(row.get("service_provision_index")) if "service_provision_index" in row else None,
                        "embedding": [
                            float(row.get("pca1")) if "pca1" in row else None,
                            float(row.get("pca2")) if "pca2" in row else None,
                        ],
                    }
                    if "anomaly_score_in_cluster" in row:
                        ml_block["anomaly"] = float(row.get("anomaly_score_in_cluster"))
                    profiles[cid]["ML"] = ml_block

        return profiles


__all__ = ["UrbanFunctionCalculator", "SERVICE_GROUPS", "SERVICE_ID_TO_NAME"]
