import geopandas as gpd
import pandera.pandas as pa
from pandera.typing import Series

class GeoDfSchema(pa.DataFrameModel):
    name: Series[str]
    is_anchor: Series[bool]
    population: Series[float]
    geometry: Series

    class Config:
        strict = "filter"
        coerce = False

# Функция валидации
    @classmethod
    def validate(cls, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        required = ["name", "is_anchor", "population", "geometry"]
        missing = [col for col in required if col not in gdf.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        gdf = gdf[required].copy()
        return super().validate(gdf)
