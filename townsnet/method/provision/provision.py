import math
from itertools import product
from typing import Literal
from shapely import LineString

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpVariable, lpSum

from ...models import Block, ServiceType
from ..base_method import BaseMethod


class Provision(BaseMethod):
    """Class provides methods for service type provision assessment"""

    def _add_basemap(self, ax):
        ...
        # cx.add_basemap(ax, source=cx.providers.Stamen.TonerLite, crs=f"EPSG:{self.city_model.epsg}")

    def plot(self, gdf: gpd.GeoDataFrame):
        """Visualizes provision assessment results"""
        ax = gdf.plot(color="#ddd")
        gdf.plot(ax=ax, column="provision", cmap="RdYlGn", vmin=0, vmax=1, legend=True)
        ax.set_title("Provision: " + f"{self.total_provision(gdf): .3f}")
        ax.set_axis_off()
        self._add_basemap(ax)
        return ax

    def plot_delta(self, gdf_before: gpd.GeoDataFrame, gdf_after: gpd.GeoDataFrame):
        gdf = gdf_after.copy()
        ax = gdf.plot(color="#ddd")
        gdf = gdf.loc[gdf["provision"] != 0]
        gdf["provision"] -= gdf_before["provision"]
        gdf.loc[gdf["provision"] != 0].plot(column="provision", ax=ax, cmap="PuOr", vmin=-1, vmax=1, legend=True)
        prov_delta = self.total_provision(gdf_after) - self.total_provision(gdf_before)
        ax.set_title("Provision delta: " + f"{prov_delta: .3f}")
        ax.set_axis_off()
        self._add_basemap(ax)

    def plot_provisions(self, provisions: dict[str, gpd.GeoDataFrame]):
        def show_me_chart(ax, gdf, name, sum):
            gdf.plot(column="provision", legend=True, ax=ax, cmap="RdYlGn", vmin=0, vmax=1)
            gdf[gdf["demand"] == 0].plot(ax=ax, color="#ddd", alpha=1)
            ax.set_title(str(name.name) + " provision: " + f"{sum: .3f}")
            ax.set_axis_off()

        n_plots = len(provisions)
        _, axs = plt.subplots(nrows=n_plots // 2 + n_plots % 2, ncols=2, figsize=(20, 20))
        for i, (service_type, provision_gdf) in enumerate(provisions.items()):
            show_me_chart(axs[i // 2][i % 2], provision_gdf, service_type, self.total_provision(provision_gdf))
        plt.show()

    def _get_filtered_blocks(self, service_type: ServiceType, type: Literal["demand", "capacity"]) -> list[Block]:
        """Get blocks filtered by demand or capacity greater than 0"""
        return list(filter(lambda b: b[service_type.name][type] > 0, self.city_model.blocks))

    def _get_sorted_neighbors(self, block, capacity_blocks: list[Block]):
        return sorted(capacity_blocks, key=lambda b: self.city_model.get_distance(block, b))

    def _get_blocks_gdf(self, service_type: ServiceType) -> gpd.GeoDataFrame:
        """Returns blocks gdf for provision assessment"""
        data: list[dict] = []
        for block in self.city_model.blocks:
            data.append({"id": block.id, "geometry": block.geometry, **block[service_type.name]})
        gdf = gpd.GeoDataFrame(data).set_index("id").set_crs(epsg=self.city_model.epsg)
        gdf["demand_left"] = gdf["demand"]
        gdf["demand_within"] = 0
        gdf["demand_without"] = 0
        gdf["capacity_left"] = gdf["capacity"]
        return gdf

    @classmethod
    def stat_provision(cls, gdf: gpd.GeoDataFrame):
        return {
            "mean": gdf["provision"].mean(),
            "median": gdf["provision"].median(),
            "min": gdf["provision"].min(),
            "max": gdf["provision"].max(),
        }

    @classmethod
    def total_provision(cls, gdf: gpd.GeoDataFrame):
        return gdf["demand_within"].sum() / gdf["demand"].sum()

    def weights_to_gdf(self, weights_df):
        weights = []
        for a in weights_df.index:
            block_a = self.city_model[a]
            for b in weights_df.columns:
                block_b = self.city_model[int(b)]
                value = weights_df.loc[a, b]
                if value > 0:
                    point_a = block_a.geometry.representative_point()
                    point_b = block_b.geometry.representative_point()
                    weights.append(
                        {"geometry": LineString([point_a, point_b]), "from": a, "to": int(b), "weight": value}
                    )
        return gpd.GeoDataFrame(weights, crs=self.city_model.epsg)

    def calculate_scenario(
        self,
        scenario: dict[str, float],
        update_df: pd.DataFrame = None,
        method: Literal["iterative", "lp"] = "lp",
    ) -> tuple[dict[str, gpd.GeoDataFrame], float]:
        result = {}
        total = 0
        for service_type, weight in scenario.items():
            prov_gdf, _ = self.calculate(service_type, update_df, method)
            result[service_type] = prov_gdf
            total += weight * self.total_provision(prov_gdf)
        return result, total

    def calculate(
        self,
        service_type: ServiceType | str,
        update_df: pd.DataFrame = None,
        method: Literal["iterative", "lp", "gravity"] = "lp",
    ) -> gpd.GeoDataFrame:
        """Provision assessment using certain method for the current city and service type, can be used with certain updated blocks GeoDataFrame"""
        if not isinstance(service_type, ServiceType):
            service_type = self.city_model[service_type]
        gdf = self._get_blocks_gdf(service_type)
        df = pd.DataFrame(0, index=gdf.index, columns=gdf.index)
        if update_df is not None:
            gdf = gdf.join(update_df)
            gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
            if not "population" in gdf:
                gdf["population"] = 0
            gdf["demand"] += gdf["population"].apply(service_type.calculate_in_need)
            gdf["demand_left"] = gdf["demand"]
            if not service_type.name in gdf:
                gdf[service_type.name] = 0
            gdf["capacity"] += gdf[service_type.name]
            gdf["capacity_left"] += gdf[service_type.name]

        match method:
            case "lp":
                gdf, df = self._lp_provision(gdf, df, service_type, "lp")
            case "gravity":
                gdf, df = self._lp_provision(gdf, df, service_type, "gravity")
            case "iterative":
                gdf, df = self._iterative_provision(gdf, df, service_type)
        gdf["provision"] = gdf["demand_within"] / gdf["demand"]
        return gdf, df

    def _lp_provision(
        self, gdf: gpd.GeoDataFrame, df: pd.DataFrame, service_type: ServiceType, type: Literal["lp", "gravity"]
    ):
        """Linear programming assessment method"""
        gdf = gdf.copy()

        delta = gdf["demand"].sum() - gdf["capacity"].sum()
        fictive_index = None
        fictive_column = None
        fictive_block_id = gdf.index.max() + 1
        if delta > 0:
            fictive_column = fictive_block_id
            gdf.loc[fictive_column, "capacity"] = delta
        if delta < 0:
            fictive_index = fictive_block_id
            gdf.loc[fictive_index, "demand"] = -delta
        gdf["capacity_left"] = gdf["capacity"]

        def _get_weight(id1: int, id2: int):
            if id1 == fictive_block_id or id2 == fictive_block_id:
                return 0
            block1 = self.city_model[id1]
            block2 = self.city_model[id2]
            self.city_model.get_distance(block1, block2)
            distance = self.city_model.get_distance(block1, block2)
            if type == "lp":
                return distance
            demand = gdf.loc[id1, "demand"]
            capacity = gdf.loc[id1, "capacity"]
            return distance * distance
            # if capacity * demand == 0:
            #     return 0
            # return distance * distance / (capacity * demand)

        def _get_distance(id1: int, id2: int):
            if id1 == fictive_block_id or id2 == fictive_block_id:
                return 0
            block1 = self.city_model[id1]
            block2 = self.city_model[id2]
            distance = self.city_model.get_distance(block1, block2)
            return distance

        demand = gdf.loc[gdf["demand"] > 0]
        capacity = gdf.loc[gdf["capacity"] > 0]

        prob = LpProblem("Transportation", LpMinimize)
        x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None)
        prob += lpSum(_get_weight(n, m) * x[n, m] for n in demand.index for m in capacity.index)
        for n in demand.index:
            prob += lpSum(x[n, m] for m in capacity.index) == demand.loc[n, "demand"]
        for m in capacity.index:
            prob += lpSum(x[n, m] for n in demand.index) == capacity.loc[m, "capacity"]
        prob.solve(PULP_CBC_CMD(msg=False))
        # make the output
        for var in prob.variables():
            value = var.value()
            name = var.name.replace("(", "").replace(")", "").replace(",", "").split("_")
            if name[2] == "dummy":
                continue
            a = int(name[1])
            b = int(name[2])
            weight = _get_distance(a, b)
            if value > 0:
                df.loc[a, b] = value
                if weight <= service_type.accessibility:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_within"] += value
                        gdf.loc[b, "capacity_left"] -= value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_within"] += value
                        gdf.loc[b, "capacity_left"] -= value
                else:
                    if fictive_index != None and a != fictive_index:
                        gdf.loc[a, "demand_without"] += value
                        gdf.loc[b, "capacity_left"] -= value
                    if fictive_column != None and b != fictive_column:
                        gdf.loc[a, "demand_without"] += value
                        gdf.loc[b, "capacity_left"] -= value
        if fictive_index != None or fictive_column != None:
            gdf.drop(labels=[fictive_block_id], inplace=True)
            if fictive_index != None:
                df.drop(labels=[fictive_index], inplace=True)
            if fictive_column != None:
                df.drop(columns=[fictive_column], inplace=True)
        gdf["demand_left"] = gdf["demand"] - gdf["demand_within"] - gdf["demand_without"]
        return gdf, df

    def _iterative_provision(self, gdf: gpd.GeoDataFrame, df: pd.DataFrame, service_type: ServiceType):
        """Iterative provision assessment method"""

        demand_blocks = self._get_filtered_blocks(service_type, "demand")
        capacity_blocks = self._get_filtered_blocks(service_type, "capacity")

        while len(demand_blocks) > 0 and len(capacity_blocks) > 0:
            for demand_block in demand_blocks:
                neighbors = self._get_sorted_neighbors(demand_block, capacity_blocks)
                if len(neighbors) == 0:
                    break
                capacity_block = neighbors[0]
                gdf.loc[demand_block.id, "demand_left"] -= 1
                df.loc[demand_block.id, capacity_block.id] += 1
                weight = self.city_model.get_distance(demand_block, capacity_block)
                if weight <= service_type.accessibility:
                    gdf.loc[demand_block.id, "demand_within"] += 1
                else:
                    gdf.loc[demand_block.id, "demand_without"] += 1
                if gdf.loc[demand_block.id, "demand_left"] == 0:
                    demand_blocks.remove(demand_block)
                gdf.loc[capacity_block.id, "capacity_left"] -= 1
                if gdf.loc[capacity_block.id, "capacity_left"] == 0:
                    capacity_blocks.remove(capacity_block)

        return gdf, df
