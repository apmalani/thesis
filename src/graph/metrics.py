import geopandas as gpd
import pandas as pd
import numpy as np

class MCalc:
    def __init__(self):
        self._cached_geometries = {}
        self._cached_districts = None
        self._cached_totals = None
        self._cached_demographics = None

    def _prepare_partition_data(self, partition, use_geometry=False):
        districts = pd.DataFrame([
            {
                "DISTRICT": d,
                "P0010001": sum(partition.graph.nodes[n].get("P0010001", 0) for n in nodes),
                "P0040001": sum(partition.graph.nodes[n].get("P0040001", 0) for n in nodes),
                "CompDemVot": sum(partition.graph.nodes[n].get("CompDemVot", 0) for n in nodes),
                "CompRepVot": sum(partition.graph.nodes[n].get("CompRepVot", 0) for n in nodes),
                "P0040002": sum(partition.graph.nodes[n].get("P0040002", 0) for n in nodes),
                "P0040005": sum(partition.graph.nodes[n].get("P0040005", 0) for n in nodes),
                "P0040006": sum(partition.graph.nodes[n].get("P0040006", 0) for n in nodes),
                "P0040007": sum(partition.graph.nodes[n].get("P0040007", 0) for n in nodes),
                "P0040008": sum(partition.graph.nodes[n].get("P0040008", 0) for n in nodes),
                "P0040009": sum(partition.graph.nodes[n].get("P0040009", 0) for n in nodes),
            }
            for d, nodes in partition.parts.items()
        ])
        
        total_dem = districts["CompDemVot"].sum()
        total_rep = districts["CompRepVot"].sum()
        total_votes = total_dem + total_rep

        if use_geometry:
            geom_series = []
            for d, nodes in partition.parts.items():
                geoms = [partition.graph.nodes[n]["geometry"] for n in nodes]
                geom_series.append(gpd.GeoSeries(geoms).geometry.union_all())
            districts_geo = gpd.GeoDataFrame(districts, geometry=geom_series)
            if districts_geo.crs is None:
                districts_geo.set_crs(epsg=2163, inplace=True)
            return districts, districts_geo, total_dem, total_rep, total_votes
        else:
            return districts, None, total_dem, total_rep, total_votes

    def _efficiency_gap(self, districts, total_votes):
        dem_votes = districts["CompDemVot"].values
        rep_votes = districts["CompRepVot"].values
        total_votes_per_district = dem_votes + rep_votes
        
        wasted_dem = np.sum(np.where(dem_votes > rep_votes, 
                                   dem_votes - (total_votes_per_district // 2 + 1), 
                                   dem_votes))
        wasted_rep = np.sum(np.where(rep_votes > dem_votes, 
                                   rep_votes - (total_votes_per_district // 2 + 1), 
                                   rep_votes))
        
        return (wasted_dem - wasted_rep) / total_votes if total_votes > 0 else np.nan

    def _partisan_proportionality(self, districts, total_dem, total_votes):
        dem_vot_shr = total_dem / total_votes if total_votes > 0 else 0
        dem_seats = (districts["CompDemVot"] > districts["CompRepVot"]).sum()
        seat_share = dem_seats / len(districts)
        return seat_share / dem_vot_shr if dem_vot_shr > 0 else np.nan

    def _seats_votes_difference(self, districts, total_dem, total_rep, total_votes):
        p_dem = total_dem / total_votes if total_votes > 0 else 0
        p_rep = total_rep / total_votes if total_votes > 0 else 0
        dem_seats = (districts["CompDemVot"] > districts["CompRepVot"]).sum()
        rep_seats = (districts["CompRepVot"] > districts["CompDemVot"]).sum()
        total_seats = len(districts)
        s_dem = dem_seats / total_seats if total_seats > 0 else 0
        s_rep = rep_seats / total_seats if total_seats > 0 else 0
        diffs = []
        if p_dem > 0:
            diffs.append(abs(p_dem - s_dem) / p_dem)
        if p_rep > 0:
            diffs.append(abs(p_rep - s_rep) / p_rep)
        return np.mean(diffs) if diffs else None

    def _polsby_popper(self, districts_geo):
        scores = []
        for _, row in districts_geo.iterrows():
            geom = row.geometry
            area = geom.area
            perim = geom.length
            score = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
            scores.append(score)
        return {"pp_avg": float(np.mean(scores)), "pp_min": float(np.min(scores))}

    def _minority_opportunity(self, districts, threshold=0.5):
        total_pop = districts["P0040001"].values
        valid_mask = total_pop > 0
        
        if not np.any(valid_mask):
            return {"minority_avg": 0.0, "minority_min": 0.0}
        
        pct_white = districts["P0040005"].values[valid_mask] / total_pop[valid_mask]
        pct_latino = districts["P0040002"].values[valid_mask] / total_pop[valid_mask]
        pct_black = districts["P0040006"].values[valid_mask] / total_pop[valid_mask]
        pct_native = districts["P0040007"].values[valid_mask] / total_pop[valid_mask]
        pct_asian = districts["P0040008"].values[valid_mask] / total_pop[valid_mask]
        pct_nhpi = districts["P0040009"].values[valid_mask] / total_pop[valid_mask]
        pct_minority = 1 - pct_white
        
        opportunity_mask = (
            (pct_latino >= threshold) | (pct_black >= threshold) | (pct_native >= threshold) |
            (pct_asian >= threshold) | (pct_nhpi >= threshold) | (pct_minority >= threshold)
        )
        
        if np.any(opportunity_mask):
            minority_shares = pct_minority[opportunity_mask]
            return {"minority_avg": float(np.mean(minority_shares)), "minority_min": float(np.min(minority_shares))}
        return {"minority_avg": 0.0, "minority_min": 0.0}

    def _mean_median_test(self, districts):
        dem_shares = districts["CompDemVot"] / (districts["CompDemVot"] + districts["CompRepVot"])
        rep_shares = districts["CompRepVot"] / (districts["CompDemVot"] + districts["CompRepVot"])
        return {
            "dem_mm": dem_shares.mean() - dem_shares.median(),
            "rep_mm": rep_shares.mean() - rep_shares.median()
        }

    def calculate_metrics(self, partition, baseline=False, include_geometry=False):
        districts, districts_geo, total_dem, total_rep, total_votes = self._prepare_partition_data(partition, use_geometry=include_geometry)

        metrics = {
            "EfficiencyGap": self._efficiency_gap(districts, total_votes),
            "PartisanProp": self._partisan_proportionality(districts, total_dem, total_votes),
            "SeatsVotesDiff": self._seats_votes_difference(districts, total_dem, total_rep, total_votes),
            "MinOppAvg": self._minority_opportunity(districts)["minority_avg"],
            "MinOppMin": self._minority_opportunity(districts)["minority_min"]
        }

        if include_geometry and districts_geo is not None:
            pp_scores = self._polsby_popper(districts_geo)
            metrics.update({
                "PolPopperAvg": pp_scores["pp_avg"],
                "PolPopperMin": pp_scores["pp_min"]
            })
        else:
            metrics.update({"PolPopperAvg": np.nan, "PolPopperMin": np.nan})

        if baseline:
            mm = self._mean_median_test(districts)
            metrics.update({"MeanMedianDem": mm["dem_mm"], "MeanMedianRep": mm["rep_mm"]})

        return pd.DataFrame([metrics])
