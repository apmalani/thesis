import geopandas as gpd
import pandas as pd
import numpy as np

class MCalc:
    def __init__(self, state, basepath):
        self.state = state
        self.basepath = basepath
        self.districts = pd.read_csv(f"{basepath}/{state}/district_summary_vap.csv")
        self.districts_geo = gpd.read_file(f"{basepath}/{state}/precincts_with_districts.shp")

        if self.districts_geo.crs.is_geographic:
            self.districts_geo = self.districts_geo.to_crs(epsg=2163)

        self.total_dem = self.districts["CompDemVot"].sum()
        self.total_rep = self.districts["CompRepVot"].sum()
        self.total_votes = self.total_dem + self.total_rep

    def _efficiency_gap(self):
        wasted_dem, wasted_rep = 0, 0
        for _, row in self.districts.iterrows():
            dem, rep = row["CompDemVot"], row["CompRepVot"]
            total = dem + rep

            if dem > rep:
                wasted_dem += dem - (total // 2 + 1)
                wasted_rep += rep
            else:
                wasted_rep += rep - (total // 2 + 1)
                wasted_dem += dem

        return (wasted_dem - wasted_rep) / self.total_votes if self.total_votes > 0 else np.nan

    def _partisan_proportionality(self):
        dem_vot_shr = self.districts["CompDemVot"].sum() / self.total_votes

        dem_seats = (self.districts["CompDemVot"] > self.districts["CompRepVot"]).sum()
        seat_share = dem_seats / len(self.districts)

        return seat_share / dem_vot_shr if dem_vot_shr > 0 else np.nan

    def _seats_votes_difference(self):

        p_dem = self.total_dem / self.total_votes if self.total_votes > 0 else 0
        p_rep = self.total_rep / self.total_votes if self.total_votes > 0 else 0

        dem_seats = (self.districts["CompDemVot"] > self.districts["CompRepVot"]).sum()
        rep_seats = (self.districts["CompRepVot"] > self.districts["CompDemVot"]).sum()
        total_seats = len(self.districts)

        s_dem = dem_seats / total_seats if total_seats > 0 else 0
        s_rep = rep_seats / total_seats if total_seats > 0 else 0

        diffs = []
        if p_dem > 0:
            diffs.append(abs(p_dem - s_dem) / p_dem)
        if p_rep > 0:
            diffs.append(abs(p_rep - s_rep) / p_rep)

        return np.mean(diffs) if diffs else None

    def _polsby_popper(self):
        scores = []
        for d, group in self.districts_geo.groupby("DISTRICT"):
            geom = group.unary_union
            area = geom.area
            perim = geom.length

            score = 4 * np.pi * area / (perim ** 2) if perim > 0 else 0
            scores.append(score)

        return {
            "pp_avg": float(np.mean(scores)),
            "pp_min": float(np.min(scores))
        }

    def _minority_opportunity(self, threshold=0.5):
        shares = []
        for _, d in self.districts.iterrows():
            total = d["P0040001"]
            if total == 0:
                continue

            pct_white  = d["P0040005"] / total
            pct_latino = d["P0040002"] / total
            pct_black  = d["P0040006"] / total
            pct_native = d["P0040007"] / total
            pct_asian  = d["P0040008"] / total
            pct_nhpi   = d["P0040009"] / total

            pct_minority = 1 - pct_white

            if (
                pct_latino >= threshold
                or pct_black >= threshold
                or pct_native >= threshold
                or pct_asian >= threshold
                or pct_nhpi >= threshold
                or pct_minority >= threshold
            ):
                shares.append(pct_minority)

        if shares:
            return {
                "minority_avg": float(np.mean(shares)),
                "minority_min": float(np.min(shares))
            }
        else:
            return {"minority_avg": 0.0, "minority_min": 0.0}
        
    def _mean_median_test(self):
        dem_shares = self.districts["CompDemVot"] / (
                self.districts["CompDemVot"] + self.districts["CompRepVot"]
            )
        rep_shares = self.districts["CompRepVot"] / (
                self.districts["CompDemVot"] + self.districts["CompRepVot"]
            )

        dem_mean_share = dem_shares.mean()
        dem_median_share = dem_shares.median()

        rep_mean_share = rep_shares.mean()
        rep_median_share = rep_shares.median()

        res =  {
            "dem_mm": dem_mean_share - dem_median_share,
            "rep_mm": rep_mean_share - rep_median_share
        }

        return res

    def calculate_metrics(self, baseline=False):
        eg = self._efficiency_gap()
        papr = self._partisan_proportionality()
        svr = self._seats_votes_difference()
        polp = self._polsby_popper()
        mod = self._minority_opportunity()

        metrics = {
            "EfficiencyGap": eg,
            "PartisanProp": papr,
            "SeatsVotesDiff": svr,
            "PolPopperAvg": polp["pp_avg"],
            "PolPopperMin": polp["pp_min"],
            "MinOppAvg": mod["minority_avg"],
            "MinOppMin": mod["minority_min"]
        }

        if baseline:
            mm = self._mean_median_test()
            metrics = {
                **metrics,
                "MeanMedianDem": mm["dem_mm"],
                "MeanMedianRep": mm["rep_mm"]
            }

        res = pd.DataFrame([metrics])
        res.to_csv(f"{self.basepath}/{self.state}/district_summary_metrics.csv", index=False)