import geopandas as gpd
import pandas as pd
import os

def preprocess_raw_data(state, blockspth, precinctspth, districtspth):
    blocks = gpd.read_file(blockspth)
    precincts = gpd.read_file(precinctspth)
    districts = gpd.read_file(districtspth)

    p4_fields = [ f"P004000{i}" for i in range(1, 10) ]

    blocks = blocks[p4_fields + ["geometry"]]
    blocks = blocks.to_crs(precincts.crs)

    blocks_precincts = gpd.sjoin(
        blocks,
        precincts[["UNIQUE_ID", "geometry"]],
        how="left",
        predicate="within"
    )

    agg = blocks_precincts.groupby("UNIQUE_ID")[p4_fields].sum().reset_index()
    precincts = precincts.merge(agg, on="UNIQUE_ID", how="left")

    for i in range(2, 10):
        precincts[f"ptP004000{i}"] = precincts[f"P004000{i}"] / precincts["P0040001"] * 100

    dem_cols = [
        col for col in precincts.columns if col.startswith("G24PRED") or col.startswith("G24USSD")
    ]
    rep_cols = [
        col for col in precincts.columns if col.startswith("G24PRER") or col.startswith("G24USSR")
    ]

    keep_cols = (
        ["UNIQUE_ID", "COUNTYFP", "COUNTY_NAM", "PRECINCTNA",
         "CONG_DIST", "SLDU_DIST", "SLDL_DIST", "geometry"]
        + p4_fields
        + [f"ptP004000{i}" for i in range(2, 10)]
        + dem_cols
        + rep_cols
    )

    precincts = precincts[[c for c in keep_cols if c in precincts.columns]]

    precincts["CompDemVot"] = precincts[dem_cols].sum(axis=1)
    precincts["CompRepVot"] = precincts[rep_cols].sum(axis=1)

    precincts["CompDemShr"] = precincts["CompDemVot"] / (
        precincts["CompDemVot"] + precincts["CompRepVot"]
    )
    
    precincts = precincts.to_crs(districts.crs)

    precincts_districts = gpd.sjoin(
        precincts,
        districts[["DISTRICT", "geometry"]],
        how="left",
        predicate="intersects"
    )

    precincts_districts = precincts_districts.rename(columns={"index_right": "idx_right"})

    agg_totals = {
        f"P004000{i}" : "sum" for i in range(1, 10)
    }

    agg_totals["CompDemVot"] = "sum"
    agg_totals["CompRepVot"] = "sum"

    district_summary = precincts_districts.groupby("DISTRICT").agg(agg_totals).reset_index()
    
    district_summary["CompDemShare"] = district_summary["CompDemVot"] / (
        district_summary["CompDemVot"] + district_summary["CompRepVot"]
    )

    for i in range(2, 10):
        district_summary[f"ptP004000{i}"] = district_summary[f"P004000{i}"] / district_summary["P0040001"] * 100
    
    os.makedirs(f"/home/arun/echo/thesis/data/processed/{state}", exist_ok=True)

    precincts.to_file(f"/home/arun/echo/thesis/data/processed/{state}/precincts_with_vap.shp")
    precincts_districts.to_file(f"/home/arun/echo/thesis/data/processed/{state}/precincts_with_districts.shp")
    district_summary.to_csv(f"/home/arun/echo/thesis/data/processed/{state}/district_summary_vap.csv", index=False)