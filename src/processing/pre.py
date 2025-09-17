import geopandas as gpd
import numpy as np
import os

def preprocess_raw_data(state, blockspth_p4, blockspth_p1, precinctspth, districtspth):
    blocks_p4 = gpd.read_file(blockspth_p4)
    p4_fields = [f"P004000{i}" for i in range(1, 10)]
    blocks_p4 = blocks_p4[p4_fields + ["geometry"]]
    
    blocks_p1 = gpd.read_file(blockspth_p1)
    p1_fields = ["P0010001"]
    blocks_p1 = blocks_p1[p1_fields + ["geometry"]]
    
    precincts = gpd.read_file(precinctspth)
    districts = gpd.read_file(districtspth)
    
    blocks_p4 = blocks_p4.to_crs(precincts.crs)
    blocks_p1 = blocks_p1.to_crs(precincts.crs)
    
    blocks_p4["block_area"] = blocks_p4.geometry.area
    overlap_p4 = gpd.overlay(blocks_p4, precincts[["UNIQUE_ID", "geometry"]], how="intersection", keep_geom_type=False)
    overlap_p4["area_frac"] = overlap_p4.geometry.area / overlap_p4["block_area"]
    for col in p4_fields:
        overlap_p4[col] = overlap_p4[col] * overlap_p4["area_frac"]
    agg_p4 = overlap_p4.groupby("UNIQUE_ID")[p4_fields].sum().reset_index()

    blocks_p1["block_area"] = blocks_p1.geometry.area
    overlap_p1 = gpd.overlay(blocks_p1, precincts[["UNIQUE_ID", "geometry"]], how="intersection", keep_geom_type=False)
    overlap_p1["area_frac"] = overlap_p1.geometry.area / overlap_p1["block_area"]
    for col in p1_fields:
        overlap_p1[col] = overlap_p1[col] * overlap_p1["area_frac"]
    agg_p1 = overlap_p1.groupby("UNIQUE_ID")[p1_fields].sum().reset_index()
    
    precincts = precincts.merge(agg_p4, on="UNIQUE_ID", how="left")
    precincts = precincts.merge(agg_p1, on="UNIQUE_ID", how="left")
    
    for i in range(2, 10):
        precincts[f"ptP004000{i}"] = precincts[f"P004000{i}"] / precincts["P0040001"] * 100
    
    dem_cols = [col for col in precincts.columns if col.startswith("G24PRED") or col.startswith("G24USSD")]
    rep_cols = [col for col in precincts.columns if col.startswith("G24PRER") or col.startswith("G24USSR")]
    
    keep_cols = (
        ["UNIQUE_ID", "COUNTYFP", "COUNTY_NAM", "PRECINCTNA",
         "CONG_DIST", "SLDU_DIST", "SLDL_DIST", "geometry"]
        + p4_fields
        + [f"ptP004000{i}" for i in range(2, 10)]
        + p1_fields
        + dem_cols
        + rep_cols
    )
    precincts = precincts[[c for c in keep_cols if c in precincts.columns]]
    for col in p4_fields + p1_fields:
        precincts[col] = np.floor(precincts[col]).astype(int)

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
    
    agg_totals = {f"P004000{i}": "sum" for i in range(1, 10)}
    agg_totals["P0010001"] = "sum"
    agg_totals["CompDemVot"] = "sum"
    agg_totals["CompRepVot"] = "sum"
    
    district_summary = precincts_districts.groupby("DISTRICT").agg(agg_totals).reset_index()
    district_summary["CompDemShare"] = district_summary["CompDemVot"] / (
        district_summary["CompDemVot"] + district_summary["CompRepVot"]
    )
    for i in range(2, 10):
        district_summary[f"ptP004000{i}"] = district_summary[f"P004000{i}"] / district_summary["P0040001"] * 100

    out_dir = f"/home/arun/echo/thesis/data/processed/{state}"
    os.makedirs(out_dir, exist_ok=True)

    precincts.to_file(f"{out_dir}/precincts_with_vap.shp")
    precincts_districts.to_file(f"{out_dir}/precincts_with_districts.shp")
    district_summary.to_csv(f"{out_dir}/district_summary_vap.csv", index=False)