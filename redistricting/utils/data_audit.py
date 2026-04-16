"""Data integrity checks for required redistricting inputs."""

import warnings
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd

from redistricting.graph.construction import build_precinct_graph

warnings.filterwarnings("ignore")


def audit_data_directory(state: str, basepath: str) -> Dict:
    """Audit shapefile and baseline stats for required columns and keys."""
    results = {
        "state": state,
        "basepath": basepath,
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "shapefile_columns": [],
        "baseline_stats_keys": [],
        "missing_columns": [],
        "missing_baseline_keys": [],
        "node_count": 0,
        "edge_count": 0,
    }
    data_dir = Path(basepath) / state
    shapefile_path = data_dir / "precincts_with_vap.shp"
    if not shapefile_path.exists():
        results["errors"].append(f"Shapefile not found: {shapefile_path}")
        results["status"] = "FAIL"
        return results

    try:
        gdf = gpd.read_file(shapefile_path)
        results["node_count"] = len(gdf)
        results["shapefile_columns"] = list(gdf.columns)
        required_columns = [
            "P0010001",
            "P0040001",
            "P0040002",
            "P0040005",
            "P0040006",
            "P0040007",
            "P0040008",
            "P0040009",
            "CompDemVot",
            "CompRepVot",
        ]
        missing = [col for col in required_columns if col not in gdf.columns]
        if not any(c in gdf.columns for c in ("CONG_DIST", "DISTRICT")):
            missing.append("CONG_DIST_or_DISTRICT")
        if missing:
            results["missing_columns"] = missing
            results["errors"].append(f"Missing required columns: {missing}")
            results["status"] = "FAIL"
    except Exception as exc:
        results["errors"].append(f"Error reading shapefile: {exc}")
        results["status"] = "FAIL"
        return results

    baseline_path = data_dir / "baseline_stats.csv"
    if not baseline_path.exists():
        results["errors"].append(f"Baseline stats file not found: {baseline_path}")
        results["status"] = "FAIL"
        return results

    try:
        baseline_df = pd.read_csv(baseline_path, index_col=0)
        results["baseline_stats_keys"] = list(baseline_df.index)
        required_keys = [
            "EfficiencyGap",
            "PartisanProp",
            "SeatsVotesDiff",
            "PolPopperMin",
            "PolPopperAvg",
            "MinOppMin",
            "MinOppAvg",
        ]
        missing = [key for key in required_keys if key not in baseline_df.index]
        if missing:
            results["missing_baseline_keys"] = missing
            results["errors"].append(f"Missing required baseline stats keys: {missing}")
            results["status"] = "FAIL"
    except Exception as exc:
        results["errors"].append(f"Error reading baseline stats: {exc}")
        results["status"] = "FAIL"
        return results

    try:
        graph, partition = build_precinct_graph(state, basepath)
        results["edge_count"] = graph.number_of_edges()
        results["district_count"] = len(partition.parts)
    except Exception as exc:
        results["warnings"].append(f"Could not verify graph connectivity: {exc}")
    return results


def print_audit_report(results: Dict) -> None:
    """Print formatted audit report."""
    print("=" * 80)
    print(f"DATA INTEGRITY AUDIT REPORT - {results['state'].upper()}")
    print("=" * 80)
    print(f"Status: {results['status']}")
    if results["errors"]:
        print("Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
    if results["warnings"]:
        print("Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    print("=" * 80)

