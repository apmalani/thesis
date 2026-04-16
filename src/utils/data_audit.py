"""
Data Integrity Audit Script for Redistricting Pipeline.

Verifies that all required data files and columns are present and consistent.
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def audit_data_directory(state: str, basepath: str) -> Dict:
    """
    Audit data directory for integrity and completeness.
    
    Args:
        state: State abbreviation (e.g., 'az')
        basepath: Base path to processed data
        
    Returns:
        Dictionary with audit results
    """
    results = {
        'state': state,
        'basepath': basepath,
        'status': 'PASS',
        'errors': [],
        'warnings': [],
        'shapefile_columns': [],
        'baseline_stats_keys': [],
        'missing_columns': [],
        'missing_baseline_keys': [],
        'node_count': 0,
        'edge_count': 0
    }
    
    data_dir = Path(basepath) / state
    
    # 1. Check shapefile exists
    shapefile_path = data_dir / "precincts_with_vap.shp"
    if not shapefile_path.exists():
        results['errors'].append(f"Shapefile not found: {shapefile_path}")
        results['status'] = 'FAIL'
        return results
    
    # 2. Load and check shapefile columns
    try:
        gdf = gpd.read_file(shapefile_path)
        results['node_count'] = len(gdf)
        results['shapefile_columns'] = list(gdf.columns)
        
        # Required columns for GNN
        required_columns = [
            'P0010001',  # Total population
            'P0040001',  # Voting age population
            'P0040002',  # Latino
            'P0040005',  # White
            'P0040006',  # Black
            'P0040007',  # Native American
            'P0040008',  # Asian
            'P0040009',  # NHPI
            'CompDemVot',  # Democratic votes
            'CompRepVot',  # Republican votes
            'CONG_DIST',  # Congressional district assignment
        ]
        
        missing = [col for col in required_columns if col not in gdf.columns]
        if missing:
            results['missing_columns'] = missing
            results['errors'].append(f"Missing required columns: {missing}")
            results['status'] = 'FAIL'
        else:
            results['warnings'].append("All required columns present in shapefile")
        
        # Check for null values in critical columns
        critical_cols = ['P0010001', 'P0040001', 'CompDemVot', 'CompRepVot']
        for col in critical_cols:
            if col in gdf.columns:
                null_count = gdf[col].isna().sum()
                if null_count > 0:
                    results['warnings'].append(
                        f"Column '{col}' has {null_count} null values"
                    )
    
    except Exception as e:
        results['errors'].append(f"Error reading shapefile: {e}")
        results['status'] = 'FAIL'
        return results
    
    # 3. Check baseline_stats.csv exists
    baseline_path = data_dir / "baseline_stats.csv"
    if not baseline_path.exists():
        results['errors'].append(f"Baseline stats file not found: {baseline_path}")
        results['status'] = 'FAIL'
        return results
    
    # 4. Load and check baseline stats
    try:
        baseline_df = pd.read_csv(baseline_path, index_col=0)
        results['baseline_stats_keys'] = list(baseline_df.index)
        
        # Required keys from reward function
        required_keys = [
            'EfficiencyGap',
            'PartisanProp',
            'SeatsVotesDiff',
            'PolPopperMin',
            'PolPopperAvg',
            'MinOppMin',
            'MinOppAvg'
        ]
        
        missing = [key for key in required_keys if key not in baseline_df.index]
        if missing:
            results['missing_baseline_keys'] = missing
            results['errors'].append(f"Missing required baseline stats keys: {missing}")
            results['status'] = 'FAIL'
        else:
            results['warnings'].append("All required baseline stats keys present")
        
        # Check required columns in baseline stats
        required_cols = ['mean', 'median', 'std']
        missing_cols = [col for col in required_cols if col not in baseline_df.columns]
        if missing_cols:
            results['errors'].append(
                f"Baseline stats missing required columns: {missing_cols}"
            )
            results['status'] = 'FAIL'
        
    except Exception as e:
        results['errors'].append(f"Error reading baseline stats: {e}")
        results['status'] = 'FAIL'
        return results
    
    # 5. Check graph connectivity (if possible)
    try:
        from graph.construction import build_precinct_graph
        graph, partition = build_precinct_graph(state, basepath)
        results['edge_count'] = graph.number_of_edges()
        results['district_count'] = len(partition.parts)
        
        # Check graph connectivity
        import networkx as nx
        if not nx.is_connected(graph):
            results['warnings'].append(
                "Graph is not fully connected (isolated components detected)"
            )
        else:
            results['warnings'].append("Graph is fully connected")
            
    except Exception as e:
        results['warnings'].append(f"Could not verify graph connectivity: {e}")
    
    return results


def print_audit_report(results: Dict):
    """Print formatted audit report."""
    print("=" * 80)
    print(f"DATA INTEGRITY AUDIT REPORT - {results['state'].upper()}")
    print("=" * 80)
    print(f"\nStatus: {results['status']}")
    print(f"Data Directory: {results['basepath']}/{results['state']}")
    
    if results['node_count'] > 0:
        print(f"\nShapefile Info:")
        print(f"  - Nodes (Precincts): {results['node_count']}")
        print(f"  - Edges: {results['edge_count']}")
        print(f"  - Districts: {results.get('district_count', 'N/A')}")
        print(f"  - Columns: {len(results['shapefile_columns'])}")
    
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['missing_columns']:
        print(f"\nMissing Shapefile Columns:")
        for col in results['missing_columns']:
            print(f"  - {col}")
    
    if results['missing_baseline_keys']:
        print(f"\nMissing Baseline Stats Keys:")
        for key in results['missing_baseline_keys']:
            print(f"  - {key}")
    
    print("\n" + "=" * 80)
    
    if results['status'] == 'PASS':
        print("✅ DATA INTEGRITY CHECK PASSED")
    else:
        print("❌ DATA INTEGRITY CHECK FAILED")
        print("\nPlease fix the errors above before proceeding with training.")
    
    print("=" * 80)


if __name__ == "__main__":
    import sys
    from utils.paths import get_data_dir
    
    state = sys.argv[1] if len(sys.argv) > 1 else "az"
    basepath = str(get_data_dir(None, "processed"))
    
    results = audit_data_directory(state, basepath)
    print_audit_report(results)
    
    sys.exit(0 if results['status'] == 'PASS' else 1)

