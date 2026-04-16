"""Build and validate GerryChain precinct graphs."""

from pathlib import Path

import gerrychain as gc
import geopandas as gpd
import networkx as nx
from gerrychain.updaters import Election, Tally, cut_edges


def build_precinct_graph(state: str, basepath: str):
    """Build a precinct graph and baseline partition from a shapefile."""
    base_path = Path(basepath)
    if base_path.name == state:
        shapefile_path = base_path / "precincts_with_vap.shp"
    else:
        shapefile_path = base_path / state / "precincts_with_vap.shp"

    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {shapefile_path}\n"
            f"Expected location: {shapefile_path.resolve()}\n"
            f"Base path: {base_path.resolve()}\n"
            f"State: {state}"
        )

    precincts = gpd.read_file(str(shapefile_path))
    precincts.fillna(0, inplace=True)
    if precincts.crs and precincts.crs.is_geographic:
        precincts = precincts.to_crs(epsg=2163)

    district_col = None
    for candidate in ("CONG_DIST", "DISTRICT", "CD116FP", "SLDLST", "SLDUST"):
        if candidate in precincts.columns:
            district_col = candidate
            break
    if district_col is None:
        alt_path = shapefile_path.parent / "precincts_with_districts.shp"
        if alt_path.exists() and "UNIQUE_ID" in precincts.columns:
            districts_gdf = gpd.read_file(str(alt_path))
            if "DISTRICT" in districts_gdf.columns and "UNIQUE_ID" in districts_gdf.columns:
                merge_cols = ["UNIQUE_ID", "DISTRICT"]
                precincts = precincts.merge(
                    districts_gdf[merge_cols],
                    on="UNIQUE_ID",
                    how="left",
                )
                district_col = "DISTRICT"
    if district_col is None:
        raise ValueError(
            f"No known district assignment column in {shapefile_path} "
            f"(tried CONG_DIST, DISTRICT, merge from precincts_with_districts). "
            f"Columns: {list(precincts.columns)}"
        )

    attr_cols = [
        district_col,
        "P0010001",
        "P0040001",
        "CompDemVot",
        "CompRepVot",
        "P0040002",
        "P0040005",
        "P0040006",
        "P0040007",
        "P0040008",
        "P0040009",
        "ptP0040002",
        "ptP0040005",
        "ptP0040006",
        "ptP0040007",
        "ptP0040008",
        "ptP0040009",
    ]
    graph = gc.Graph.from_geodataframe(
        precincts,
        adjacency="rook",
        cols_to_add=[col for col in attr_cols if col in precincts.columns],
        reproject=False,
    )

    assignment = {node: data[district_col] for node, data in graph.nodes(data=True)}
    updaters_dict = {
        "population": Tally("P0010001", alias="population"),
        "voting_population": Tally("P0040001", alias="voting_population"),
        "DemVotes": Tally("CompDemVot", alias="DemVotes"),
        "RepVotes": Tally("CompRepVot", alias="RepVotes"),
        "2024_Gen": Election(
            "2024 General",
            {"Democratic": "CompDemVot", "Republican": "CompRepVot"},
            alias="2024_Gen",
        ),
        "cut_edges": cut_edges,
    }
    partition = gc.Partition(graph, assignment, updaters=updaters_dict)
    return graph, partition


def validate_precinct_graph(graph: gc.Graph, partition: gc.Partition, tolerance: float = 0.05):
    """Validate contiguity and district population balance."""
    results = {"contiguous": {}, "population_balance": {}, "overall": True}

    for dist, nodes in partition.parts.items():
        subgraph = graph.subgraph(nodes)
        is_contig = nx.is_connected(subgraph)
        results["contiguous"][dist] = is_contig
        if not is_contig:
            results["overall"] = False

    total_pop = sum(partition["population"].values())
    num_districts = len(partition.parts)
    ideal_pop = total_pop / num_districts

    for dist, pop in partition["population"].items():
        deviation = (pop - ideal_pop) / ideal_pop
        within_tol = abs(deviation) <= tolerance
        results["population_balance"][dist] = {
            "population": pop,
            "deviation": deviation,
            "within_tolerance": within_tol,
        }
        if not within_tol:
            results["overall"] = False

    return results

