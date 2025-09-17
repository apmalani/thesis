import geopandas as gpd
import pandas as pd
import gerrychain as gc
from gerrychain.updaters import Tally, Election, cut_edges
import networkx as nx

def build_precinct_graph(state, basepath):
    precincts = gpd.read_file(f"{basepath}/{state}/precincts_with_vap.shp")
    precincts.fillna(0, inplace=True)

    if precincts.crs.is_geographic:
        precincts = precincts.to_crs(epsg=2163)

    attr_cols = [
        "CONG_DIST",
        "P0010001",
        "P0040001",
        "CompDemVot",
        "CompRepVot",
        "ptP0040002",
        "ptP0040005",
        "ptP0040006",
        "ptP0040007",
        "ptP0040008",
        "ptP0040009"
    ]
    
    graph = gc.Graph.from_geodataframe(
        precincts,
        adjacency="rook",
        cols_to_add=[col for col in attr_cols if col in precincts.columns],
        reproject=False
    )

    assignment = {node: data["CONG_DIST"] for node, data in graph.nodes(data=True)}

    updaters_dict = {
        "population": Tally("P0010001", alias="population"),
        "voting_population": Tally("P0040001", alias="voting_population"),
        "DemVotes": Tally("CompDemVot", alias="DemVotes"),
        "RepVotes": Tally("CompRepVot", alias="RepVotes"),

        "2024_Gen": Election(
            "2024 General",
            {"Democratic": "CompDemVot", "Republican": "CompRepVot"},
            alias="2024_Gen"
        ),

        "cut_edges": cut_edges
    }

    partition = gc.Partition(graph, assignment, updaters=updaters_dict)

    return graph, partition

def validate_precinct_graph(graph, partition, tolerance=0.05):
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
            "within_tolerance": within_tol
        }
        if not within_tol:
            results["overall"] = False

    return results