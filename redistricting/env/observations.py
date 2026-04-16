"""Observation and feature extraction for graph-based RL."""

from dataclasses import dataclass
from typing import Dict

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Feature column names used to build node vectors."""

    total_pop: str = "P0010001"
    voting_age_pop: str = "P0040001"
    dem_votes: str = "CompDemVot"
    rep_votes: str = "CompRepVot"
    pct_latino: str = "P0040002"
    pct_white: str = "P0040005"
    pct_black: str = "P0040006"
    pct_native: str = "P0040007"
    pct_asian: str = "P0040008"
    pct_nhpi: str = "P0040009"


def _feature_totals(graph: nx.Graph, cfg: FeatureConfig) -> Dict[str, float]:
    pop = sum(graph.nodes[n].get(cfg.total_pop, 0) for n in graph.nodes())
    vap = sum(graph.nodes[n].get(cfg.voting_age_pop, 0) for n in graph.nodes())
    dem = sum(graph.nodes[n].get(cfg.dem_votes, 0) for n in graph.nodes())
    rep = sum(graph.nodes[n].get(cfg.rep_votes, 0) for n in graph.nodes())
    votes = dem + rep if (dem + rep) > 0 else 1
    return {"pop": pop, "vap": vap, "dem": dem, "rep": rep, "votes": votes}


def build_node_features(
    graph: nx.Graph, assignment: Dict[int, int], n_districts: int, cfg: FeatureConfig = FeatureConfig()
) -> np.ndarray:
    """Build normalized node feature matrix with district one-hot encoding."""
    totals = _feature_totals(graph, cfg)
    node_features = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        pop = node_data.get(cfg.total_pop, 0)
        vap = node_data.get(cfg.voting_age_pop, 0)
        dem_votes = node_data.get(cfg.dem_votes, 0)
        rep_votes = node_data.get(cfg.rep_votes, 0)
        vote_margin = abs(dem_votes - rep_votes) / (dem_votes + rep_votes) if (dem_votes + rep_votes) > 0 else 0
        pct_white = node_data.get(cfg.pct_white, 0) / vap if vap > 0 else 0
        pct_latino = node_data.get(cfg.pct_latino, 0) / vap if vap > 0 else 0
        pct_black = node_data.get(cfg.pct_black, 0) / vap if vap > 0 else 0
        pct_native = node_data.get(cfg.pct_native, 0) / vap if vap > 0 else 0
        pct_asian = node_data.get(cfg.pct_asian, 0) / vap if vap > 0 else 0
        pct_nhpi = node_data.get(cfg.pct_nhpi, 0) / vap if vap > 0 else 0
        pct_minority = 1 - pct_white

        district_onehot = np.zeros(n_districts, dtype=np.float32)
        district_id = assignment[node]
        if isinstance(district_id, (int, np.integer)) and 0 <= district_id < n_districts:
            district_onehot[district_id] = 1.0

        features = np.array(
            [
                pop / totals["pop"] if totals["pop"] > 0 else 0,
                vap / totals["vap"] if totals["vap"] > 0 else 0,
                dem_votes / totals["votes"] if totals["votes"] > 0 else 0,
                rep_votes / totals["votes"] if totals["votes"] > 0 else 0,
                vote_margin,
                pct_white,
                pct_latino,
                pct_black,
                pct_native,
                pct_asian,
                pct_nhpi,
                pct_minority,
            ],
            dtype=np.float32,
        )
        node_features.append(np.concatenate([features, district_onehot], dtype=np.float32))
    return np.array(node_features, dtype=np.float32)

