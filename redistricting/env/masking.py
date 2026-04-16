"""Action validity and masking helpers."""

from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np


def build_action_mask(valid_actions: List[Tuple[int, int]], action_space_size: int) -> np.ndarray:
    """Build a binary action mask for the fixed action space size."""
    mask = np.zeros(action_space_size, dtype=np.float32)
    mask[: min(len(valid_actions), action_space_size)] = 1.0
    return mask


def check_contiguity(graph: nx.Graph, assignment: Dict[int, int], district: int) -> bool:
    """Return True if all nodes assigned to `district` are connected."""
    district_nodes = [node for node, dist in assignment.items() if dist == district]
    if len(district_nodes) <= 1:
        return True
    if len(district_nodes) == 2:
        return district_nodes[1] in graph.neighbors(district_nodes[0])
    district_subgraph = graph.subgraph(district_nodes)
    return nx.is_connected(district_subgraph)


def population_bounds(graph: nx.Graph, n_districts: int, pop_tol: float) -> Tuple[float, float, float]:
    """Compute ideal, minimum, and maximum district populations."""
    total_pop = sum(graph.nodes[n].get("P0010001", 0) for n in graph.nodes())
    ideal_pop = total_pop / n_districts
    return ideal_pop, ideal_pop * (1.0 - pop_tol), ideal_pop * (1.0 + pop_tol)


def district_populations(
    graph: nx.Graph, assignment: Dict[int, int], districts: Iterable[int]
) -> Dict[int, float]:
    """Compute district populations for requested district IDs."""
    out = {}
    for district in districts:
        out[district] = sum(
            graph.nodes[n].get("P0010001", 0) for n in graph.nodes() if assignment.get(n) == district
        )
    return out

