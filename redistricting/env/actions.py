"""Valid action generation and incremental updates."""

from typing import Dict, List, Set, Tuple

import networkx as nx

from redistricting.env.masking import check_contiguity, district_populations, population_bounds


def _is_move_valid(
    graph: nx.Graph,
    assignment: Dict[int, int],
    node: int,
    target_district: int,
    pop_tol: float,
    n_districts: int,
) -> bool:
    current_district = assignment[node]
    if current_district == target_district:
        return False
    temp_assignment = dict(assignment)
    temp_assignment[node] = target_district
    _, min_allowed_pop, max_allowed_pop = population_bounds(graph, n_districts, pop_tol)
    pops = district_populations(graph, temp_assignment, [current_district, target_district])
    if pops[current_district] < min_allowed_pop or pops[target_district] > max_allowed_pop:
        return False
    if not check_contiguity(graph, temp_assignment, current_district):
        return False
    if not check_contiguity(graph, temp_assignment, target_district):
        return False
    return True


def generate_valid_actions(
    graph: nx.Graph,
    assignment: Dict[int, int],
    pop_tol: float,
    n_districts: int,
    max_actions: int | None = None,
) -> List[Tuple[int, int]]:
    """Generate legal actions as tuples: (node, target_district)."""
    actions: List[Tuple[int, int]] = []
    for node in graph.nodes():
        target_districts: Set[int] = {assignment[nbr] for nbr in graph.neighbors(node)}
        for target_district in target_districts:
            if _is_move_valid(graph, assignment, node, target_district, pop_tol, n_districts):
                actions.append((node, target_district))
                if max_actions is not None and len(actions) >= max_actions:
                    return actions
    return actions


def update_valid_actions_incremental(
    graph: nx.Graph,
    assignment: Dict[int, int],
    old_district: int,
    new_district: int,
    moved_node: int,
    prev_actions: List[Tuple[int, int]],
    pop_tol: float,
    n_districts: int,
    max_actions: int | None = None,
) -> List[Tuple[int, int]]:
    """Incrementally refresh valid actions near affected districts."""
    retained = [(n, d) for n, d in prev_actions if n != moved_node]
    affected_nodes = set()
    for node in graph.nodes():
        if assignment[node] in {old_district, new_district}:
            affected_nodes.add(node)
            affected_nodes.update(graph.neighbors(node))

    refreshed = set(retained)
    for node in affected_nodes:
        target_districts = {assignment[nbr] for nbr in graph.neighbors(node)}
        for target_district in target_districts:
            action = (node, target_district)
            if action in refreshed:
                continue
            if _is_move_valid(graph, assignment, node, target_district, pop_tol, n_districts):
                refreshed.add(action)

    # Remove actions that may have become invalid.
    pruned = [
        action
        for action in refreshed
        if _is_move_valid(graph, assignment, action[0], action[1], pop_tol, n_districts)
    ]
    pruned_sorted = sorted(pruned)
    if max_actions is not None:
        return pruned_sorted[:max_actions]
    return pruned_sorted

