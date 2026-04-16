"""Shared pytest fixtures."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally

from redistricting.utils.paths import get_data_dir


@pytest.fixture
def tiny_graph():
    """Return a small synthetic graph with required node attributes."""
    g = nx.grid_2d_graph(4, 5)
    relabeled = {node: idx for idx, node in enumerate(g.nodes())}
    g = nx.relabel_nodes(g, relabeled)
    for node in g.nodes():
        g.nodes[node]["P0010001"] = 100
        g.nodes[node]["P0040001"] = 70
        g.nodes[node]["CompDemVot"] = 40 + (node % 3)
        g.nodes[node]["CompRepVot"] = 35 + (node % 2)
        g.nodes[node]["P0040002"] = 20
        g.nodes[node]["P0040005"] = 40
        g.nodes[node]["P0040006"] = 10
        g.nodes[node]["P0040007"] = 3
        g.nodes[node]["P0040008"] = 5
        g.nodes[node]["P0040009"] = 1
        g.nodes[node]["geometry"] = None
    return g


@pytest.fixture
def mock_partition(tiny_graph):
    """Return partition with 4 districts."""
    assignment = {n: (n % 4) for n in tiny_graph.nodes()}
    updaters = {"population": Tally("P0010001", alias="population")}
    return Partition(Graph.from_networkx(tiny_graph), assignment, updaters=updaters)


@pytest.fixture
def az_basepath():
    """Return processed data path for integration tests."""
    return str(get_data_dir(None, "processed"))

