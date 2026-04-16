"""Graph and metric smoke tests."""

import pytest

from redistricting.graph.construction import build_precinct_graph, validate_precinct_graph
from redistricting.graph.metrics import MCalc


@pytest.mark.slow
def test_graph_loads(az_basepath):
    graph, _partition = build_precinct_graph("az", az_basepath)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


@pytest.mark.slow
def test_graph_validation(az_basepath):
    graph, partition = build_precinct_graph("az", az_basepath)
    validation = validate_precinct_graph(graph, partition)
    assert "overall" in validation


@pytest.mark.slow
def test_metrics_calculation(az_basepath):
    _graph, partition = build_precinct_graph("az", az_basepath)
    metrics_df = MCalc().calculate_metrics(partition, include_geometry=False)
    expected = {"EfficiencyGap", "PartisanProp", "SeatsVotesDiff", "MinOppAvg", "PolPopperAvg"}
    assert expected.issubset(set(metrics_df.columns))

