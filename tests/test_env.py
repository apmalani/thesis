"""Environment behavior tests."""

import numpy as np
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally

from redistricting.env.core import GerrymanderingEnv


def _fake_builder(tiny_graph):
    assignment = {n: (n % 4) for n in tiny_graph.nodes()}
    updaters = {"population": Tally("P0010001", alias="population")}
    return Graph.from_networkx(tiny_graph), Partition(
        Graph.from_networkx(tiny_graph), assignment, updaters=updaters
    )


def test_action_mask_prevents_illegal(monkeypatch, tiny_graph):
    monkeypatch.setattr(
        "redistricting.env.core.build_precinct_graph",
        lambda state, basepath: _fake_builder(tiny_graph),
    )
    env = GerrymanderingEnv(
        state="xx",
        basepath="unused",
        reward_fn=lambda metrics, weights: 0.0,
        max_steps=20,
    )
    _obs, _info = env.reset()
    mask = env.get_valid_action_mask()
    valid_ids = np.where(mask > 0.0)[0]
    assert len(valid_ids) >= 0
    for action in valid_ids[: min(20, len(valid_ids))]:
        _obs, _reward, _terminated, _truncated, info = env.step(int(action))
        assert info["max_pop_deviation"] <= 5.0 + 1e-6


def test_env_step_returns_5tuple(monkeypatch, tiny_graph):
    monkeypatch.setattr(
        "redistricting.env.core.build_precinct_graph",
        lambda state, basepath: _fake_builder(tiny_graph),
    )
    env = GerrymanderingEnv(
        state="xx",
        basepath="unused",
        reward_fn=lambda metrics, weights: 0.0,
        max_steps=2,
    )
    _obs, _info = env.reset()
    mask = env.get_valid_action_mask()
    action = int(np.where(mask > 0.0)[0][0]) if np.any(mask > 0.0) else 0
    output = env.step(action)
    assert len(output) == 5


def test_env_reset_restores_baseline(monkeypatch, tiny_graph):
    monkeypatch.setattr(
        "redistricting.env.core.build_precinct_graph",
        lambda state, basepath: _fake_builder(tiny_graph),
    )
    env = GerrymanderingEnv(
        state="xx",
        basepath="unused",
        reward_fn=lambda metrics, weights: 0.0,
        max_steps=5,
    )
    _obs, _info = env.reset()
    baseline = dict(env.partition.assignment)
    mask = env.get_valid_action_mask()
    if np.any(mask > 0.0):
        env.step(int(np.where(mask > 0.0)[0][0]))
    _obs, _info = env.reset()
    assert dict(env.partition.assignment) == baseline

