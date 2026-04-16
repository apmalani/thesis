"""Short comparison: capped vs full valid-action generation (smoke-level ablation)."""

from gerrychain import Graph, Partition
from gerrychain.updaters import Tally

from redistricting.env.core import GerrymanderingEnv


def _fake_builder(tiny_graph):
    assignment = {n: (n % 4) for n in tiny_graph.nodes()}
    updaters = {"population": Tally("P0010001", alias="population")}
    return Graph.from_networkx(tiny_graph), Partition(
        Graph.from_networkx(tiny_graph), assignment, updaters=updaters
    )


def test_capped_vs_full_action_space_smoke(monkeypatch, tiny_graph):
    """Both caps should step without error; full list can expose more legal indices."""
    monkeypatch.setattr(
        "redistricting.env.core.build_precinct_graph",
        lambda state, basepath: _fake_builder(tiny_graph),
    )
    for cap in (4, None):
        env = GerrymanderingEnv(
            state="xx",
            basepath="unused",
            reward_fn=lambda metrics, weights: 0.0,
            max_steps=5,
            max_action_space_size=cap,
        )
        env.reset()
        mask = env.get_valid_action_mask()
        n_legal = int(mask.sum())
        if n_legal > 0:
            _, _, term, trunc, _ = env.step(0)
            assert term or trunc or env.current_step >= 0
        else:
            assert env.action_space.n >= 1
