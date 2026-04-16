"""Pipeline smoke tests."""

from gerrychain import Graph, Partition
from gerrychain.updaters import Tally

from redistricting.env.core import GerrymanderingEnv
from redistricting.rl.agent import PPOAgent, PPOHyperParams
from redistricting.rl.trainer import TrainingConfig, TrainingLoop


def _fake_builder(tiny_graph):
    assignment = {n: (n % 4) for n in tiny_graph.nodes()}
    updaters = {"population": Tally("P0010001", alias="population")}
    return Graph.from_networkx(tiny_graph), Partition(
        Graph.from_networkx(tiny_graph), assignment, updaters=updaters
    )


def test_3_episode_smoke(monkeypatch, tiny_graph, tmp_path):
    monkeypatch.setattr(
        "redistricting.env.core.build_precinct_graph",
        lambda state, basepath: _fake_builder(tiny_graph),
    )
    env = GerrymanderingEnv(state="xx", basepath="unused", reward_fn=lambda metrics, weights: 0.0, max_steps=10)
    _graph, features = env.get_graph_observation()
    agent = PPOAgent(
        node_feature_dim=features.shape[1],
        action_dim=env.action_space.n,
        hyperparams=PPOHyperParams(k_epochs=2),
    )
    trainer = TrainingLoop(
        env=env,
        agent=agent,
        config=TrainingConfig(num_episodes=3, update_frequency=1, save_frequency=1000),
        run_dir=tmp_path / "run",
    )
    history = trainer.train()
    assert len(history["episode_rewards"]) == 3
    assert len(history["policy_losses"]) >= 1

