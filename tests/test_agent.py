"""PPO agent tests."""

import numpy as np

from redistricting.rl.agent import PPOAgent


def _random_features(graph, dim=12):
    return np.random.randn(len(graph.nodes()), dim).astype(np.float32)


def test_ppo_get_action_returns_4(tiny_graph):
    features = _random_features(tiny_graph, dim=12)
    agent = PPOAgent(node_feature_dim=12, action_dim=16)
    action, log_prob, value, entropy = agent.get_action(
        tiny_graph, features, action_mask=np.ones(16, dtype=np.float32)
    )
    assert isinstance(action, int)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    assert isinstance(entropy, float)


def test_ppo_update_runs(tiny_graph):
    features = _random_features(tiny_graph, dim=12)
    agent = PPOAgent(node_feature_dim=12, action_dim=16)
    for _ in range(10):
        mask = np.ones(16, dtype=np.float32)
        action, log_prob, value, _entropy = agent.get_action(
            tiny_graph, features, action_mask=mask
        )
        agent.store_transition(tiny_graph, features, action, 0.1, log_prob, value, False, action_mask=mask)
    loss_info = agent.update()
    assert loss_info is not None
    assert set(loss_info.keys()) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"}
    for value in loss_info.values():
        assert not np.isnan(value)


def test_model_save_load_roundtrip(tiny_graph, tmp_path):
    features = _random_features(tiny_graph, dim=12)
    agent = PPOAgent(node_feature_dim=12, action_dim=16)
    # Touch weights by running one forward sample.
    agent.get_action(tiny_graph, features, action_mask=np.ones(16, dtype=np.float32))

    save_path = tmp_path / "model.pth"
    agent.save_model(str(save_path))

    agent2 = PPOAgent(node_feature_dim=12, action_dim=16)
    agent2.load_model(str(save_path))

    for p1, p2 in zip(agent.policy.parameters(), agent2.policy.parameters()):
        assert p1.shape == p2.shape

