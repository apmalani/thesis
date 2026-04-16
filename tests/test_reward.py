"""Reward tests."""

import pandas as pd

from redistricting.reward.shaping import DeltaRewardWrapper
from redistricting.reward.zscore import ZScoreReward


def test_zscore_reward_known_input(tmp_path):
    baseline = pd.DataFrame(
        {
            "mean": [0.1, 0.5],
            "median": [0.1, 0.5],
            "std": [0.2, 0.1],
        },
        index=["EfficiencyGap", "PolPopperAvg"],
    )
    baseline_path = tmp_path / "baseline_stats.csv"
    baseline.to_csv(baseline_path)

    reward_fn = ZScoreReward(str(baseline_path))
    metrics = {"EfficiencyGap": 0.0, "PolPopperAvg": 0.6}
    weights = {"EfficiencyGap": 1.0, "PolPopperAvg": 1.0}
    reward = reward_fn(metrics, weights)
    # EG uses abs+lower-is-better => z=(0.0-0.1)/0.2=-0.5 then negated => +0.5; PP => +1.0.
    assert abs(reward - 1.5) < 1e-6


def test_delta_reward_wrapper():
    wrapper = DeltaRewardWrapper(scale_factor=2.0, exploration_coef=0.0)
    first = wrapper(1.2)
    second = wrapper(1.7)
    assert first == 0.0
    assert abs(second - 1.0) < 1e-8


def test_zscore_reward_ignores_nan_metric(tmp_path):
    baseline = pd.DataFrame(
        {
            "mean": [0.1],
            "median": [0.1],
            "std": [0.2],
        },
        index=["EfficiencyGap"],
    )
    baseline_path = tmp_path / "baseline_stats.csv"
    baseline.to_csv(baseline_path)
    reward_fn = ZScoreReward(str(baseline_path))
    reward = reward_fn({"EfficiencyGap": float("nan")}, {"EfficiencyGap": 1.0})
    assert reward == 0.0

