"""Core Gymnasium environment for redistricting."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from redistricting.env.actions import generate_valid_actions, update_valid_actions_incremental
from redistricting.env.masking import build_action_mask
from redistricting.env.observations import FeatureConfig, build_node_features
from redistricting.graph.construction import build_precinct_graph
from redistricting.graph.metrics import MCalc
from redistricting.reward.shaping import (
    DeltaRewardWrapper,
    EMADeltaRewardWrapper,
    build_default_reward_weights,
)
from redistricting.reward.zscore import ZScoreReward


class GerrymanderingEnv(gym.Env):
    """Graph-based redistricting environment with hard action masking."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        state: str,
        basepath: str,
        pop_tol: float = 0.05,
        reward_fn: Optional[ZScoreReward] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        max_steps: int = 250,
        feature_config: FeatureConfig = FeatureConfig(),
        delta_reward: Optional[DeltaRewardWrapper] = None,
        ema_delta_reward: Optional[EMADeltaRewardWrapper] = None,
        reward_mode: str = "delta",
        score_reward_scale: float = 1.0,
        delta_scale_factor: float = 100.0,
        exploration_coef: float = 0.0001,
        ema_alpha: float = 0.1,
        include_geometry_metrics: bool = False,
        max_action_space_size: Optional[int] = None,
    ):
        super().__init__()
        self.state = state
        self.basepath = basepath
        self.pop_tol = pop_tol
        self.max_steps = max_steps
        self.feature_config = feature_config
        self.include_geometry_metrics = include_geometry_metrics
        self.max_action_space_size = max_action_space_size
        self.reward_mode = reward_mode
        self.score_reward_scale = float(score_reward_scale)

        self.graph, self.partition = build_precinct_graph(state, basepath)
        self.metrics_calc = MCalc()

        if reward_fn is None:
            baseline_path = self._resolve_baseline_path()
            reward_fn = ZScoreReward(str(baseline_path))
        self.reward_fn = reward_fn
        self.reward_weights = reward_weights or build_default_reward_weights()
        self.delta_reward = delta_reward or DeltaRewardWrapper(
            scale_factor=delta_scale_factor, exploration_coef=exploration_coef
        )
        self.ema_delta_reward = ema_delta_reward or EMADeltaRewardWrapper(
            scale_factor=delta_scale_factor,
            exploration_coef=exploration_coef,
            ema_alpha=ema_alpha,
        )

        self.n_districts = len(self.partition.parts)
        self.n_precincts = len(self.graph.nodes)
        self._baseline_assignment = dict(self.partition.assignment)
        self.current_step = 0

        self._cached_graph_observation: Optional[Tuple[nx.Graph, np.ndarray]] = None
        self._partition_hash: Optional[int] = None

        self._valid_actions = generate_valid_actions(
            self.graph,
            dict(self.partition.assignment),
            self.pop_tol,
            self.n_districts,
            max_actions=self.max_action_space_size,
        )
        self._initial_action_space_size = len(self._valid_actions)
        self.action_space = spaces.Discrete(max(1, self._initial_action_space_size))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def _resolve_baseline_path(self) -> Path:
        base_path = Path(self.basepath)
        if base_path.name == self.state:
            return base_path / "baseline_stats.csv"
        return base_path / self.state / "baseline_stats.csv"

    def _get_observation(self) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)

    def get_graph_observation(self) -> Tuple[nx.Graph, np.ndarray]:
        """Return cached tuple (graph, node_features)."""
        current_partition_hash = hash(tuple(sorted(self.partition.assignment.items())))
        if self._cached_graph_observation is None or self._partition_hash != current_partition_hash:
            features = build_node_features(
                self.graph, dict(self.partition.assignment), self.n_districts, self.feature_config
            )
            self._cached_graph_observation = (self.graph, features)
            self._partition_hash = current_partition_hash
        return self._cached_graph_observation

    def get_valid_action_mask(self) -> np.ndarray:
        """Return binary mask for current valid actions."""
        return build_action_mask(self._valid_actions, self.action_space.n)

    def _distance_from_baseline(self) -> int:
        return sum(
            1
            for node in self.graph.nodes()
            if self.partition.assignment[node] != self._baseline_assignment.get(node, -1)
        )

    def _max_population_deviation(self) -> float:
        total_pop = sum(self.graph.nodes[n].get("P0010001", 0) for n in self.graph.nodes())
        ideal_pop = total_pop / self.n_districts
        pop_deviations = []
        for district_id in self.partition.parts.keys():
            district_pop = sum(
                self.graph.nodes[n].get("P0010001", 0) for n in self.partition.parts[district_id]
            )
            pop_deviations.append(abs(district_pop - ideal_pop) / ideal_pop)
        return float(max(pop_deviations) * 100 if pop_deviations else 0.0)

    def step(self, action: int):
        """Apply action and return Gymnasium 5-tuple."""
        terminated = False
        truncated = False

        if action >= len(self._valid_actions):
            info = {
                "error": "Action index out of bounds",
                "action": int(action),
                "valid_actions_count": len(self._valid_actions),
                "action_space_size": self.action_space.n,
            }
            return self._get_observation(), np.float32(-10.0), True, False, info

        node, target_district = self._valid_actions[action]
        old_district = self.partition.assignment[node]
        new_assignment = dict(self.partition.assignment)
        new_assignment[node] = target_district

        from gerrychain import Partition

        self.partition = Partition(self.graph, new_assignment, self.partition.updaters)
        self._valid_actions = update_valid_actions_incremental(
            self.graph,
            dict(self.partition.assignment),
            old_district,
            target_district,
            node,
            self._valid_actions,
            self.pop_tol,
            self.n_districts,
            self.max_action_space_size,
        )
        self._cached_graph_observation = None
        self._partition_hash = None

        metrics_df = self.metrics_calc.calculate_metrics(
            self.partition, include_geometry=self.include_geometry_metrics
        )
        metrics = metrics_df.iloc[0].to_dict()
        total_score = self.reward_fn(metrics, self.reward_weights)
        distance = self._distance_from_baseline()
        max_pop_deviation = self._max_population_deviation()
        if self.reward_mode == "score":
            reward = np.float32(self.score_reward_scale * float(total_score))
        elif self.reward_mode == "ema_delta":
            reward = np.float32(self.ema_delta_reward(total_score, distance))
        else:
            reward = np.float32(self.delta_reward(total_score, distance))

        self.current_step += 1
        no_valid_actions = self.get_valid_action_mask().sum() == 0
        terminated = bool(no_valid_actions)
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "action": int(action),
            "node": int(node),
            "old_district": int(old_district),
            "new_district": int(target_district),
            "step": int(self.current_step),
            "valid_actions": int(self.get_valid_action_mask().sum()),
            "distance_from_baseline": int(distance),
            "total_score": float(total_score),
            "efficiency_gap": float(metrics.get("EfficiencyGap", 0.0)),
            "max_pop_deviation": float(max_pop_deviation),
        }
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset env to baseline map."""
        del options
        super().reset(seed=seed)
        self.graph, self.partition = build_precinct_graph(self.state, self.basepath)
        self._baseline_assignment = dict(self.partition.assignment)
        self.current_step = 0
        self.delta_reward.reset()
        self.ema_delta_reward.reset()
        self._cached_graph_observation = None
        self._partition_hash = None
        self._valid_actions = generate_valid_actions(
            self.graph,
            dict(self.partition.assignment),
            self.pop_tol,
            self.n_districts,
            max_actions=self.max_action_space_size,
        )
        self.action_space = spaces.Discrete(max(1, self._initial_action_space_size))
        return self._get_observation(), {}

    def render(self):
        """Render summary stats."""
        print(
            f"Step={self.current_step}, Districts={self.n_districts}, ValidActions={len(self._valid_actions)}"
        )

