"""Z-score ensemble reward implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricDirection:
    """Direction convention for metric optimization."""

    lower_is_better: bool = False
    absolute_value: bool = False


class ZScoreReward:
    """Reward function based on weighted z-scores against ensemble baselines."""

    def __init__(self, baseline_stats_path: str):
        baseline_path = Path(baseline_stats_path)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline stats file not found: {baseline_path}")
        self.baseline_stats = pd.read_csv(baseline_path, index_col=0)
        required = {"mean", "median", "std"}
        missing = required - set(self.baseline_stats.columns)
        if missing:
            raise ValueError(f"Baseline stats missing columns: {sorted(missing)}")

        self.metric_directions: Dict[str, MetricDirection] = {
            "EfficiencyGap": MetricDirection(lower_is_better=True, absolute_value=True),
            "SeatsVotesDiff": MetricDirection(lower_is_better=True, absolute_value=False),
            "PolPopperAvg": MetricDirection(lower_is_better=False, absolute_value=False),
            "PolPopperMin": MetricDirection(lower_is_better=False, absolute_value=False),
            "MinOppAvg": MetricDirection(lower_is_better=False, absolute_value=False),
            "MinOppMin": MetricDirection(lower_is_better=False, absolute_value=False),
            "PartisanProp": MetricDirection(lower_is_better=False, absolute_value=False),
        }

    def metric_zscore(self, metric_name: str, value: float, clip: float = 3.0) -> float:
        """Compute clipped z-score for one metric."""
        if metric_name not in self.baseline_stats.index:
            return 0.0
        if not np.isfinite(value):
            return 0.0
        row = self.baseline_stats.loc[metric_name]
        center = row["median"]
        std = row["std"]
        if pd.isna(center) or pd.isna(std) or std == 0:
            return 0.0
        direction = self.metric_directions.get(metric_name, MetricDirection())
        target_value = abs(value) if direction.absolute_value else value
        if not np.isfinite(target_value):
            return 0.0
        z = (target_value - center) / std
        if not np.isfinite(z):
            return 0.0
        if direction.lower_is_better:
            z = -z
        return float(np.clip(z, -clip, clip))

    def __call__(
        self, district_metrics: Mapping[str, float], weights: Optional[Mapping[str, float]] = None
    ) -> float:
        """Return weighted z-score reward sum."""
        if weights is None:
            weights = {key: 1.0 for key in district_metrics.keys()}
        reward = 0.0
        for metric_name, weight in weights.items():
            if metric_name not in district_metrics:
                continue
            metric_value = district_metrics[metric_name]
            if metric_value is None:
                continue
            z = self.metric_zscore(metric_name, float(metric_value))
            if not np.isfinite(z):
                continue
            reward += float(weight) * z
        return float(reward)

