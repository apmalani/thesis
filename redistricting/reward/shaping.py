"""Reward shaping wrappers."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DeltaRewardWrapper:
    """Compute delta score reward with optional exploration bonus."""

    scale_factor: float = 1.0
    exploration_coef: float = 0.0
    previous_score: Optional[float] = None

    def reset(self) -> None:
        """Clear previous score state."""
        self.previous_score = None

    def __call__(self, current_score: float, distance_from_baseline: int = 0) -> float:
        """Return scaled score delta plus optional exploration bonus."""
        if self.previous_score is None:
            delta = 0.0
        else:
            delta = float(current_score) - float(self.previous_score)
        self.previous_score = float(current_score)
        exploration_bonus = self.exploration_coef * float(distance_from_baseline)
        return float((delta + exploration_bonus) * self.scale_factor)


@dataclass
class EMADeltaRewardWrapper:
    """Delta vs exponential moving-average baseline of score (variance reduction)."""

    scale_factor: float = 1.0
    exploration_coef: float = 0.0
    ema_alpha: float = 0.1
    ema_baseline: Optional[float] = None

    def reset(self) -> None:
        self.ema_baseline = None

    def __call__(self, current_score: float, distance_from_baseline: int = 0) -> float:
        score = float(current_score)
        if self.ema_baseline is None:
            self.ema_baseline = score
            delta = 0.0
        else:
            delta = score - self.ema_baseline
            self.ema_baseline = (1.0 - self.ema_alpha) * self.ema_baseline + self.ema_alpha * score
        exploration_bonus = self.exploration_coef * float(distance_from_baseline)
        return float((delta + exploration_bonus) * self.scale_factor)


def build_default_reward_weights() -> Dict[str, float]:
    """Return default z-score metric weights."""
    return {
        "EfficiencyGap": 1.0,
        "SeatsVotesDiff": 1.0,
        "PolPopperAvg": 1.0,
        "MinOppAvg": 1.0,
        "PartisanProp": 1.0,
    }

