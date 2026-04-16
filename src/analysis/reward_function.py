"""
Z-Score Ensemble Reward Function for Redistricting Optimization.

Implements a Markovian reward function as a weighted sum of Z-scores:
R = Σ w_i · Z_i

Components:
- Efficiency Gap (EG)
- Partisan Proportionality
- Seats-Votes Ratio
- Polsby-Popper (Compactness)
- VRA Minority Districts
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
import os
import networkx as nx


class GerrymanderingRewardFunction:
    """
    Reward function based on Z-score ensemble of redistricting metrics.
    
    The reward is Markovian (depends only on current map state) and is computed
    as a weighted sum of standardized (Z-score) metrics.
    """
    
    def __init__(self, baseline_stats_file: str):
        """
        Initialize reward function with baseline statistics.
        
        Args:
            baseline_stats_file: Path to CSV file with baseline statistics
                (mean, median, std) for each metric
        """
        if not os.path.exists(baseline_stats_file):
            raise FileNotFoundError(
                f"Baseline stats file not found: {baseline_stats_file}\n"
                "Please ensure the file exists or run baseline statistics calculation first."
            )
        
        self.baseline_stats = pd.read_csv(baseline_stats_file, index_col=0)
        
        # Validate required columns exist
        required_cols = ['mean', 'median', 'std']
        missing_cols = [col for col in required_cols if col not in self.baseline_stats.columns]
        if missing_cols:
            raise ValueError(
                f"Baseline stats file missing required columns: {missing_cols}\n"
                f"Found columns: {list(self.baseline_stats.columns)}"
            )
    
    def standardize_metric(
        self,
        metric_name: str,
        value: float,
        use_median: bool = True
    ) -> float:
        """
        Standardize metric value to Z-score.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            use_median: If True, use median as center; else use mean
            
        Returns:
            Z-score of the metric (0.0 if metric not found or invalid)
        """
        # Safe check: if metric not in baseline stats, warn and return 0
        if metric_name not in self.baseline_stats.index:
            warnings.warn(
                f"Metric '{metric_name}' not found in baseline statistics. "
                f"Available metrics: {list(self.baseline_stats.index)}. "
                "Using Z-score of 0.0.",
                UserWarning
            )
            return 0.0
        
        try:
            stats = self.baseline_stats.loc[metric_name]
            center = stats['median'] if use_median else stats['mean']
            scale = stats['std']
            
            # Safe check: handle invalid scale values
            if pd.isna(scale) or scale == 0 or np.isnan(scale):
                warnings.warn(
                    f"Invalid standard deviation for metric '{metric_name}': {scale}. "
                    "Using Z-score of 0.0.",
                    UserWarning
                )
                return 0.0
            
            # Safe check: handle invalid center values
            if pd.isna(center) or np.isnan(center):
                warnings.warn(
                    f"Invalid center value for metric '{metric_name}': {center}. "
                    "Using Z-score of 0.0.",
                    UserWarning
                )
                return 0.0
            
            z_score = (value - center) / scale
            
            # Clip extreme values to prevent outliers from dominating
            z_score = np.clip(z_score, -3.0, 3.0)
            
            return float(z_score)
        except Exception as e:
            warnings.warn(
                f"Error standardizing metric '{metric_name}': {e}. "
                "Using Z-score of 0.0.",
                UserWarning
            )
            return 0.0
    
    def calculate_reward(
        self,
        district_metrics: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        clip_extreme: bool = True,
        max_pop_deviation: Optional[float] = None,
        skip_compactness: bool = False
    ) -> Tuple[float, bool]:
        """
        Calculate multi-objective reward for map quality.
        
        Since illegal moves are now impossible (hard action masking), reward is strictly
        the improvement in map quality: Efficiency Gap + Compactness + Minority.
        
        Total Reward = EG_Reward + Compactness_Reward + Minority_Reward
        
        Where:
        - EG_Reward: Efficiency Gap reward (scaled to [-1, 1])
        - Compactness_Reward: Polsby-Popper scores * 5 (higher is better)
        - Minority_Reward: VRA compliance * 2 (higher is better)
        
        Args:
            district_metrics: Dictionary of current metric values
            weights: Dictionary of weights for each component (kept for compatibility, not used)
            clip_extreme: Whether to clip extreme Z-scores (kept for compatibility, not used)
            max_pop_deviation: Maximum population deviation percentage (kept for compatibility, not used)
            
        Returns:
            Tuple of (reward value, should_end_episode)
            - reward: Reward value (scalar, clipped to [-10, 10])
            - should_end_episode: Always False (illegal actions are prevented, not penalized)
        """
        # SIMPLIFIED REWARD: Since illegal moves are now impossible (hard action masking),
        # reward is strictly the improvement in map quality (Compactness + Efficiency Gap)
        # No penalties needed - illegal actions are prevented, not penalized
        
        total_reward = 0.0
        
        # Note: Weights are not used in the new reward function
        # Each component is already scaled appropriately:
        # - Efficiency Gap: scaled to [-1, 1]
        # - Compactness: Polsby-Popper * 5
        # - Minority: MinOppAvg * 2
        # Final total is clipped to [-10, 10]
        if weights is not None:
            # Weights parameter kept for compatibility but not used
            pass
        
        # 2. EFFICIENCY GAP REWARD (scaled to [-1, 1] - REDUCED BY HALF)
        # Reward = (1.0 - abs(current_EG)) * 1.0 (was 2.0)
        # Reduced weight to prevent agent from 'sacrificing' population for partisan gains
        if 'EfficiencyGap' in district_metrics:
            eg_value = district_metrics['EfficiencyGap']
            # Scale to [-1, 1] range (reduced from [-2, 2])
            eg_reward = (1.0 - abs(eg_value)) * 1.0
            # Clip to ensure it stays in range
            eg_reward = np.clip(eg_reward, -1.0, 1.0)
            total_reward += eg_reward
        else:
            # Default to 0 if EG not available
            total_reward += 0.0
        
        # 3. COMPACTNESS REWARD (Polsby-Popper * 5)
        # Higher Polsby-Popper is better (more compact/round districts)
        if not skip_compactness:
            if 'PolPopperAvg' in district_metrics:
                pp_avg = district_metrics['PolPopperAvg']
                # Reward = Average_Polsby_Popper_Score * 5
                # Typical PP scores range 0.0-0.5, so this gives rewards 0-2.5
                compactness_reward = pp_avg * 5.0
                total_reward += compactness_reward
            else:
                total_reward += 0.0
        
        # 4. MINORITY REWARD (VRA compliance - optional, scaled similarly)
        # Keep this simple for now - can be enhanced later
        minority_reward = 0.0
        if 'MinOppAvg' in district_metrics:
            min_opp_avg = district_metrics.get('MinOppAvg', 0.0)
            # Scale minority opportunity to reasonable range
            minority_reward = min_opp_avg * 2.0  # Scale to roughly [0, 2]
            total_reward += minority_reward
        
        # Final clipping to ensure reward stays in sane range [-10, +10]
        # Since illegal actions are now prevented, all maps are legal
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        # Return reward and False (no episode end needed - illegal actions are prevented)
        return float(total_reward), False
    