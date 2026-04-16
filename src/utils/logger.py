"""
Best Legal Map Logger for redistricting training.

Tracks and saves the best legal redistricting maps found during training.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import numpy as np


def _convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalar to Python native type
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    else:
        return obj


class BestMapLogger:
    """
    Logger for tracking and saving the best legal redistricting maps.
    
    A "Best Map" is defined as:
    - Max_Population_Deviation <= 5.0% (legal)
    - Current_Step_Reward > previous best_legal_score
    - Score improvement > 0.05% (to avoid disk clutter)
    """
    
    def __init__(self, output_dir: Path, min_improvement: float = 0.0005):
        """
        Initialize the best map logger.
        
        Args:
            output_dir: Directory to save best maps (e.g., outputs/best_maps/)
            min_improvement: Minimum score improvement (0.0005 = 0.05%) to save a new map
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_legal_score = -float('inf')
        self.min_improvement = min_improvement
        self.best_map_count = 0
        
    def is_best_legal_map(
        self,
        max_pop_deviation: float,
        current_reward: float
    ) -> bool:
        """
        Check if current state qualifies as a best legal map.
        
        Args:
            max_pop_deviation: Maximum population deviation percentage
            current_reward: Current step reward
            
        Returns:
            True if this is a new best legal map
        """
        # Must be legal (Max_Dev <= 5.0%)
        if max_pop_deviation > 5.0:
            return False
        
        # Must have improved score
        score_improvement = current_reward - self.best_legal_score
        if score_improvement <= self.min_improvement:
            return False
        
        return True
    
    def save_best_map(
        self,
        partition,
        episode: int,
        step: int,
        reward: float,
        max_pop_deviation: float,
        metrics: Optional[Dict] = None
    ) -> Path:
        """
        Save the best legal map to disk.
        
        Args:
            partition: Gerrychain Partition object
            episode: Current episode number
            step: Current step number
            reward: Reward score for this map
            max_pop_deviation: Maximum population deviation percentage
            metrics: Optional dictionary of metrics (EG, Polsby-Popper, etc.)
            
        Returns:
            Path to saved CSV file
        """
        self.best_legal_score = reward
        self.best_map_count += 1
        
        # Create filename: best_map_ep[EPISODE]_step[STEP]_score[SCORE].csv
        score_str = f"{reward:.4f}".replace('.', 'p').replace('-', 'neg')
        filename = f"best_map_ep{episode}_step{step}_score{score_str}.csv"
        csv_path = self.output_dir / filename
        
        # Extract precinct -> district assignment
        assignment_dict = dict(partition.assignment)
        
        # Create DataFrame with precinct assignments
        assignment_df = pd.DataFrame([
            {
                'precinct_id': node,
                'district': district
            }
            for node, district in assignment_dict.items()
        ])
        
        # Sort by precinct_id for consistency
        assignment_df = assignment_df.sort_values('precinct_id').reset_index(drop=True)
        
        # Save CSV
        assignment_df.to_csv(csv_path, index=False)

        # Save numeric assignments (aligned to sorted precinct_id)
        assignment_array = assignment_df['district'].to_numpy(dtype=np.int64)
        npy_path = csv_path.with_suffix('.npy')
        np.save(npy_path, assignment_array)
        
        # Save metadata JSON
        metadata = {
            'episode': episode,
            'step': step,
            'reward_score': float(reward),
            'max_population_deviation': float(max_pop_deviation),
            'n_precincts': len(assignment_dict),
            'n_districts': len(partition.parts),
            'districts': sorted(list(partition.parts.keys())),
            'metrics': metrics if metrics else {}
        }
        
        # Convert numpy types to native Python types for JSON serialization
        metadata = _convert_to_native_types(metadata)
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        metadata_path = self.output_dir / metadata_filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return csv_path
    
    def get_best_score(self) -> float:
        """Get the current best legal score."""
        return self.best_legal_score
    
    def get_best_map_count(self) -> int:
        """Get the number of best maps saved."""
        return self.best_map_count

