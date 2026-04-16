"""Best legal map persistence utilities."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _convert_to_native_types(obj):
    """Convert numpy scalar/array objects to Python-native JSON-safe values."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    return obj


class BestMapLogger:
    """Save improved legal maps and metadata during training."""

    def __init__(self, output_dir: Path, min_improvement: float = 0.0005):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_legal_score = -float("inf")
        self.min_improvement = min_improvement
        self.best_map_count = 0

    def is_best_legal_map(self, max_pop_deviation: float, current_reward: float) -> bool:
        """Return True if map is legal and improves best score."""
        if max_pop_deviation > 5.0:
            return False
        return (current_reward - self.best_legal_score) > self.min_improvement

    def save_best_map(
        self,
        partition,
        episode: int,
        step: int,
        reward: float,
        max_pop_deviation: float,
        metrics: Optional[Dict] = None,
    ) -> Path:
        """Persist precinct-to-district assignment as CSV + NPY + metadata JSON."""
        self.best_legal_score = reward
        self.best_map_count += 1

        score_str = f"{reward:.4f}".replace(".", "p").replace("-", "neg")
        filename = f"best_map_ep{episode}_step{step}_score{score_str}.csv"
        csv_path = self.output_dir / filename

        assignment_df = pd.DataFrame(
            [{"precinct_id": node, "district": district} for node, district in dict(partition.assignment).items()]
        ).sort_values("precinct_id")
        assignment_df.to_csv(csv_path, index=False)

        npy_path = csv_path.with_suffix(".npy")
        np.save(npy_path, assignment_df["district"].to_numpy(dtype=np.int64))

        metadata = {
            "episode": episode,
            "step": step,
            "reward_score": float(reward),
            "max_population_deviation": float(max_pop_deviation),
            "n_precincts": len(dict(partition.assignment)),
            "n_districts": len(partition.parts),
            "districts": sorted(list(partition.parts.keys())),
            "metrics": metrics if metrics else {},
        }
        metadata_path = self.output_dir / filename.replace(".csv", "_metadata.json")
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(_convert_to_native_types(metadata), handle, indent=2)
        return csv_path

    def get_best_score(self) -> float:
        """Return current best legal score."""
        return self.best_legal_score

    def get_best_map_count(self) -> float:
        """Return number of saved maps."""
        return self.best_map_count

