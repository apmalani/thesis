"""Path utility functions for project directories."""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return repository root by searching for project markers."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current.parent.parent


def get_models_dir(state: Optional[str] = None) -> Path:
    """Return model directory path and create if missing."""
    models_dir = get_project_root() / "models"
    if state:
        models_dir = models_dir / state
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_data_dir(state: Optional[str] = None, subdir: str = "processed") -> Path:
    """Return data directory path and create if missing."""
    data_dir = get_project_root() / "data" / subdir
    if state:
        data_dir = data_dir / state
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_logs_dir(state: Optional[str] = None) -> Path:
    """Return logs directory path and create if missing."""
    logs_dir = get_project_root() / "logs"
    if state:
        logs_dir = logs_dir / state
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_outputs_dir(state: Optional[str] = None, subdir: str = "best_maps") -> Path:
    """Return outputs directory path and create if missing."""
    outputs_dir = get_project_root() / "outputs" / subdir
    if state:
        outputs_dir = outputs_dir / state
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir

